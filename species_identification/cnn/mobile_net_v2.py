"""
MobileNetV2 fine-tuning for La Jolla Cove species classification.

Upgrade from MobileNetV3-Small (51-58% test top-1) for better fine-grained
discrimination. About 3x the params, 2-3x slower inference on the Uno Q.
Still comfortably real-time with TTA.

Three new techniques vs the MobileNetV3 training:
    - Backbone: MobileNetV2 instead of V3
    - Calibration phase: final few epochs with label_smoothing=0 and no
      MixUp/CutMix, so the model learns to produce confident predictions
      on clean inputs.
    - Temperature scaling: after training, fit a single scalar T on
      validation logits. Divide logits by T at inference to correct
      for under/over-confidence. Saved to outputs/temperature.json for
      the inference script to apply.

Dataset layout expected:
    data/
        species_a/ img001.jpg img002.jpg ...
        species_b/ img001.jpg ...
        ...  (41 folders total)

Pipeline:
    1. Load folder-per-class data, split 70/15/15 train/val/test (stratified).
    2. Heavy augmentation on train split.
    3. Phase 1: freeze backbone, train classifier head (~10 epochs).
    4. Phase 2: unfreeze top of backbone, fine-tune at low LR (~20-30 epochs).
    5. Phase 3 (NEW): calibration — few epochs with no label smoothing,
       no MixUp/CutMix, very low LR. Sharpens confidence without hurting
       accuracy.
    6. Fit temperature scaling on val split.
    7. Evaluate on held-out test split.
    8. Save Keras model + TFLite (float32 and INT8) for Uno Q deployment.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # forces tf.keras to be the standalone Keras package, not tf's built-in

import json
import pathlib
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Config — tune these
# ---------------------------------------------------------------------------
DATA_DIR        = "../ucsd-data"       # folder-per-class root
OUTPUT_DIR      = "../outputs"         # where models + logs go
IMG_SIZE        = 320                  # MobileNetV2 standard input
BATCH_SIZE      = 16                   
SEED            = 42

# Split ratios
VAL_FRACTION    = 0.15
TEST_FRACTION   = 0.15

# Training schedule
EPOCHS_HEAD     = 10                   # phase 1: frozen backbone
EPOCHS_FT       = 25                   # phase 2: fine-tune top blocks
EPOCHS_CALIB    = 5                    # phase 3: confidence calibration (NEW)
LR_HEAD         = 1e-3
LR_FT           = 1e-4
LR_CALIB        = 1e-5                 # very small — we're polishing, not learning
UNFREEZE_FROM   = 90                   # MobileNetV2 has ~150 layers;
                                       # 90+ ≈ last few blocks, proportional
                                       # to the 100/155 we used for Small.

# Regularization
LABEL_SMOOTH    = 0.1
DROPOUT         = 0.3
WEIGHT_DECAY    = 1e-4

# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Stratified 70/15/15 split by symlinking into split folders.
#    Keras' image_dataset_from_directory doesn't do stratified splits, so
#    we build the splits ourselves to guarantee every class appears in
#    val and test even with only 20 images per class.
# ---------------------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(folder):
    return sorted([p for p in pathlib.Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _find_class_dirs(data_dir):
    """Find class folders under data_dir, handling two common layouts:
        (A) data_dir/<species>/*.jpg                 (flat, what we want)
        (B) data_dir/{train,val,test}/<species>/*.jpg (pre-split)
    In case (B), we flatten by using the 'train' folder as the class source
    and let our own splitter partition it — ignoring the original split.
    """
    data_dir = pathlib.Path(data_dir)
    top = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    names = {d.name.lower() for d in top}

    # Case B: pre-split dataset. Use whichever split folder has the most data.
    split_names = {"train", "val", "valid", "validation", "test"}
    if names & split_names:
        candidates = [d for d in top if d.name.lower() in split_names]
        best = max(candidates,
                   key=lambda d: sum(len(_list_images(c))
                                     for c in d.iterdir() if c.is_dir()))
        print(f"Detected pre-split dataset. Using '{best.name}/' "
              f"as the source and re-splitting from scratch.")
        return sorted([c for c in best.iterdir() if c.is_dir()])

    # Case A: flat, one folder per class.
    return top


def build_splits(data_dir, out_root="_splits"):
    out_root = pathlib.Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)

    class_dirs = _find_class_dirs(data_dir)
    # Drop empty folders and anything without real images.
    class_dirs = [d for d in class_dirs if len(_list_images(d)) > 0]
    if not class_dirs:
        raise ValueError(
            f"No class folders with images found under '{data_dir}'. "
            "Expected layout: data/<species_name>/*.jpg"
        )
    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} class folders.")

    for split in ("train", "val", "test"):
        for cname in class_names:
            (out_root / split / cname).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    skipped = []
    for cdir in class_dirs:
        imgs = _list_images(cdir)
        random.Random(SEED).shuffle(imgs)
        n = len(imgs)
        n_test = max(1, int(round(n * TEST_FRACTION)))
        n_val  = max(1, int(round(n * VAL_FRACTION)))
        n_train = n - n_val - n_test
        if n_train < 1:
            skipped.append((cdir.name, n))
            continue

        splits = (("train", imgs[:n_train]),
                  ("val",   imgs[n_train:n_train + n_val]),
                  ("test",  imgs[n_train + n_val:]))
        for split_name, split_imgs in splits:
            for img_path in split_imgs:
                dst = out_root / split_name / cdir.name / img_path.name
                # Symlink so we don't duplicate the dataset on disk.
                # Fall back to copy on systems where symlinks aren't allowed.
                try:
                    dst.symlink_to(img_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst)
                counts[split_name] += 1

    print(f"Split totals: {counts}")
    if skipped:
        print(f"Skipped {len(skipped)} classes with too few images: {skipped}")
        skipped_names = {name for name, _ in skipped}
        class_names = [c for c in class_names if c not in skipped_names]
        for split in ("train", "val", "test"):
            for cname in skipped_names:
                empty = out_root / split / cname
                if empty.exists():
                    shutil.rmtree(empty)
    print(f"Classes used: {len(class_names)}")
    return out_root, class_names


# ---------------------------------------------------------------------------
# 2. Datasets
# ---------------------------------------------------------------------------
def make_datasets(split_root, class_names):
    train_ds = keras.utils.image_dataset_from_directory(
        split_root / "train",
        labels="inferred",
        label_mode="categorical",   # one-hot, matches CategoricalCrossentropy
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        split_root / "val",
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        split_root / "test",
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache()
    # MixUp/CutMix is attached by train() after we know num_classes.
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 3. Augmentation — two levels:
#    A) spatial/color aug baked into the model as Keras layers (always on
#       during training, no-op at inference).
#    B) MixUp + CutMix applied in the tf.data pipeline because they mix
#       pairs of images with their labels — labels must be available, so
#       these can't live inside the model's forward pass.
# ---------------------------------------------------------------------------
def build_augmentation():
    """Per-image spatial and color augmentation. Stronger than the first
    iteration — uses RandAugment if available (keras_cv), otherwise falls
    back to a tuned hand-rolled stack."""

    # Try keras_cv RandAugment first — it's a strong, well-tested
    # augmentation policy for image classification.
    try:
        import keras_cv
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            keras_cv.layers.RandAugment(
                value_range=(0, 255),
                augmentations_per_image=2,
                magnitude=0.4,
            ),
            layers.RandomZoom(0.15),
        ], name="augment")
    except ImportError:
        pass

    # Fallback: stronger version of the hand-rolled stack from iteration 1.
    # Increased magnitudes across the board vs. the original.
    layer_list = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),                # was 0.08, now ~±22°
        layers.RandomZoom(0.2),                     # was 0.15
        layers.RandomTranslation(0.15, 0.15),       # was 0.1/0.1
        layers.RandomContrast(0.3),                 # was 0.2
        layers.RandomBrightness(0.3, value_range=(0, 255)),  # was 0.2
    ]
    if hasattr(layers, "RandomColorJitter"):
        layer_list.append(
            layers.RandomColorJitter(
                value_range=(0, 255),
                brightness_factor=0.0,
                contrast_factor=0.0,
                saturation_factor=(0.7, 1.3),       # was (0.8, 1.2)
                hue_factor=0.1,                     # was 0.05
            )
        )
    elif hasattr(layers, "RandomHue"):
        layer_list += [
            layers.RandomHue(factor=0.1, value_range=(0, 255)),
            layers.RandomSaturation(factor=(0.7, 1.3), value_range=(0, 255)),
        ]
    return keras.Sequential(layer_list, name="augment")


def apply_mixup_cutmix(ds, num_classes, alpha=0.2, prob=0.5):
    """Apply MixUp *or* CutMix to each training batch.

    MixUp: linearly interpolates two images and their one-hot labels,
    yielding a 'half-this-half-that' training signal. Unreasonably
    effective regularizer on small datasets.

    CutMix: replaces a rectangular patch of one image with the same region
    from another image; labels are mixed proportionally to patch area.
    Complements MixUp well because it preserves local texture.

    We randomly pick one of the two per batch.
    """
    def _mixup(imgs, labels):
        batch_size = tf.shape(imgs)[0]
        lam = tf.random.uniform([], 0.0, 1.0)
        lam = tf.maximum(lam, 1.0 - lam)            # bias toward identity
        idx = tf.random.shuffle(tf.range(batch_size))
        imgs2 = tf.gather(imgs, idx)
        labels2 = tf.gather(labels, idx)
        imgs = lam * imgs + (1.0 - lam) * imgs2
        labels = lam * labels + (1.0 - lam) * labels2
        return imgs, labels

    def _cutmix(imgs, labels):
        batch_size = tf.shape(imgs)[0]
        h, w = IMG_SIZE, IMG_SIZE
        lam = tf.random.uniform([], 0.3, 0.7)       # patch covers 30–70%
        cut_h = tf.cast(tf.cast(h, tf.float32) *
                        tf.sqrt(1.0 - lam), tf.int32)
        cut_w = tf.cast(tf.cast(w, tf.float32) *
                        tf.sqrt(1.0 - lam), tf.int32)
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
        y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
        x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

        idx = tf.random.shuffle(tf.range(batch_size))
        imgs2 = tf.gather(imgs, idx)
        labels2 = tf.gather(labels, idx)

        # Build a binary mask of the patch and splice.
        mask_y = tf.logical_and(
            tf.range(h)[:, None] >= y1, tf.range(h)[:, None] < y2)
        mask_x = tf.logical_and(
            tf.range(w)[None, :] >= x1, tf.range(w)[None, :] < x2)
        mask = tf.cast(tf.logical_and(mask_y, mask_x),
                       imgs.dtype)[None, :, :, None]
        imgs = imgs * (1.0 - mask) + imgs2 * mask

        actual_lam = 1.0 - tf.cast((y2 - y1) * (x2 - x1),
                                   tf.float32) / float(h * w)
        labels = actual_lam * labels + (1.0 - actual_lam) * labels2
        return imgs, labels

    def _maybe_mix(imgs, labels):
        r = tf.random.uniform([])
        # With probability `prob`, apply MixUp or CutMix (50/50 split).
        # Otherwise pass through unchanged.
        return tf.cond(
            r < prob * 0.5,
            lambda: _mixup(imgs, labels),
            lambda: tf.cond(
                r < prob,
                lambda: _cutmix(imgs, labels),
                lambda: (imgs, labels),
            ),
        )

    return ds.map(_maybe_mix, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 4. Model — MobileNetV2 with ImageNet weights + custom head.
#    IMPORTANT: Keras' MobileNetV3 already includes a Rescaling layer
#    internally, so we feed it raw [0, 255] pixel values. Do NOT normalize
#    manually — doing so halves accuracy.
# ---------------------------------------------------------------------------
def build_model(num_classes):
    # 1. Instantiate MobileNetV2
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    # 2. Build the dense head directly on top
    x = backbone.output
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(
        num_classes,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x) 

    model = keras.Model(inputs=backbone.input, outputs=outputs, name="mobilenetv2_lajolla")
    return model, backbone

# ---------------------------------------------------------------------------
# Sanity checks to run BEFORE the full retrain (5 minutes total).
#
# 1. Confirm the model builds at 320x320 and ImageNet weights load:
# ---------------------------------------------------------------------------
def smoke_test_build():
    """Quick check that everything wires up at the new resolution."""
 
    # Just build the model, don't train.
    aug = keras.Sequential([layers.RandomFlip("horizontal")], name="augment")
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug(inputs)
    backbone = keras.applications.MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
        pooling="avg",
    )
    x = backbone(x, training=False)
    outputs = layers.Dense(51)(x)
    model = keras.Model(inputs, outputs)
 
    # Pass a dummy batch through to verify shapes flow.
    dummy = tf.random.uniform([2, IMG_SIZE, IMG_SIZE, 3],
                              minval=0, maxval=255)
    out = model(dummy, training=False)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {model.count_params():,}")
    assert out.shape == (2, 51), "output shape mismatch"
    print("OK: model builds and runs at 320x320.")
 
    # Approximate the per-image FLOPs increase. MobileNetV2 at 224
    # is ~0.22 GFLOPs. At 320 it's ~(320/224)^2 ≈ 2.04x = ~0.45 GFLOPs.
    # On your Uno Q's Cortex-A53, expect inference latency to roughly double.
    ratio = (IMG_SIZE / 224.0) ** 2
    print(f"FLOPs ratio vs 224x224: {ratio:.2f}x")
 
 
# ---------------------------------------------------------------------------
# 2. Confirm a single batch loads at the new resolution.
# ---------------------------------------------------------------------------
def smoke_test_data():
    """Peek at one training batch to confirm image shape and label shape."""
    split_root, class_names = build_splits(DATA_DIR)
    train_ds, val_ds, test_ds = make_datasets(split_root, class_names)
    for imgs, labels in train_ds.take(1):
        print(f"Batch images shape: {imgs.shape}")  # expect (16, 320, 320, 3)
        print(f"Batch labels shape: {labels.shape}")  # expect (16, num_classes)
        print(f"Image dtype: {imgs.dtype}, range: "
              f"[{tf.reduce_min(imgs).numpy():.1f}, "
              f"{tf.reduce_max(imgs).numpy():.1f}]")
        assert imgs.shape[1:3] == (IMG_SIZE, IMG_SIZE), "wrong input size"
        print("OK: data loads at 320x320.")
        break


# ---------------------------------------------------------------------------
# 5. Training
# 
# Experiment: Compute per-class weights from the training split.
#    Call this AFTER build_splits() and BEFORE the training loop.
# ---------------------------------------------------------------------------
def compute_class_weights(split_root, class_names, beta=0.9999):
    """Effective-number-of-samples class weighting (Cui et al., CVPR 2019).
 
        w_c = (1 - beta) / (1 - beta^n_c)
 
    where n_c is the count of class c in the training split. As n_c grows,
    the weight saturates — this is what makes it more stable than 1/n_c on
    small datasets.
 
    beta controls how aggressive the reweighting is:
        beta = 0     -> uniform weights (no reweighting)
        beta = 0.9   -> mild reweighting
        beta = 0.99  -> moderate
        beta = 0.999 -> strong (recommended start)
        beta = 0.9999 -> very strong; good for our imbalance ratios
 
    For your dataset: Junco has 27 test examples (so ~125 train), most
    species have ~12-13 test (~60 train), and seashore has 139 test
    (~650 train). That's a ~10x imbalance. beta=0.9999 reduces it to
    roughly 2x effective weight after reweighting — enough to help,
    not so much that minority classes dominate.
 
    The weights are normalized so the mean weight is 1.0, which keeps
    the effective learning rate roughly unchanged.
    """
    import pathlib
 
    train_root = pathlib.Path(split_root) / "train"
    counts = np.array([
        len(list((train_root / cname).iterdir()))
        for cname in class_names
    ], dtype=np.float64)
 
    print(f"\nTrain split class counts:")
    print(f"  min:    {int(counts.min())} ({class_names[counts.argmin()]})")
    print(f"  max:    {int(counts.max())} ({class_names[counts.argmax()]})")
    print(f"  median: {int(np.median(counts))}")
    print(f"  ratio:  {counts.max() / counts.min():.1f}x")
 
    # Effective number of samples per class.
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-12)
 
    # Normalize so mean weight = 1.0. Keeps the loss magnitude comparable
    # to the unweighted version, so existing learning rates still work.
    weights = weights / weights.mean()
 
    print(f"\nClass weights (beta={beta}):")
    print(f"  min:    {weights.min():.3f} ({class_names[weights.argmin()]})")
    print(f"  max:    {weights.max():.3f} ({class_names[weights.argmax()]})")
    print(f"  ratio:  {weights.max() / weights.min():.2f}x "
          f"(was {counts.max() / counts.min():.1f}x before reweighting)")
    return weights.astype(np.float32)
 
 
# ---------------------------------------------------------------------------
# Experiment: Attach per-example sample weights to the dataset.
#    Apply AFTER MixUp/CutMix so the mixed labels are accounted for
#    proportionally. Sample weight for a mixed example is the weighted
#    average of its two source classes' weights, which falls out naturally
#    from a dot product with the soft label.
# ---------------------------------------------------------------------------
def attach_sample_weights(ds, class_weights):
    """Map a (images, labels) dataset to (images, labels, sample_weights).
 
    For one-hot labels: sample_weight = dot(label, class_weights).
        Pure class c -> weight = class_weights[c].
    For MixUp/CutMix soft labels (e.g. 0.7 of class a + 0.3 of class b):
        sample_weight = 0.7 * w_a + 0.3 * w_b.
    This is the right thing — a mixed example contributes proportionally
    to whichever classes it represents.
 
    Keras' loss accepts sample_weight per example; it multiplies the
    per-example loss by this scalar before reducing.
    """
    weights_tensor = tf.constant(class_weights, dtype=tf.float32)
 
    def _attach(imgs, labels):
        # labels shape: (batch, num_classes). Dot with weights -> (batch,).
        sw = tf.reduce_sum(labels * weights_tensor[None, :], axis=1)
        return imgs, labels, sw
 
    return ds.map(_attach, num_parallel_calls=tf.data.AUTOTUNE)


def compile_model(model, lr, num_classes, label_smoothing=None):
    """Compile with configurable label smoothing.

    When label_smoothing is None, use the global LABEL_SMOOTH. The calibration
    phase passes 0.0 to get sharp, confident predictions.
    """
    ls = LABEL_SMOOTH if label_smoothing is None else label_smoothing
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        # CategoricalCrossentropy (dense/one-hot) is the only variant that
        # accepts label_smoothing. Sparse labels are one-hotted in the
        # dataset pipeline below.
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=ls
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="top1"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def fit_temperature(model, val_ds, grid=None):
    """Find the scalar T that minimizes NLL on the validation set.

    The model outputs logits; we want to scale them by 1/T before softmax
    so reported probabilities match empirical accuracy. T < 1 sharpens
    (higher confidence on top prediction); T > 1 softens.

    Uses a simple grid search — overkill would be gradient descent on T,
    but a single scalar fit to ~15% of ~3000 images takes a fraction of
    a second and gives the same answer.
    """
    if grid is None:
        # Covers strong sharpening (T=0.25) through strong softening (T=3.0).
        grid = np.concatenate([np.linspace(0.25, 1.0, 16),
                               np.linspace(1.0, 3.0, 21)[1:]])

    # Collect all validation logits and labels in one pass.
    all_logits = []
    all_labels = []
    for imgs, labels in val_ds:
        logits = model(imgs, training=False).numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def nll(T):
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        # Labels are one-hot; pick each example's true-class probability.
        p_true = np.sum(probs * labels, axis=1)
        # Clip to avoid log(0).
        p_true = np.clip(p_true, 1e-12, 1.0)
        return -np.mean(np.log(p_true))

    scores = [(T, nll(T)) for T in grid]
    best_T, best_nll = min(scores, key=lambda x: x[1])
    print(f"Temperature scaling: best T = {best_T:.3f} (val NLL = {best_nll:.4f})")
    return float(best_T)


def train():
    split_root, class_names = build_splits(DATA_DIR)
    num_classes = len(class_names)
    train_ds, val_ds, test_ds = make_datasets(split_root, class_names)
 
    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
 
    augment = build_augmentation()
 
    # 1. Compile the augmentation and math into a fast graph
    @tf.function
    def process_train(x, y):
        x = augment(x, training=True)
        # Fast, native V2 preprocessing (scales to [-1.0, 1.0])
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        return x, y

    @tf.function
    def process_val_test(x, y):
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        return x, y

    # 2. Apply the compiled functions
    aug_train_ds = train_ds.map(process_train, num_parallel_calls=tf.data.AUTOTUNE)
    clean_train_ds = aug_train_ds.prefetch(tf.data.AUTOTUNE)
 
    train_ds = apply_mixup_cutmix(aug_train_ds, num_classes, alpha=0.2, prob=0.5)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(process_val_test, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(process_val_test, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # Build the flat model
    model, backbone = build_model(num_classes)
    model.summary()
 
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best.keras"),
            monitor="val_top1", mode="max", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_top1", mode="max",
            patience=8, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_top1", mode="max",
            factor=0.5, patience=4, min_lr=1e-6,
        ),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "history.csv")),
    ]
 
    # -------- Phase 1: head-only --------
    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    compile_model(model, LR_HEAD, num_classes)
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_HEAD, callbacks=callbacks)
 
    # -------- Phase 2: fine-tune top of backbone --------
    print("\n=== Phase 2: fine-tuning top of backbone ===")
    backbone.trainable = True
    for layer in backbone.layers[:UNFREEZE_FROM]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
 
    compile_model(model, LR_FT, num_classes)
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FT, callbacks=callbacks)
 
    # -------- Phase 3: confidence calibration --------
    print("\n=== Phase 3: confidence calibration (no label smoothing) ===")
    calib_callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best.keras"),
            monitor="val_top1", mode="max", save_best_only=True,
        ),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "history_calib.csv")),
    ]
    compile_model(model, LR_CALIB, num_classes, label_smoothing=0.0)
    model.fit(clean_train_ds, validation_data=val_ds,
              epochs=EPOCHS_CALIB, callbacks=calib_callbacks)

    # -------- Phase 4: Quantization-Aware Training (QAT) --------
    print("\n=== Phase 4: Quantization-Aware Training (QAT) ===")
    
    # Wrap the entire fully-trained FP32 model with fake quantization nodes
    qat_model = tfmot.quantization.keras.quantize_model(model)
    
    # Recompile the model. Use a very low learning rate so we don't destroy
    # the learned features, we just want to adapt them to INT8.
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.0
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="top1"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    
    qat_callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "qat_best.keras"),
            monitor="val_top1", mode="max", save_best_only=True,
        ),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "history_qat.csv")),
    ]
    
    # 3 to 5 epochs is usually plenty for the weights to adapt
    qat_model.fit(clean_train_ds, validation_data=val_ds,
                  epochs=3, callbacks=qat_callbacks)
                  
    # Override the main model with the QAT model for TFLite export
    model = qat_model
 
    # -------- Temperature scaling --------
    print("\n=== Fitting temperature scaling on validation split ===")
    T = fit_temperature(model, val_ds)
    with open(os.path.join(OUTPUT_DIR, "temperature.json"), "w") as f:
        json.dump({"temperature": T}, f, indent=2)
 
    # -------- Evaluate --------
    print("\n=== Test evaluation ===")
    results = model.evaluate(test_ds, return_dict=True)
    print(results)
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
 
    model.save(os.path.join(OUTPUT_DIR, "final.keras"))
    return model, test_ds, class_names


# ---------------------------------------------------------------------------
# 6. TFLite export — what you actually deploy to the Uno Q.
# ---------------------------------------------------------------------------
def export_tflite(model, test_ds):
    print("\n=== Exporting TFLite Models ===")
    
    # 1. Float32 Export (Baseline)
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()
    with open(os.path.join(OUTPUT_DIR, "model_fp32.tflite"), "wb") as f:
        f.write(tflite_fp32)
    print(f"[info] Saved: model_fp32.tflite ({len(tflite_fp32)/1e6:.2f} MB)")

    # 2. INT8 Quantized Export
    # The converter strictly requires a generator yielding representative data
    # to calibrate the activation ranges for the input/output tensors.
    def representative_dataset_gen():
        # Take 10 batches from the pre-processed test dataset
        for imgs, _labels in test_ds.take(10):
            # The TFLite converter expects a list containing the input tensor(s)
            # The images are already cast to float32 and scaled to [-1.0, 1.0]
            yield [imgs]

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset_gen
    
    # Enforce full integer quantization for edge compatibility
    converter_int8.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    
    # Use standard int8 (-128 to 127) to support MobileNetV2's [-1.0, 1.0] range
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    
    tflite_int8 = converter_int8.convert()
    with open(os.path.join(OUTPUT_DIR, "model_int8_qat.tflite"), "wb") as f:
        f.write(tflite_int8)
    print(f"[info] Saved: model_int8_qat.tflite ({len(tflite_int8)/1e6:.2f} MB)")


if __name__ == "__main__":
    # Run smoke tests first, then training.
    # print("=" * 60)
    # print("Smoke test 1: model build at 320x320")
    # print("=" * 60)
    # smoke_test_build()
 
    # print("\n" + "=" * 60)
    # print("Smoke test 2: data loads at 320x320")
    # print("=" * 60)
    # smoke_test_data()
 
    # print("\n" + "=" * 60)
    # print("Smoke tests passed. Training would start here.")
    # print("=" * 60)

    model, test_ds, class_names = train()
    export_tflite(model, test_ds)