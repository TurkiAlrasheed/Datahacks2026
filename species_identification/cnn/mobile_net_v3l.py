"""
MobileNetV3-Large fine-tuning for La Jolla Cove species classification.

Upgrade from MobileNetV3-Small (51-58% test top-1) for better fine-grained
discrimination. About 3x the params, 2-3x slower inference on the Uno Q.
Still comfortably real-time with TTA.

Three new techniques vs the MobileNetV3-Small training:
    - Backbone: MobileNetV3-Large instead of Small
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
import json
import pathlib
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Config — tune these
# ---------------------------------------------------------------------------
DATA_DIR        = "../ucsd-data"               # folder-per-class root
OUTPUT_DIR      = "../outputs"            # where models + logs go
IMG_SIZE        = 224                  # MobileNetV3-Small standard input
BATCH_SIZE      = 32                   # CPU-friendly; drop to 16 if RAM-tight
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
UNFREEZE_FROM   = 130                  # MobileNetV3-Large has ~260 layers;
                                       # 130+ ≈ last few blocks, proportional
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
# 4. Model — MobileNetV3-Small with ImageNet weights + custom head.
#    IMPORTANT: Keras' MobileNetV3 already includes a Rescaling layer
#    internally, so we feed it raw [0, 255] pixel values. Do NOT normalize
#    manually — doing so halves accuracy.
# ---------------------------------------------------------------------------
def build_model(num_classes, augment):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)

    backbone = keras.applications.MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,   # handles [0,255] -> [-1,1] internally
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(
        num_classes,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)  # logits — softmax folded into the loss below

    model = keras.Model(inputs, outputs, name="mobilenetv3large_lajolla")
    return model, backbone


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
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

    # Save class-name mapping — you'll need this at inference time on the Uno Q.
    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # Keep a clean training dataset (no MixUp/CutMix) for the calibration
    # phase. Calibration needs honest, hard labels — mixed labels defeat
    # the purpose of training the model to be confident.
    clean_train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Attach MixUp/CutMix to the main training pipeline only. Not applied
    # to val/test because those need clean labels for honest evaluation.
    train_ds = apply_mixup_cutmix(train_ds, num_classes,
                                  alpha=0.2, prob=0.5)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    augment = build_augmentation()
    model, backbone = build_model(num_classes, augment)
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
    # BatchNorm layers should usually stay in inference mode when fine-tuning
    # on tiny datasets — their running stats are better than what you'd
    # estimate from small training batches.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, LR_FT, num_classes)
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FT, callbacks=callbacks)

    # -------- Phase 3: confidence calibration --------
    # Final few epochs with no label smoothing and no MixUp/CutMix.
    # Model has already learned to classify; here it learns to be
    # confident on clean examples. Very low LR so accuracy doesn't drift.
    print("\n=== Phase 3: confidence calibration (no label smoothing) ===")
    # Remove MixUp/CutMix stochasticity from checkpointing — we want the
    # calibration-phase best, not the earlier phase's best.
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
    # Float32 — easy baseline, larger file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    with open(os.path.join(OUTPUT_DIR, "model_fp32.tflite"), "wb") as f:
        f.write(tflite_fp32)

    # INT8 quantized — 3-4x smaller, 2-4x faster on ARM CPU.
    def rep_dataset():
        # ~100 representative samples for calibration.
        count = 0
        for imgs, _ in test_ds.unbatch().batch(1):
            yield [tf.cast(imgs, tf.float32)]
            count += 1
            if count >= 100:
                break

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_int8 = converter.convert()
    with open(os.path.join(OUTPUT_DIR, "model_int8.tflite"), "wb") as f:
        f.write(tflite_int8)

    print(f"Saved: model_fp32.tflite ({len(tflite_fp32)/1e6:.2f} MB)")
    print(f"Saved: model_int8.tflite ({len(tflite_int8)/1e6:.2f} MB)")


if __name__ == "__main__":
    model, test_ds, class_names = train()
    export_tflite(model, test_ds)