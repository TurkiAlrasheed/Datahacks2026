"""
MobileNetV3-Small fine-tuning for La Jolla Cove species classification.

Dataset layout expected:
    data/
        species_a/ img001.jpg img002.jpg ...
        species_b/ img001.jpg ...
        ...  (41 folders total)

Pipeline:
    1. Load folder-per-class data, split 70/15/15 train/val/test (stratified).
    2. Heavy augmentation on train split (critical at ~20 imgs/class).
    3. Phase 1: freeze backbone, train classifier head (~10 epochs).
    4. Phase 2: unfreeze top of backbone, fine-tune at low LR (~20-30 epochs).
    5. Evaluate on held-out test split.
    6. Save Keras model + TFLite (float32 and INT8) for Uno Q deployment.
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
DATA_DIR        = "data"               # folder-per-class root
OUTPUT_DIR      = "outputs"            # where models + logs go
IMG_SIZE        = 224                  # MobileNetV3-Small standard input
BATCH_SIZE      = 32                   # CPU-friendly; drop to 16 if RAM-tight
SEED            = 42

# Split ratios
VAL_FRACTION    = 0.15
TEST_FRACTION   = 0.15

# Training schedule
EPOCHS_HEAD     = 10                   # phase 1: frozen backbone
EPOCHS_FT       = 25                   # phase 2: fine-tune top blocks
LR_HEAD         = 1e-3
LR_FT           = 1e-4
UNFREEZE_FROM   = 100                  # unfreeze layers from this index onward
                                       # (MobileNetV3-Small has ~155 layers;
                                       # 100+ = roughly the last two blocks)

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
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 3. Augmentation — heavy, because 20 imgs/class.
#    Applied as a layer block so it runs on-device during training and is
#    automatically turned off at inference.
# ---------------------------------------------------------------------------
def build_augmentation():
    layer_list = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),                # ~±15°
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ]
    # RandomColorJitter handles hue + saturation and serializes cleanly
    # (unlike tf.image.random_* wrapped in a Lambda). Available in Keras 3.3+.
    if hasattr(layers, "RandomColorJitter"):
        layer_list.append(
            layers.RandomColorJitter(
                value_range=(0, 255),
                brightness_factor=0.0,   # already handled above
                contrast_factor=0.0,     # already handled above
                saturation_factor=(0.8, 1.2),
                hue_factor=0.05,
            )
        )
    elif hasattr(layers, "RandomHue"):
        # Fallback for Keras versions with separate hue/saturation layers
        layer_list += [
            layers.RandomHue(factor=0.05, value_range=(0, 255)),
            layers.RandomSaturation(factor=(0.8, 1.2), value_range=(0, 255)),
        ]
    # If neither exists on this Keras version, silently skip color-cast
    # augmentation rather than pulling in a Lambda layer.
    return keras.Sequential(layer_list, name="augment")


# ---------------------------------------------------------------------------
# 4. Model — MobileNetV3-Small with ImageNet weights + custom head.
#    IMPORTANT: Keras' MobileNetV3 already includes a Rescaling layer
#    internally, so we feed it raw [0, 255] pixel values. Do NOT normalize
#    manually — doing so halves accuracy.
# ---------------------------------------------------------------------------
def build_model(num_classes, augment):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)

    backbone = keras.applications.MobileNetV3Small(
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

    model = keras.Model(inputs, outputs, name="mobilenetv3small_lajolla")
    return model, backbone


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
def compile_model(model, lr, num_classes):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        # CategoricalCrossentropy (dense/one-hot) is the only variant that
        # accepts label_smoothing. Sparse labels are one-hotted in the
        # dataset pipeline below.
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=LABEL_SMOOTH
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="top1"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )


def train():
    split_root, class_names = build_splits(DATA_DIR)
    num_classes = len(class_names)
    train_ds, val_ds, test_ds = make_datasets(split_root, class_names)

    # Save class-name mapping — you'll need this at inference time on the Uno Q.
    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

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
    # estimate from 14 images per class.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, LR_FT, num_classes)
    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS_FT, callbacks=callbacks)

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