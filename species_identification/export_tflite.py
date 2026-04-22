"""
Standalone: convert a trained Keras model to TFLite (float32 + INT8).

Rebuilds the model architecture from scratch and loads weights from
best.keras — this sidesteps broken Lambda-layer deserialization, which
affects older checkpoints after the training script was updated.

Run on your laptop in the project folder. Produces:
    outputs/model_fp32.tflite
    outputs/model_int8.tflite   <- deploy this one to the Uno Q
"""

import json
import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Config — must match train_mobilenetv3.py
# ---------------------------------------------------------------------------
WEIGHTS_PATH  = "outputs/best.keras"       # the trained file
CLASSES_PATH  = "outputs/class_names.json" # saved during training
SPLITS_ROOT   = "cnn/_splits"                  # or "data" if _splits was deleted
IMG_SIZE      = 224
DROPOUT       = 0.3
WEIGHT_DECAY  = 1e-4
OUT_DIR       = "outputs"
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Load class count from the mapping we saved at training time.
with open(CLASSES_PATH) as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Rebuilding model for {num_classes} classes")


# ---------------------------------------------------------------------------
# Rebuild architecture identical to what train_mobilenetv3.py produces.
# No augmentation here — we don't need it for inference / export.
# The augmentation layers only activate during training anyway.
# ---------------------------------------------------------------------------
def build_inference_model(num_classes):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    backbone = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,                    # we'll load from checkpoint
        include_preprocessing=True,
        pooling="avg",
    )
    x = backbone(inputs, training=False)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(
        num_classes,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    return keras.Model(inputs, outputs, name="mobilenetv3small_lajolla")


model = build_inference_model(num_classes)

# ---------------------------------------------------------------------------
# Load weights only. `by_name=True` + `skip_mismatch=True` means the loader
# ignores the augmentation Sublayer entirely — backbone + head weights
# transfer cleanly because their layer names match.
# ---------------------------------------------------------------------------
print(f"Loading weights from {WEIGHTS_PATH}...")
model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
model.summary()


# ---------------------------------------------------------------------------
# Float32 export — sanity baseline.
# ---------------------------------------------------------------------------
print("\nExporting float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
fp32_path = os.path.join(OUT_DIR, "model_fp32.tflite")
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)
print(f"  {fp32_path} ({len(tflite_fp32)/1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# INT8 export — what we deploy.
# ---------------------------------------------------------------------------
def rep_dataset():
    src = pathlib.Path(SPLITS_ROOT)
    # Walk whatever structure exists under SPLITS_ROOT and pull images.
    # Works with _splits/train/<class>/*.jpg OR data/<class>/*.jpg.
    if not src.exists():
        raise FileNotFoundError(
            f"Expected calibration images under '{src}'. "
            "Set SPLITS_ROOT to either '_splits' (from training) or 'data'."
        )
    count = 0
    for img_path in src.rglob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png",
                                           ".bmp", ".webp"):
            continue
        img = tf.io.read_file(str(img_path))
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, 0)
        yield [img]
        count += 1
        if count >= 100:
            return
    if count == 0:
        raise RuntimeError(f"No images found under '{src}' for calibration")


print("\nExporting INT8 TFLite (this takes a minute)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8 = converter.convert()
int8_path = os.path.join(OUT_DIR, "model_int8.tflite")
with open(int8_path, "wb") as f:
    f.write(tflite_int8)
print(f"  {int8_path} ({len(tflite_int8)/1e6:.2f} MB)")

print("\nDone. Transfer these to the Uno Q:")
print(f"  {int8_path}")
print(f"  {CLASSES_PATH}")