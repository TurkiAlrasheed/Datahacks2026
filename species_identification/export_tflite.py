"""
Export a trained Keras classifier to TFLite + model_manifest.json.

This is the export path for the Keras MobileNet models (mobile_net_v3l.py,
mobile_net_v3s.py, mobile_net_v2.py). It is NOT the ONNX path —
export_int8_tflite_subprocess.py is for the PyTorch/DINOv2 notebooks and uses
ImageNet mean/std, a different input domain.

What it does:
  1. keras.models.load_model(--keras) and rebuild a clean inference graph
     that skips the training-only `augment` block.
  2. Convert with --quant:
       int8     full-integer PTQ, representative set drawn from the TRAIN and
                VAL splits only (never test)
       dynamic  int8 weights / float activations — use this for MobileNetV3:
                full-int8 PTQ collapses its accuracy (see tests/eval_tflite.py)
       float16, float32
  3. Write model_<quant>.tflite + model_manifest.json declaring the input
     domain (input_range), I/O dtype, image size, and class order, then load
     the result in LiteRT and check it against the manifest.

Temperature and id_threshold are left at placeholders: they must be fit on
the exported model's TTA-averaged outputs, which is what
tests/eval_tflite.py --write does. Always compare against the float model
(--keras) before shipping a new export.

Usage (from the repo root):
    python species_identification/export_tflite.py \\
        --keras species_identification/outputs/mobilenetv3l/best.keras \\
        --arch mobilenetv3l --quant dynamic \\
        --out species_identification/outputs/mobilenetv3l_dynamic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "vision"))
from tflite_classifier import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelManifest,
    TFLiteClassifierTTA,
)

# Input domain each architecture was trained with. MobileNetV3 in Keras has
# include_preprocessing=True (rescale inside the graph, feed raw pixels);
# mobile_net_v2.py scales to [-1, 1] in the tf.data pipeline instead.
ARCH_INPUT_RANGE = {
    "mobilenetv3l": "raw_0_255",
    "mobilenetv3s": "raw_0_255",
    "mobilenetv2": "minus1_1",
    "mobilenetv2_qat": "minus1_1",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inference_model(model):
    """
    Rebuild `model` without its training-only augmentation block. The
    MobileNet trainers build a linear chain: Input -> augment -> backbone ->
    dropout -> dense. Augmentation layers are identity at inference, but
    dropping them keeps random ops out of the converted graph.
    """
    import keras

    names = [layer.name for layer in model.layers]
    if "augment" not in names:
        return model
    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer) or layer.name == "augment":
            continue
        x = layer(x, training=False)
    return keras.Model(inputs, x, name=f"{model.name}_inference")


def calibration_paths(splits: Path, class_names: list[str], n: int,
                      seed: int) -> list[Path]:
    """Round-robin across classes over train+val so every class is covered."""
    rng = random.Random(seed)
    per_class = []
    for cname in class_names:
        files = []
        for split in ("train", "val"):
            d = splits / split / cname
            if d.is_dir():
                files += sorted(p for p in d.iterdir()
                                if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        per_class.append(files)
    if not any(per_class):
        raise FileNotFoundError(f"no train/val images under {splits}")
    out, i = [], 0
    while len(out) < n and any(per_class):
        bucket = per_class[i % len(per_class)]
        if bucket:
            out.append(bucket.pop())
        i += 1
    return out


def representative_dataset(paths: list[Path], img_size: int,
                           input_range: str):
    import tensorflow as tf

    def gen():
        for p in paths:
            img = tf.io.decode_image(tf.io.read_file(str(p)), channels=3,
                                     expand_animations=False)
            # Same resize as keras.utils.image_dataset_from_directory.
            img = tf.image.resize(img, (img_size, img_size))
            x = tf.cast(img, tf.float32).numpy()
            if input_range == "minus1_1":
                x = x / 127.5 - 1.0
            elif input_range == "imagenet":
                x = (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            yield [x[None].astype(np.float32)]
    return gen


QUANT_MODES = ("int8", "dynamic", "float16", "float32")


def convert(model, rep_gen, io_dtype: str, quant: str = "int8") -> bytes:
    """
    int8     full-integer PTQ (int8 activations, integer I/O). Smallest and
             fastest, but post-training quantization of MobileNetV3's
             hard-swish/SE activations can destroy accuracy — measure with
             tests/eval_tflite.py --keras before shipping.
    dynamic  int8 weights, float activations, float32 I/O. ~4x smaller than
             float32 with near-float accuracy; no calibration data needed.
    float16  float16 weights, float32 I/O.
    float32  no quantization.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant == "int8":
        tf_dtype = {"uint8": tf.uint8, "int8": tf.int8}[io_dtype]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf_dtype
        converter.inference_output_type = tf_dtype
    elif quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant != "float32":
        raise ValueError(f"unknown quant mode {quant!r}")
    return converter.convert()


def export_model(model, *, arch: str, class_names: list[str], splits: Path,
                 out_dir: Path, keras_path: Path | None = None,
                 temperature: float | None = None, n_calib: int = 300,
                 io_dtype: str | None = None, seed: int = 42,
                 quant: str = "int8") -> Path:
    """
    Convert an in-memory Keras model to model_<quant>.tflite + manifest in
    out_dir. Shared by the CLI below and the training scripts' export step.
    Returns the manifest path.
    """
    input_range = ARCH_INPUT_RANGE[arch]
    if quant == "int8":
        io_dtype = io_dtype or ("uint8" if input_range == "raw_0_255"
                                else "int8")
    else:
        io_dtype = "float32"
    model = inference_model(model)
    img_size = int(model.input_shape[1])
    n_out = int(model.output_shape[-1])
    if n_out != len(class_names):
        raise ValueError(f"model outputs {n_out} classes but class_names "
                         f"lists {len(class_names)}")

    rep_gen, calibration = None, "none (no activation quantization)"
    if quant == "int8":
        paths = calibration_paths(Path(splits), class_names, n_calib, seed)
        print(f"calibrating on {len(paths)} train/val images "
              f"({input_range}, {img_size}x{img_size}) ...")
        rep_gen = representative_dataset(paths, img_size, input_range)
        calibration = f"{len(paths)} images from train+val, seed {seed}"
    tflite = convert(model, rep_gen, io_dtype, quant)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{quant}.tflite"
    model_path.write_bytes(tflite)

    source = {
        "tflite_sha256": _sha256(model_path),
        "converter": f"tf.lite.TFLiteConverter.from_keras_model ({quant})",
        "calibration": calibration,
        "notes": "id_threshold is a placeholder and temperature was not fit "
                 "on this exported model; run tests/eval_tflite.py --write",
    }
    if keras_path is not None and Path(keras_path).is_file():
        keras_path = Path(keras_path)
        source.update({
            "keras": str(keras_path).replace("\\", "/"),
            "keras_sha256": _sha256(keras_path),
            "keras_mtime": datetime.fromtimestamp(
                keras_path.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds"),
        })
    if temperature is not None:
        source["float_temperature"] = (
            f"{temperature} (fit on float single-view val logits during "
            f"training; used as the starting temperature)")

    manifest = ModelManifest(
        model_file=model_path.name,
        arch=arch,
        input_range=input_range,
        img_size=img_size,
        input_dtype=io_dtype,
        class_names=list(class_names),
        temperature=float(temperature) if temperature else 1.0,
        source=source,
    )
    manifest_path = manifest.save(out_dir)

    # Round-trip: the runtime classifier must accept what we just wrote.
    clf = TFLiteClassifierTTA(ModelManifest.load(manifest_path),
                              reference_kernels=False)
    clf.predict(np.full((480, 640, 3), 128, dtype=np.uint8))
    print(f"wrote {model_path} ({len(tflite) / 1e6:.2f} MB)")
    print(f"wrote {manifest_path}")
    print(f"  {clf.describe()}")
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keras", type=Path, required=True,
                    help="trained .keras checkpoint")
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_INPUT_RANGE))
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for model_<quant>.tflite + "
                         "manifest")
    ap.add_argument("--class-names", type=Path, default=None,
                    help="class_names.json written by the training run "
                         "(default: next to --keras)")
    ap.add_argument("--splits", type=Path,
                    default=Path("species_identification/cnn/_splits"),
                    help="split root with train/ and val/ class folders")
    ap.add_argument("--n-calib", type=int, default=300,
                    help="representative images for INT8 calibration")
    ap.add_argument("--quant", choices=QUANT_MODES, default="int8",
                    help="quantization mode (see convert())")
    ap.add_argument("--io-dtype", choices=("uint8", "int8"), default=None,
                    help="int8 mode only; default: uint8 for raw_0_255 "
                         "models, int8 otherwise")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import keras

    class_names_path = args.class_names or args.keras.with_name(
        "class_names.json")
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

    print(f"loading {args.keras} ...")
    model = keras.models.load_model(args.keras, compile=False)
    try:
        export_model(model, arch=args.arch, class_names=class_names,
                     splits=args.splits, out_dir=args.out,
                     keras_path=args.keras, n_calib=args.n_calib,
                     io_dtype=args.io_dtype, seed=args.seed,
                     quant=args.quant)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
