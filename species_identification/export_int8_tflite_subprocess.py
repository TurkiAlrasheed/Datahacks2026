#!/usr/bin/env python3
"""ONNX → onnx2tf INT8 TFLite only (subprocess; keeps TensorFlow out of the PyTorch notebook).

PyTorch/ONNX models ONLY (species_identification_quantizationaware.ipynb, DINOv2).
Do NOT use this for the Keras MobileNet models — those go through
export_tflite.py. The input domains are different and incompatible:

    this script (PyTorch/ONNX)    ImageNet-normalized input: (x/255 - mean) / std,
                                  int8 input tensor
    Keras MobileNetV3             raw pixels 0..255 (rescale inside the graph),
                                  uint8 input tensor
    Keras MobileNetV2             x / 127.5 - 1, int8 input tensor

Requires calibration images as a single .npy file: float32, shape [N,H,W,3], values in [0,1]
(RGB, NHWC). onnx2tf applies --mean/--std to them for integer calibration, and the resulting
model expects ImageNet-normalized input at inference — the device must normalize the same way
(model_manifest.json input_range "imagenet").

Usage:
  python export_int8_tflite_subprocess.py <onnx_path> <output_dir> <calib_npy_path> [onnx_input_name]
      [--mean 0.485 0.456 0.406] [--std 0.229 0.224 0.225]
      [--class-names class_names.json] [--arch dinov2_vitb14]

Writes output_dir/model_int8.tflite (full integer quant variant) and, when --class-names is
given, output_dir/model_manifest.json for the on-device classifier. Intermediate float tflite
files from onnx2tf are removed after a successful run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def write_manifest(out_dir: Path, model_path: Path, *, arch: str,
                   img_size: int, class_names: list[str], onnx_path: Path,
                   n_calib: int) -> Path:
    """
    Same schema as vision/tflite_classifier.ModelManifest, written as plain
    JSON so this subprocess doesn't need OpenCV. temperature/id_threshold
    are placeholders until tests/eval_tflite.py --write.
    """
    manifest = {
        "model_file": model_path.name,
        "arch": arch,
        "input_range": "imagenet",
        "img_size": img_size,
        "input_dtype": "int8",
        "class_names": class_names,
        "negative_classes": [c for c in ("seashore",) if c in class_names],
        "temperature": 1.0,
        "id_threshold": 0.55,
        "tta_views": [[1.0, False], [1.0, True], [0.9, False], [0.8, False]],
        "source": {
            "onnx": str(onnx_path).replace("\\", "/"),
            "tflite_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "converter": "onnx2tf full_integer_quant",
            "calibration": f"{n_calib} images, ImageNet mean/std",
            "notes": "temperature/id_threshold are placeholders until "
                     "tests/eval_tflite.py --write",
        },
        "eval": {},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = out_dir / "model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("calib_npy_path", type=Path)
    ap.add_argument("onnx_input_name", nargs="?", default="input")
    ap.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN,
                    help="per-channel mean applied to [0,1] calibration data")
    ap.add_argument("--std", type=float, nargs=3, default=IMAGENET_STD,
                    help="per-channel std applied to [0,1] calibration data")
    ap.add_argument("--class-names", type=Path, default=None,
                    help="class_names.json; if given, also write "
                         "model_manifest.json")
    ap.add_argument("--arch", default="dinov2",
                    help="architecture label for the manifest")
    args = ap.parse_args()

    onnx_path = args.onnx_path.resolve()
    out_dir = args.output_dir.resolve()
    calib_npy = args.calib_npy_path.resolve()
    input_name = args.onnx_input_name

    if not onnx_path.is_file():
        sys.exit(f"ONNX not found: {onnx_path}")
    if not calib_npy.is_file():
        sys.exit(f"Calibration .npy not found: {calib_npy}")

    import numpy as np
    import onnx2tf

    calib = np.load(calib_npy)
    if calib.ndim != 4 or calib.shape[-1] != 3:
        sys.exit(
            f"Calibration array must be NHWC float32 with 3 channels; got shape {calib.shape}"
        )
    if calib.dtype != np.float32:
        calib = calib.astype(np.float32)
    if calib.min() < 0.0 or calib.max() > 1.0:
        sys.exit(
            f"Calibration data must be in [0,1] before normalization; got range "
            f"[{calib.min():.3f}, {calib.max():.3f}]. Passing already-normalized or "
            f"0..255 data here double-normalizes the int8 input ranges."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array(args.mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(args.std, dtype=np.float32).reshape(1, 1, 1, 3)

    # ViT/attention: unfold BatchMatMul so more ops can be coerced into int8 TFLite ops.
    # Set ONNX2TF_NO_BATCHMATMUL_UNFOLD=1 to disable if conversion fails or regresses accuracy.
    _unfold = os.environ.get("ONNX2TF_NO_BATCHMATMUL_UNFOLD", "").lower() not in (
        "1",
        "true",
        "yes",
    )

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        output_integer_quantized_tflite=True,
        non_verbose=True,
        enable_batchmatmul_unfold=_unfold,
        custom_input_op_name_np_data_path=[
            [input_name, str(calib_npy), mean, std],
        ],
    )

    stem = onnx_path.stem
    full_int = out_dir / f"{stem}_full_integer_quant.tflite"
    if not full_int.is_file():
        # Fallback: any *_full_integer_quant.tflite
        cands = list(out_dir.glob("*_full_integer_quant.tflite"))
        if len(cands) == 1:
            full_int = cands[0]
        else:
            sys.exit(
                f"Expected {stem}_full_integer_quant.tflite in {out_dir}; "
                f"onnx2tf may have failed or used a different name."
            )

    final_path = out_dir / "model_int8.tflite"
    shutil.copyfile(full_int, final_path)

    # Remove all other .tflite artifacts from this conversion.
    for f in out_dir.glob("*.tflite"):
        if f.resolve() != final_path.resolve():
            try:
                f.unlink()
            except OSError:
                pass

    print(f"Wrote {final_path}")

    if args.class_names is not None:
        if list(args.mean) != IMAGENET_MEAN or list(args.std) != IMAGENET_STD:
            print("warn: non-ImageNet --mean/--std can't be expressed as a "
                  "manifest input_range; model_manifest.json not written",
                  file=sys.stderr)
            return
        class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
        path = write_manifest(out_dir, final_path, arch=args.arch,
                              img_size=int(calib.shape[1]),
                              class_names=class_names, onnx_path=onnx_path,
                              n_calib=int(calib.shape[0]))
        print(f"Wrote {path}")
    else:
        print("note: no --class-names given, so no model_manifest.json; the "
              "on-device classifier refuses to run a model without one "
              "(input_range for this export is 'imagenet').")


if __name__ == "__main__":
    main()
