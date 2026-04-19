#!/usr/bin/env python3
"""ONNX → onnx2tf INT8 TFLite only (subprocess; keeps TensorFlow out of the PyTorch notebook).

Requires calibration images as a single .npy file: float32, shape [N,H,W,3], values in [0,1]
(RGB, NHWC — onnx2tf applies ImageNet mean/std for quantization).

Usage:
  python export_int8_tflite_subprocess.py <onnx_path> <output_dir> <calib_npy_path> [onnx_input_name]

Writes only output_dir/model_int8.tflite (full integer quant variant). Intermediate float
tflite files from onnx2tf are removed after a successful run.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python export_int8_tflite_subprocess.py "
            "<onnx_path> <output_dir> <calib_npy_path> [onnx_input_name]",
            file=sys.stderr,
        )
        sys.exit(2)

    onnx_path = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    calib_npy = Path(sys.argv[3]).resolve()
    input_name = sys.argv[4] if len(sys.argv) > 4 else "input"

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

    out_dir.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization for onnx2tf integer calibration (NHWC).
    mean = np.array([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
    std = np.array([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)

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


if __name__ == "__main__":
    main()
