#!/usr/bin/env python3
"""Run ONNX → onnx2tf → TFLite in an isolated process (not inside a PyTorch Jupyter kernel).

Usage:
  python export_tflite_subprocess.py <onnx_path> <output_dir> [calib_npz|none]

onnx2tf's default flatbuffer_direct mode often writes model_float32/16.tflite without saved_model.pb.
This script only uses TFLiteConverter.from_saved_model when that file exists.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python export_tflite_subprocess.py <onnx_path> <output_dir> [calib_npz|none]",
            file=sys.stderr,
        )
        sys.exit(2)

    onnx_path = Path(sys.argv[1]).resolve()
    saved_model_dir = Path(sys.argv[2]).resolve()
    calib = sys.argv[3] if len(sys.argv) > 3 else "none"

    import numpy as np
    import onnx2tf
    import tensorflow as tf

    if not onnx_path.is_file():
        sys.exit(f"ONNX not found: {onnx_path}")

    saved_model_dir.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
    )

    def find_saved_model_dir(root: Path) -> Path | None:
        if (root / "saved_model.pb").is_file() or (root / "saved_model.pbtxt").is_file():
            return root
        for pb in root.rglob("saved_model.pb"):
            return pb.parent
        return None

    sm = find_saved_model_dir(saved_model_dir)

    if sm is not None:
        print(f"Found SavedModel at: {sm}")
        conv32 = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        p32 = saved_model_dir / "model_float32.tflite"
        p32.write_bytes(conv32.convert())
        print(f"Wrote {p32}")

        conv16 = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        conv16.optimizations = [tf.lite.Optimize.DEFAULT]
        conv16.target_spec.supported_types = [tf.float16]
        p16 = saved_model_dir / "model_float16.tflite"
        p16.write_bytes(conv16.convert())
        print(f"Wrote {p16}")
    else:
        f32 = list(saved_model_dir.glob("model_float32.tflite")) + list(
            saved_model_dir.glob("*_float32.tflite")
        )
        f16 = list(saved_model_dir.glob("model_float16.tflite")) + list(
            saved_model_dir.glob("*_float16.tflite")
        )
        if not f32:
            print(
                "ERROR: onnx2tf did not write saved_model.pb or a float32 .tflite.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "onnx2tf used flatbuffer_direct (no saved_model.pb). "
            "Using TFLite file(s) onnx2tf already wrote:"
        )
        print(" ", f32[0])
        if f16:
            print(" ", f16[0])

    if sm is not None and calib != "none" and Path(calib).exists():
        data = np.load(calib)
        keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))

        def rep():
            for k in keys:
                yield [data[k]]

        try:
            conv_int = tf.lite.TFLiteConverter.from_saved_model(str(sm))
            conv_int.optimizations = [tf.lite.Optimize.DEFAULT]
            conv_int.representative_dataset = rep
            conv_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv_int.inference_input_type = tf.uint8
            conv_int.inference_output_type = tf.uint8
            ip = saved_model_dir / "model_int8.tflite"
            ip.write_bytes(conv_int.convert())
            print(f"Wrote {ip}")
        except Exception as ex:
            print("Skipping full INT8 TFLite (optional):", ex)
    elif calib != "none":
        print(
            "Skipping INT8 from representative dataset: no SavedModel (flatbuffer_direct only)."
        )


if __name__ == "__main__":
    main()
