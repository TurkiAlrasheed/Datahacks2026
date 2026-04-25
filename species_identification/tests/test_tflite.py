"""
Quick sanity test for the exported TFLite model.

Run on your laptop:
    python test_tflite.py [path/to/test/images/]

With no argument, uses _splits/test/ from the training run.
Prints top-1/top-5 accuracy and per-class confusion summary.
"""

import json
import pathlib
import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict

MODEL_PATH   = "../outputs/model_int8.tflite"
CLASSES_PATH = "../outputs/class_names.json"
IMG_SIZE     = 224


def load_image(path):
    img = tf.io.read_file(str(path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32).numpy()   # [0, 255], matches training


def main():
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    test_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                             else "_splits/test")
    if not test_root.exists():
        print(f"ERROR: {test_root} not found. Pass a path as an argument.")
        sys.exit(1)

    # Load model
    interp = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp   = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    # Walk test set
    totals = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})
    all_n = 0; all_top1 = 0; all_top5 = 0

    for class_dir in sorted(test_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in cls_to_idx:
            continue
        true_idx = cls_to_idx[class_dir.name]

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png",
                                               ".bmp", ".webp"):
                continue

            x = load_image(img_path)
            x = x / in_scale + in_zp
            x = np.clip(x, 0, 255).astype(np.uint8)[None]

            interp.set_tensor(in_det["index"], x)
            interp.invoke()
            out = interp.get_tensor(out_det["index"])[0]
            out = (out.astype(np.float32) - out_zp) * out_scale
            top5 = np.argsort(out)[::-1][:5]

            totals[class_dir.name]["n"] += 1
            all_n += 1
            if top5[0] == true_idx:
                totals[class_dir.name]["top1"] += 1
                all_top1 += 1
            if true_idx in top5:
                totals[class_dir.name]["top5"] += 1
                all_top5 += 1

    print(f"\nOverall: top1 = {all_top1}/{all_n} = {all_top1/all_n*100:5.1f}%"
          f"  |  top5 = {all_top5}/{all_n} = {all_top5/all_n*100:5.1f}%\n")

    # Per-class, worst first — tells you which species the model is blind to.
    print(f"{'Class':<40s} {'N':>3s} {'Top1':>6s} {'Top5':>6s}")
    print("-" * 58)
    by_acc = sorted(totals.items(),
                    key=lambda kv: kv[1]["top1"] / max(kv[1]["n"], 1))
    for name, t in by_acc:
        if t["n"] == 0:
            continue
        print(f"{name:<40s} {t['n']:>3d} "
              f"{t['top1']/t['n']*100:>5.1f}% {t['top5']/t['n']*100:>5.1f}%")


if __name__ == "__main__":
    main()