"""
Headless live species identification with Test-Time Augmentation (TTA)
and temperature scaling, for any manifest-described TFLite model.

The classifier itself lives in species_identification/vision/tflite_classifier.py
and is shared with full_roboranger_run.py. It reads model_manifest.json next to
the .tflite to learn the model's input domain (raw 0..255 for MobileNetV3,
[-1, 1] for MobileNetV2, ImageNet mean/std for ONNX exports), its class order,
temperature and TTA views. There is no hard-coded preprocessing here any more:
the old copy fed raw pixels to every model, which is only correct for the
MobileNetV3 graphs.

Each loop tick grabs a frame, averages temperature-scaled softmax over the TTA
views (original, horizontal flip, 90% and 80% center crops), logs top-3 to a
CSV and optionally saves an annotated capture. Logs go to a per-model folder
so runs of different models are never mixed in one CSV.

Setup:
    pip install --break-system-packages ai-edge-litert numpy opencv-python

Files to transfer to the Uno Q:
    species_identification/vision/tflite_classifier.py
    <model dir>/model_*.tflite + model_manifest.json
    live_inference_mv3.py (this file)

Run:
    python3 live_inference_mv3.py                                    # outputs/deploy
    python3 live_inference_mv3.py --model-dir species_identification/outputs/mobilenetv3l
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "species_identification" / "vision"))
from tflite_classifier import TFLiteClassifierTTA  # noqa: E402


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = _ROOT / "species_identification" / "outputs" / "deploy"
LIVE_LOG_ROOT     = _ROOT / "species_identification" / "outputs" / "live"

CAMERA_INDEX    = 1            # laptop external webcam is 1, built-in / Uno Q is 0
TOP_K           = 5
CONF_THRESH     = 0.15
INFER_HZ        = 2.0          # TTA uses 4x inferences — keep rate modest
LOG_EVERY_S     = 5.0


# ---------------------------------------------------------------------------
def _annotate(frame, preds, tag="TTA"):
    h, w = frame.shape[:2]
    strip_h = 30 + 26 * len(preds)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame,
                time.strftime("%Y-%m-%d %H:%M:%S") + f"  [{tag}]",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)

    y = 50
    for name, p in preds:
        if p < CONF_THRESH:
            color = (140, 140, 140)
        elif p > 0.7:
            color = (0, 255, 0)
        else:
            color = (0, 200, 255)
        cv2.putText(frame, f"{name:<28s} {p*100:5.1f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 26
    return frame


# ---------------------------------------------------------------------------
def main(default_model_dir: Path = DEFAULT_MODEL_DIR) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=default_model_dir,
                    help="directory with model_*.tflite + "
                         "model_manifest.json")
    ap.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    ap.add_argument("--hz", type=float, default=INFER_HZ)
    ap.add_argument("--no-save-frames", action="store_true",
                    help="don't write an annotated capture every tick")
    args = ap.parse_args()

    clf = TFLiteClassifierTTA.from_dir(args.model_dir)
    print(f"[info] {clf.describe()}", flush=True)

    # Per-model log folder: comparing two models used to mean reading one
    # CSV that both scripts appended to.
    log_dir = LIVE_LOG_ROOT / args.model_dir.resolve().name
    capture_dir = log_dir / "captures_tta"
    log_path = log_dir / "predictions_tta.csv"
    os.makedirs(capture_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[error] Could not open /dev/video{args.camera_index}",
              file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[info] Camera opened; logging to {log_path}", flush=True)

    log_is_new = not log_path.exists()
    log_file = open(log_path, "a", newline="", buffering=1)
    writer = csv.writer(log_file)
    if log_is_new:
        writer.writerow(["timestamp_iso", "top1_label", "top1_conf",
                         "top2_label", "top2_conf", "top3_label", "top3_conf",
                         "saved_image"])

    running = {"ok": True}
    def stop(_sig, _frm):
        running["ok"] = False
        print("\n[info] Stopping...", flush=True)
    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / args.hz
    last_label = None
    last_log_t = 0.0
    n_infer = 0
    t_start = time.time()

    try:
        while running["ok"]:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                print("[warn] Frame grab failed", flush=True)
                time.sleep(0.1)
                continue

            preds = clf.predict(frame, top_k=TOP_K)
            n_infer += 1
            top_label, top_conf = preds[0]

            saved = ""
            if not args.no_save_frames:
                annotated = _annotate(frame.copy(), preds)
                ts_tag = time.strftime("%Y%m%d_%H%M%S") + f"_{n_infer:05d}"
                safe = top_label.replace(" ", "_").replace("/", "_")
                out_path = capture_dir / f"{ts_tag}_{safe}_{int(top_conf*100):02d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                saved = out_path.name

            now = time.time()
            label_changed = (top_label != last_label)
            heartbeat = (now - last_log_t) >= LOG_EVERY_S

            if label_changed or heartbeat:
                stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                line = f"[{stamp}] "
                for name, p in preds[:3]:
                    line += f"{name}:{p*100:4.1f}%  "
                if top_conf < CONF_THRESH:
                    line += "(uncertain)"
                print(line, flush=True)

                row = [stamp]
                for i in range(3):
                    row += [preds[i][0], f"{preds[i][1]:.4f}"]
                row.append(saved)
                writer.writerow(row)

                last_label = top_label
                last_log_t = now

            sleep_for = period - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        elapsed = time.time() - t_start
        print(f"[info] Ran {n_infer} TTA inferences in {elapsed:.1f}s "
              f"({n_infer/max(elapsed,1e-6):.2f} Hz, "
              f"{len(clf.tta_views) * n_infer / max(elapsed,1e-6):.1f} "
              f"total model runs/s)", flush=True)
        cap.release()
        log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
