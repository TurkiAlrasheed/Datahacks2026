"""
Headless live species identification with Test-Time Augmentation (TTA)
and temperature scaling.

Same as live_inference.py but also:
    - Runs inference on multiple augmented views of each frame and
      averages the probabilities (TTA).
    - Applies temperature scaling: divides logits by a learned scalar T
      before softmax, correcting for under/over-confidence. T is saved by
      the training script at outputs/temperature.json. If the file isn't
      present, T=1.0 is used (no change).

Views used: original, horizontal flip, center crop (90%), center crop (80%).
Roughly 4x the inference cost per frame. Still well above real-time on the
Uno Q with MobileNetV3-Small or -Large.

Setup:
    pip install --break-system-packages ai-edge-litert numpy opencv-python

Files to transfer to the Uno Q:
    model_int8.tflite
    class_names.json
    temperature.json      (optional — falls back to T=1 if missing)
    live_inference_tta_calibrated.py

Run:
    python3 live_inference_tta_calibrated.py
"""

import csv
import json
import os
import signal
import sys
import time
import numpy as np
import cv2

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter


# ---------------------------------------------------------------------------
# Config — same defaults as live_inference.py
# ---------------------------------------------------------------------------
MODEL_PATH      = "model_int8.tflite"
CLASSES_PATH    = "class_names.json"
TEMP_PATH       = "temperature.json"   # optional — T=1.0 if missing
LOG_PATH        = "predictions_tta.csv"
CAPTURE_DIR     = "captures_tta"

CAMERA_INDEX    = 0
IMG_SIZE        = 224
TOP_K           = 5
CONF_THRESH     = 0.15
INFER_HZ        = 2.0          # TTA uses 4x inferences — keep rate modest
LOG_EVERY_S     = 5.0
SAVE_CONF       = 0.60
SAVE_ALL_FRAMES = True

# TTA views. Each entry is a crop fraction (1.0 = no crop) and whether
# to horizontally flip. Keep this list short — each entry costs one
# full forward pass.
TTA_VIEWS = [
    (1.00, False),   # original
    (1.00, True),    # horizontal flip
    (0.90, False),   # 90% center crop
    (0.80, False),   # 80% center crop
]


# ---------------------------------------------------------------------------
def _build_interpreter(model_path):
    os.environ["TFLITE_DISABLE_XNNPACK"] = "1"
    try:
        from ai_edge_litert.interpreter import OpResolverType
        return Interpreter(
            model_path=model_path,
            num_threads=4,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
    except (ImportError, AttributeError, TypeError):
        pass
    return Interpreter(model_path=model_path, num_threads=4)


def _center_crop(img, frac):
    """Crop the central fraction of the image (H, W, C)."""
    if frac >= 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = int(h * frac), int(w * frac)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


class TFLiteClassifierTTA:
    def __init__(self, model_path, class_names, temperature=1.0):
        self.interpreter = _build_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.in_det  = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        self.class_names = class_names
        self.temperature = max(float(temperature), 1e-3)   # avoid div-by-zero

        self.in_scale,  self.in_zp  = self.in_det["quantization"]
        self.out_scale, self.out_zp = self.out_det["quantization"]
        self.is_qin  = self.in_det["dtype"]  == np.uint8
        self.is_qout = self.out_det["dtype"] == np.uint8

    def _preprocess_view(self, rgb_img, crop_frac, flip):
        """Build one TTA view ready for the interpreter."""
        img = _center_crop(rgb_img, crop_frac)
        if flip:
            img = img[:, ::-1, :]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32)
        if self.is_qin:
            x = x / self.in_scale + self.in_zp
            x = np.clip(x, 0, 255).astype(np.uint8)
        return np.expand_dims(x, 0)

    def _infer_once(self, x):
        """Returns softmax probabilities with temperature applied."""
        self.interpreter.set_tensor(self.in_det["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_det["index"])[0]
        if self.is_qout:
            # Dequantize to float logits, then apply temperature scaling
            # before softmax. T<1 sharpens, T>1 softens.
            logits = (out.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            logits = out.astype(np.float32)
        return _softmax(logits / self.temperature)

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Average temperature-scaled softmax probs across all TTA views.
        prob_sum = None
        for crop_frac, flip in TTA_VIEWS:
            x = self._preprocess_view(rgb, crop_frac, flip)
            probs = self._infer_once(x)
            prob_sum = probs if prob_sum is None else prob_sum + probs
        probs = prob_sum / len(TTA_VIEWS)
        top_idx = np.argsort(probs)[::-1][:TOP_K]
        return [(self.class_names[i], float(probs[i])) for i in top_idx]


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _annotate(frame, preds):
    h, w = frame.shape[:2]
    strip_h = 30 + 26 * len(preds)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame,
                time.strftime("%Y-%m-%d %H:%M:%S") + "  [TTA]",
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
def main():
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    # Load temperature scaling factor if available. Defaults to 1.0 (no-op).
    temperature = 1.0
    if os.path.exists(TEMP_PATH):
        try:
            with open(TEMP_PATH) as f:
                temperature = float(json.load(f).get("temperature", 1.0))
            print(f"[info] Loaded temperature T={temperature:.3f}", flush=True)
        except Exception as e:
            print(f"[warn] Could not read {TEMP_PATH} ({e}), using T=1.0",
                  flush=True)
    else:
        print(f"[info] No {TEMP_PATH} found, using T=1.0 (uncalibrated)",
              flush=True)

    clf = TFLiteClassifierTTA(MODEL_PATH, class_names, temperature=temperature)
    print(f"[info] Loaded model with {len(class_names)} classes, "
          f"{len(TTA_VIEWS)} TTA views", flush=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[error] Could not open /dev/video{CAMERA_INDEX}",
              file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[info] Camera opened", flush=True)

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    log_is_new = not os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="", buffering=1)
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

    period = 1.0 / INFER_HZ
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

            preds = clf.predict(frame)
            n_infer += 1
            top_label, top_conf = preds[0]

            if SAVE_ALL_FRAMES:
                annotated = _annotate(frame.copy(), preds)
                ts_tag = time.strftime("%Y%m%d_%H%M%S") + f"_{n_infer:05d}"
                safe = top_label.replace(" ", "_").replace("/", "_")
                out_path = os.path.join(
                    CAPTURE_DIR,
                    f"{ts_tag}_{safe}_{int(top_conf*100):02d}.jpg")
                cv2.imwrite(out_path, annotated)

            now = time.time()
            label_changed = (top_label != last_label)
            heartbeat = (now - last_log_t) >= LOG_EVERY_S

            if label_changed or heartbeat:
                saved = ""
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
              f"{len(TTA_VIEWS) * n_infer / max(elapsed,1e-6):.1f} "
              f"total model runs/s)", flush=True)
        cap.release()
        log_file.close()


if __name__ == "__main__":
    main()