"""
Headless live species identification on the Arduino Uno Q.

No display window — prints top-5 predictions to the terminal and appends
each inference to a CSV log. Designed for SSH / field deployment.

Setup:
    pip install --break-system-packages ai-edge-litert numpy opencv-python

Run:
    python3 infer_headless.py

Stop with Ctrl+C. Predictions land in predictions.csv.
"""

import csv
import json
import os
import signal
import sys
import time
import numpy as np
import cv2

# Try import paths in order of preference.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH    = "model_int8.tflite"
CLASSES_PATH  = "class_names.json"
LOG_PATH      = "predictions.csv"
CAPTURE_DIR   = "captures"

CAMERA_INDEX  = 0
IMG_SIZE      = 224
TOP_K         = 5
CONF_THRESH   = 0.15           # below this, prediction is labeled "uncertain"

# How often to run inference (Hz). Keep modest — there's no display,
# so no point burning CPU inferring 30 times a second.
INFER_HZ      = 2.0
# Only log / print when the top-1 label changes OR every Nth second —
# prevents filling the terminal with duplicates.
LOG_EVERY_S   = 5.0
# Save a JPG whenever top-1 confidence exceeds this and the label changes.
SAVE_CONF     = 0.60
# Verification mode: save EVERY frame with prediction annotated onto the
# image. Useful for debugging — set to False for production field use so
# you don't fill the disk.
SAVE_ALL_FRAMES = True


# ---------------------------------------------------------------------------
# Classifier — identical logic to the windowed version, minus the overlay.
# ---------------------------------------------------------------------------
def _build_interpreter(model_path):
    """Construct a TFLite interpreter with XNNPACK disabled.

    XNNPACK fails to prepare some ops in INT8 MobileNetV3 on
    ai-edge-litert on ARM64. We try several disable mechanisms in order
    of reliability; any one working is enough.
    """
    # 1) Env var — read by the underlying native TFLite runtime at
    #    interpreter construction. Works for tflite-runtime and older
    #    tensorflow builds; unclear for ai-edge-litert but harmless to set.
    os.environ["TFLITE_DISABLE_XNNPACK"] = "1"

    # 2) Explicit BUILTIN_REF op resolver — forces reference kernels,
    #    bypassing the XNNPACK delegate entirely. Available on tf.lite
    #    and some ai-edge-litert builds.
    try:
        from ai_edge_litert.interpreter import OpResolverType  # noqa
        return Interpreter(
            model_path=model_path,
            num_threads=4,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
    except (ImportError, AttributeError, TypeError):
        pass

    # 3) Plain construction. If XNNPACK still tries to engage, the env
    #    var above should suppress it. If not, the user sees the same
    #    error and we debug further.
    return Interpreter(model_path=model_path, num_threads=4)


class TFLiteClassifier:
    def __init__(self, model_path, class_names):
        self.interpreter = _build_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.in_det  = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        self.class_names = class_names

        self.in_scale,  self.in_zp  = self.in_det["quantization"]
        self.out_scale, self.out_zp = self.out_det["quantization"]
        self.is_qin  = self.in_det["dtype"]  == np.uint8
        self.is_qout = self.out_det["dtype"] == np.uint8

    def preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32)
        if self.is_qin:
            x = x / self.in_scale + self.in_zp
            x = np.clip(x, 0, 255).astype(np.uint8)
        return np.expand_dims(x, 0)

    def predict(self, bgr):
        x = self.preprocess(bgr)
        self.interpreter.set_tensor(self.in_det["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_det["index"])[0]
        if self.is_qout:
            out = (out.astype(np.float32) - self.out_zp) * self.out_scale
        probs = _softmax(out)
        top_idx = np.argsort(probs)[::-1][:TOP_K]
        return [(self.class_names[i], float(probs[i])) for i in top_idx]


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _annotate(frame, preds):
    """Draw top-5 predictions onto the frame for offline review."""
    h, w = frame.shape[:2]
    # Dark strip at top for readability, same trick as the windowed version.
    strip_h = 30 + 26 * len(preds)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame,
                time.strftime("%Y-%m-%d %H:%M:%S"),
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
# Main loop
# ---------------------------------------------------------------------------
def main():
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    clf = TFLiteClassifier(MODEL_PATH, class_names)
    print(f"[info] Loaded model with {len(class_names)} classes", flush=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[error] Could not open camera /dev/video{CAMERA_INDEX}",
              file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[info] Camera opened", flush=True)

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    # Open log file in append mode so restarts don't nuke history.
    log_is_new = not os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="", buffering=1)   # line-buffered
    writer = csv.writer(log_file)
    if log_is_new:
        writer.writerow(["timestamp_iso", "top1_label", "top1_conf",
                         "top2_label", "top2_conf", "top3_label", "top3_conf",
                         "saved_image"])

    # Graceful Ctrl+C
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
                print("[warn] Frame grab failed, retrying", flush=True)
                time.sleep(0.1)
                continue

            preds = clf.predict(frame)
            n_infer += 1
            top_label, top_conf = preds[0]

            # Verification mode: annotate every frame and save it. The
            # annotated image shows exactly what the model saw and what it
            # decided, so you can flip through them later to verify.
            if SAVE_ALL_FRAMES:
                annotated = _annotate(frame.copy(), preds)
                ts_tag = time.strftime("%Y%m%d_%H%M%S") + f"_{n_infer:05d}"
                safe = top_label.replace(" ", "_").replace("/", "_")
                out_path = os.path.join(CAPTURE_DIR,
                                        f"{ts_tag}_{safe}_{int(top_conf*100):02d}.jpg")
                cv2.imwrite(out_path, annotated)

            now = time.time()
            label_changed = (top_label != last_label)
            heartbeat = (now - last_log_t) >= LOG_EVERY_S

            if label_changed or heartbeat:
                saved = ""
                if top_conf >= SAVE_CONF and label_changed and not SAVE_ALL_FRAMES:
                    ts_tag = time.strftime("%Y%m%d_%H%M%S")
                    safe = top_label.replace(" ", "_").replace("/", "_")
                    saved = os.path.join(CAPTURE_DIR,
                                         f"{ts_tag}_{safe}.jpg")
                    cv2.imwrite(saved, frame)

                stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                # Pretty console print
                line = f"[{stamp}] "
                for name, p in preds[:3]:
                    line += f"{name}:{p*100:4.1f}%  "
                if top_conf < CONF_THRESH:
                    line += "(uncertain)"
                if saved:
                    line += f"  -> {saved}"
                print(line, flush=True)

                # CSV row (top 3 + saved path)
                row = [stamp]
                for i in range(3):
                    row += [preds[i][0], f"{preds[i][1]:.4f}"]
                row.append(saved)
                writer.writerow(row)

                last_label = top_label
                last_log_t = now

            # Pace the loop to INFER_HZ
            sleep_for = period - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        elapsed = time.time() - t_start
        print(f"[info] Ran {n_infer} inferences in {elapsed:.1f}s "
              f"({n_infer/max(elapsed,1e-6):.2f} Hz)", flush=True)
        cap.release()
        log_file.close()


if __name__ == "__main__":
    main()