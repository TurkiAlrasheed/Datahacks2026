"""
List camera devices visible to OpenCV. Run this once to find the Brio's
index, then pass it via --camera-index to voice_loop.py.

On Windows, OpenCV's default backend is MSMF, which can be flaky and slow
to enumerate. DirectShow (CAP_DSHOW) is usually more reliable for picking
specific USB cameras. This script tries both and saves a thumbnail from
each working index so you can eyeball which is which.

Usage:
    python list_cameras.py
    # then open cam_*.jpg in the current dir
"""
import cv2
import time

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),    # DirectShow — usually best for USB cams
    ("MSMF",  cv2.CAP_MSMF),     # Media Foundation — OpenCV default on Windows
    ("ANY",   cv2.CAP_ANY),      # let OpenCV decide
]

for backend_name, backend in BACKENDS:
    print(f"\n--- backend: {backend_name} ---")
    for idx in range(5):  # check indices 0..4
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Set 1080p just so the Brio doesn't get stuck at 640x480.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # Flush a couple of frames before grabbing.
        for _ in range(3):
            cap.read()
            time.sleep(0.05)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            out = f"cam_{backend_name}_{idx}.jpg"
            cv2.imwrite(out, frame)
            print(f"  index {idx}: OK  {w}x{h}  -> {out}")
        else:
            print(f"  index {idx}: opened but no frame")
        cap.release()