#!/usr/bin/env python3
"""
Benchmark the vision classifier on the device it will run on (the UNO Q).

Answers the one question eval_tflite.py can't: how long does an
identification take on this hardware, and which model/kernel/thread
combination should ship? Needs no dataset — timing a 320x320 forward pass
doesn't depend on image content, so it works on a bare device with just the
model directory, numpy and OpenCV.

For each (model dir, kernels, threads) it reports:
    load        interpreter construction + allocate_tensors
    preprocess  per TTA view: crop, resize, normalize, quantize
    invoke      one forward pass (what 4 TTA views each cost)
    identify    the full startup path: 4 views + temperature + policy
                — this is the number the demo feels
    rss         process memory after the run (needs psutil)

Each configuration runs in a fresh subprocess, because LiteRT's
TFLITE_DISABLE_XNNPACK is process-wide and read at interpreter construction:
running reference and default kernels in one process would mis-measure the
second one.

With --images it also prints each model's top-3 per image, so you can diff
device predictions against the laptop (run the same command in both places)
and confirm a faster kernel didn't change the answers.

Usage on the device (from the repo root):
    python3 species_identification/tests/bench_device.py
    python3 species_identification/tests/bench_device.py \\
        --model-dir species_identification/outputs/deploy \\
        --model-dir species_identification/outputs/mobilenetv2_qat \\
        --threads 1,2,4 --kernels ref,default --runs 10 \\
        --images species_identification/tests/bench_images \\
        --json bench_unoq.json

    # include the real camera path (open, flush, grab, classify):
    python3 species_identification/tests/bench_device.py --camera-index 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "vision"))
from tflite_classifier import (  # noqa: E402
    ModelContractError,
    ModelManifest,
    TFLiteClassifierTTA,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_DIRS = [
    _HERE.parent / "outputs" / "deploy",
    _HERE.parent / "outputs" / "mobilenetv2_qat",
]


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1e6


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "mean_ms": round(statistics.fmean(samples), 1),
        "p50_ms": round(statistics.median(samples), 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
    }


def _bench_one(conf: dict, out: "mp.Queue") -> None:
    """Run one configuration in this (fresh) process and queue the result."""
    import cv2

    result = {**conf}
    try:
        manifest = ModelManifest.load(conf["model_dir"])
        t0 = time.perf_counter()
        clf = TFLiteClassifierTTA(
            manifest, num_threads=conf["threads"],
            reference_kernels=conf["kernels"] == "ref")
        result["load_s"] = round(time.perf_counter() - t0, 2)
        result["describe"] = clf.describe()
        result["model_file"] = manifest.model_path.name
        result["model_mb"] = round(manifest.model_path.stat().st_size / 1e6, 2)
        result["views"] = len(clf.tta_views)

        # Time a CAMERA-SIZED frame: preprocessing cost scales with the
        # source resolution, so timing a multi-megapixel JPEG would report a
        # cost the device never pays at runtime (the camera delivers
        # 640x480). --timing-image overrides this deliberately.
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result["timing_frame"] = "synthetic 640x480"
        if conf.get("timing_image"):
            img = cv2.imread(conf["timing_image"])
            if img is None:
                raise RuntimeError(f"cannot read {conf['timing_image']}")
            frame = img
            result["timing_frame"] = (f"{Path(conf['timing_image']).name} "
                                      f"{frame.shape[1]}x{frame.shape[0]}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        clf.predict(frame)  # warm up: first invoke pays one-off setup

        # Per view, because they are not equally expensive: the flipped and
        # cropped views hand cv2.resize a strided array.
        per_view, inv = [], []
        for crop, flip in clf.tta_views:
            pre_v = []
            for _ in range(conf["runs"]):
                t = time.perf_counter()
                x = clf.preprocess_view(rgb, crop, flip)
                pre_v.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter()
                clf.logits(x)
                inv.append((time.perf_counter() - t) * 1000)
            per_view.append({"crop": crop, "flip": flip,
                             "p50_ms": round(statistics.median(pre_v), 1)})
        result["preprocess_per_view"] = per_view
        result["preprocess_total_ms"] = round(
            sum(v["p50_ms"] for v in per_view), 1)
        result["invoke"] = _stats(inv)
        result["invoke_total_ms"] = round(
            result["invoke"]["p50_ms"] * len(clf.tta_views), 1)

        ident = []
        for _ in range(max(3, conf["runs"] // 3)):
            t = time.perf_counter()
            clf.identify(frame)
            ident.append((time.perf_counter() - t) * 1000)
        result["identify"] = _stats(ident)

        if conf.get("images"):
            preds = {}
            for p in conf["images"]:
                img = cv2.imread(p)
                if img is None:
                    continue
                ident_p = clf.identify(img)
                preds[Path(p).name] = {
                    "accepted": ident_p.accepted,
                    "reason": ident_p.reason,
                    "top3": [[n, round(float(c), 4)]
                             for n, c in ident_p.top[:3]],
                }
            result["predictions"] = preds

        if conf.get("camera_index") is not None:
            result["camera"] = _bench_camera(clf, conf["camera_index"])

        result["rss_mb"] = _rss_mb()
    except ModelContractError as e:
        result["error"] = f"ModelContractError: {e}"
    except Exception as e:  # keep one bad config from killing the sweep
        result["error"] = f"{type(e).__name__}: {e}"
    out.put(result)


def _bench_camera(clf: TFLiteClassifierTTA, camera_index: int) -> dict:
    """Time the real startup path: open, flush stale frames, grab, classify."""
    import cv2

    timings: dict = {}
    t = time.perf_counter()
    cap = (cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
           if sys.platform == "win32" else cv2.VideoCapture(camera_index))
    timings["open_ms"] = round((time.perf_counter() - t) * 1000, 1)
    if not cap.isOpened():
        cap.release()
        return {"error": f"could not open camera {camera_index}"}
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t = time.perf_counter()
        for _ in range(5):  # CAMERA_WARMUP_FRAMES in full_roboranger_run.py
            cap.read()
        timings["flush5_ms"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        ok, frame = cap.read()
        timings["grab_ms"] = round((time.perf_counter() - t) * 1000, 1)
        if not ok or frame is None:
            timings["error"] = "frame grab failed"
            return timings
        t = time.perf_counter()
        ident = clf.identify(frame)
        timings["identify_ms"] = round((time.perf_counter() - t) * 1000, 1)
        timings["top1"] = [ident.top[0][0], round(float(ident.top[0][1]), 4)]
        timings["accepted"] = ident.accepted
        timings["total_ms"] = round(sum(
            v for k, v in timings.items() if k.endswith("_ms")), 1)
    finally:
        cap.release()
    return timings


def run_config(conf: dict, timeout: float = 900.0) -> dict:
    """Run one configuration in a fresh process (isolates XNNPACK env)."""
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(target=_bench_one, args=(conf, q), daemon=True)
    proc.start()
    try:
        result = q.get(timeout=timeout)
    except Exception:
        result = {**conf, "error": f"no result within {timeout:.0f}s"}
    finally:
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
    return result


def collect_images(spec: list[str]) -> list[str]:
    out: list[str] = []
    for s in spec:
        p = Path(s)
        if p.is_dir():
            out += [str(f) for f in sorted(p.iterdir())
                    if f.suffix.lower() in IMG_EXTS]
        elif p.is_file():
            out.append(str(p))
        else:
            print(f"warn: no such image path: {s}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, action="append",
                    help="model directory (repeatable; default: deploy + "
                         "mobilenetv2_qat if present)")
    ap.add_argument("--kernels", default="ref,default",
                    help="comma list: ref (XNNPACK off; 14-43x slower on "
                         "the UNO Q) and/or default (XNNPACK, the "
                         "runtime default)")
    ap.add_argument("--threads", default="4",
                    help="comma list of interpreter thread counts")
    ap.add_argument("--runs", type=int, default=10,
                    help="timed invokes per configuration")
    ap.add_argument("--images", nargs="*", default=[],
                    help="image files or a directory: also print top-3 per "
                         "image so device and laptop can be diffed. These "
                         "are NOT used for timing")
    ap.add_argument("--timing-image", default=None,
                    help="time this image instead of a synthetic 640x480 "
                         "camera-sized frame (preprocessing cost scales "
                         "with source resolution)")
    ap.add_argument("--camera-index", type=int, default=None,
                    help="also time the real camera path (open/flush/grab)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write all results here")
    args = ap.parse_args()

    model_dirs = args.model_dir or [d for d in DEFAULT_MODEL_DIRS
                                    if (d / "model_manifest.json").is_file()]
    if not model_dirs:
        print("no model directories found; pass --model-dir",
              file=sys.stderr)
        return 1
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    threads = [int(t) for t in args.threads.split(",") if t.strip()]
    images = collect_images(args.images)

    print(f"host     {platform.node()}  {platform.machine()}  "
          f"{platform.system()} {platform.release()}")
    print(f"python   {platform.python_version()}")
    try:
        import ai_edge_litert
        print(f"runtime  ai-edge-litert {ai_edge_litert.__version__}")
    except Exception:
        print("runtime  ai-edge-litert not importable (falling back)")
    print(f"configs  {len(model_dirs)} model(s) x {kernels} x {threads} "
          f"threads, {args.runs} runs each")
    if images:
        print(f"images   {len(images)} for prediction comparison")

    results = []
    for d in model_dirs:
        for k in kernels:
            for t in threads:
                conf = {"model_dir": str(d), "model": d.name, "kernels": k,
                        "threads": t, "runs": args.runs, "images": images,
                        "timing_image": args.timing_image,
                        "camera_index": args.camera_index}
                print(f"\n>>> {d.name}  kernels={k}  threads={t} ...",
                      flush=True)
                r = run_config(conf)
                results.append(r)
                if "error" in r:
                    print(f"    ERROR: {r['error']}")
                    continue
                print(f"    {r['describe']}")
                print(f"    timing frame: {r['timing_frame']}")
                print(f"    load {r['load_s']}s   "
                      f"preprocess {r['preprocess_total_ms']}ms total   "
                      f"invoke {r['invoke']['p50_ms']}ms x "
                      f"{r['views']} = {r['invoke_total_ms']}ms   "
                      f"identify {r['identify']['p50_ms']}ms"
                      + (f"   rss {r['rss_mb']:.0f}MB" if r.get("rss_mb")
                         else ""))
                print("    per view: " + "  ".join(
                    f"crop{v['crop']}{'+flip' if v['flip'] else ''}="
                    f"{v['p50_ms']}ms" for v in r["preprocess_per_view"]))
                if r.get("camera"):
                    print(f"    camera: {r['camera']}")
                for name, pred in (r.get("predictions") or {}).items():
                    top = "  ".join(f"{n}:{c * 100:.1f}%"
                                    for n, c in pred["top3"])
                    print(f"    {name:<28s} {top}"
                          + ("" if pred["accepted"] else
                             f"   [refused: {pred['reason']}]"))

    ok = [r for r in results if "error" not in r]
    if ok:
        print("\n=== summary (p50) ===")
        print(f"  {'model':<24s} {'kern':<8s} {'thr':>4s} {'MB':>6s} "
              f"{'invoke':>9s} {'prep all':>10s} {'identify':>10s} "
              f"{'Hz':>6s} {'RSS MB':>7s}")
        for r in sorted(ok, key=lambda r: r["identify"]["p50_ms"]):
            ident_ms = r["identify"]["p50_ms"]
            print(f"  {r['model']:<24s} {r['kernels']:<8s} "
                  f"{r['threads']:>4d} {r['model_mb']:>6.2f} "
                  f"{r['invoke']['p50_ms']:>8.1f}ms "
                  f"{r['preprocess_total_ms']:>9.1f}ms "
                  f"{ident_ms:>9.1f}ms {1000 / ident_ms:>6.2f} "
                  f"{(r.get('rss_mb') or 0):>7.0f}")
        best = min(ok, key=lambda r: r["identify"]["p50_ms"])
        print(f"\n  fastest identification: {best['model']} "
              f"({best['kernels']} kernels, {best['threads']} threads) at "
              f"{best['identify']['p50_ms']:.0f}ms")
        print("  Identification runs ONCE per session, so anything under a "
              "couple of seconds is fine;\n  prefer the more accurate model "
              "unless it is far slower. Check the predictions above\n  match "
              "the laptop before shipping a faster kernel.")

    if args.json:
        args.json.write_text(json.dumps(
            {"host": platform.node(), "machine": platform.machine(),
             "python": platform.python_version(), "results": results},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
