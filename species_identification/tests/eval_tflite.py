#!/usr/bin/env python3
"""
Evaluate on-device TFLite classifiers exactly the way the device runs them,
and calibrate temperature + the identification threshold on that output.

Replaces test_tflite.py, which hard-coded a 224px uint8 MobileNetV3-Small and
could not evaluate the models we actually ship.

Every image goes through TFLiteClassifierTTA (the runtime class): OpenCV
decode -> BGR->RGB -> TTA crops/flip -> INTER_AREA resize -> the manifest's
input_range -> quantize -> LiteRT. So these numbers are for the exported
model, its preprocessing contract, and TTA together — not for the float Keras
model.

Per model it reports, on the val and test splits:
  - top-1 / top-5, single view vs 4-view TTA
  - NLL and ECE of the TTA probabilities at the manifest's temperature and
    at a temperature refit on val (the device averages temperature-scaled
    softmax over views, so T is fit on exactly that)
  - the identification policy (negative class or conf < threshold -> refuse):
    the lowest threshold whose val precision >= --target-precision, and that
    policy's precision / coverage / seashore false-accepts on test
  - optional float-vs-int8 top-1 agreement (--keras), reference-kernel
    parity (--ref-parity) and a deliberate wrong-domain run (--input-range)

--write stores the refit temperature, chosen id_threshold and a summary in
the model's manifest. Logits are cached per model file hash, so re-running
(e.g. with a different --target-precision) is instant.

Usage (from the repo root):
    python species_identification/tests/eval_tflite.py \\
        --model-dir species_identification/outputs/mobilenetv3l \\
        --model-dir species_identification/outputs/mobilenetv2_qat --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
from tflite_classifier import (  # noqa: E402
    INPUT_RANGES,
    ModelManifest,
    TFLiteClassifierTTA,
    softmax,
    tta_probs,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
T_GRID = np.round(np.arange(0.25, 5.0001, 0.05), 4)
THR_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)


# ---------------------------------------------------------------------------
# Data + logits
# ---------------------------------------------------------------------------

def list_split(split_dir: Path, class_names: list[str], limit: int = 0
               ) -> tuple[list[Path], np.ndarray]:
    index = {c: i for i, c in enumerate(class_names)}
    paths, labels, unknown = [], [], []
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if d.name not in index:
            unknown.append(d.name)
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if limit:
            files = files[:limit]
        paths += files
        labels += [index[d.name]] * len(files)
    if unknown:
        print(f"  warn: {split_dir.name}: skipped folders not in "
              f"class_names: {unknown}")
    return paths, np.array(labels, dtype=np.int64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_WORKER_CLF: TFLiteClassifierTTA | None = None


def _worker_init(manifest_path: str, input_range: str | None,
                 reference_kernels: bool, threads: int) -> None:
    global _WORKER_CLF
    manifest = ModelManifest.load(manifest_path)
    if input_range:
        manifest.input_range = input_range
    _WORKER_CLF = TFLiteClassifierTTA(manifest, num_threads=threads,
                                      reference_kernels=reference_kernels)


def _worker_logits(item: tuple[int, str]) -> tuple[int, np.ndarray | None]:
    i, path = item
    bgr = cv2.imread(path)
    if bgr is None:
        return i, None
    return i, _WORKER_CLF.view_logits(bgr, tta=True)


def compute_logits(clf: TFLiteClassifierTTA, paths: list[Path], cache: Path,
                   args) -> tuple[np.ndarray, np.ndarray]:
    """
    (images, views, classes) logits and a valid-image mask. With
    --workers > 1, images are sharded across processes, each with its own
    interpreter (LiteRT's int8 kernels scale poorly past ~4 threads).
    """
    if cache.is_file():
        data = np.load(cache)
        if len(data["valid"]) == len(paths):
            return data["logits"], data["valid"]
    n_views, n_cls = len(clf.tta_views), len(clf.class_names)
    logits = np.zeros((len(paths), n_views, n_cls), dtype=np.float32)
    valid = np.ones(len(paths), dtype=bool)
    t0 = time.perf_counter()

    def record(i, out, done):
        if out is None:
            valid[i] = False
        else:
            logits[i] = out
        if done % 100 == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"    {done}/{len(paths)} images "
                  f"({rate:.2f} img/s)", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        items = [(i, str(p)) for i, p in enumerate(paths)]
        init = (str(clf.manifest.path), args.input_range, args.ref_kernels,
                args.threads)
        with mp.get_context("spawn").Pool(args.workers, _worker_init,
                                          init) as pool:
            for done, (i, out) in enumerate(
                    pool.imap_unordered(_worker_logits, items, chunksize=2),
                    1):
                record(i, out, done)
    else:
        for done, (i, p) in enumerate(enumerate(paths), 1):
            bgr = cv2.imread(str(p))
            record(i, None if bgr is None else clf.view_logits(bgr, tta=True),
                   done)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits, valid=valid)
    return logits, valid


def invoke_latency_ms(clf: TFLiteClassifierTTA, n: int = 5) -> float:
    """Single-invoke latency on this machine, measured in isolation."""
    x = clf.preprocess_view(np.zeros((480, 640, 3), dtype=np.uint8))
    clf.logits(x)
    t = time.perf_counter()
    for _ in range(n):
        clf.logits(x)
    return (time.perf_counter() - t) * 1000 / n


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def topk_acc(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([labels[i] in top[i] for i in range(len(labels))]))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(view_logits: np.ndarray, labels: np.ndarray) -> float:
    scores = [nll(tta_probs(view_logits, t), labels) for t in T_GRID]
    return float(T_GRID[int(np.argmin(scores))])


def policy(probs: np.ndarray, labels: np.ndarray, neg_idx: set[int],
           thr: float) -> dict:
    """
    Identification policy as the device applies it. Precision counts every
    accepted announcement (including a species announced on a seashore
    frame) as right or wrong; coverage is the share of species images that
    get announced at all.
    """
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    is_neg_label = np.isin(labels, list(neg_idx))
    accepted = (~np.isin(top, list(neg_idx))) & (conf >= thr)
    correct = accepted & (top == labels)
    n_acc = int(accepted.sum())
    n_species = int((~is_neg_label).sum())
    n_neg = int(is_neg_label.sum())
    return {
        "threshold": float(thr),
        "precision": correct.sum() / n_acc if n_acc else float("nan"),
        "coverage": (correct & ~is_neg_label).sum() / n_species
                    if n_species else float("nan"),
        "announced": (accepted & ~is_neg_label).sum() / n_species
                     if n_species else float("nan"),
        "negative_false_accept": (accepted & is_neg_label).sum() / n_neg
                                 if n_neg else float("nan"),
    }


def choose_threshold(probs, labels, neg_idx, target: float) -> dict:
    for thr in THR_GRID:
        p = policy(probs, labels, neg_idx, thr)
        if p["precision"] >= target:
            return p
    return policy(probs, labels, neg_idx, 1.0)


# ---------------------------------------------------------------------------
# Optional checks
# ---------------------------------------------------------------------------

def keras_agreement(clf, keras_path: Path, paths, int8_top1, batch=16):
    import keras

    try:  # QAT checkpoints need the tfmot scope to deserialize.
        import tensorflow_model_optimization as tfmot
        scope = tfmot.quantization.keras.quantize_scope()
    except ImportError:
        from contextlib import nullcontext
        scope = nullcontext()
    with scope:
        model = keras.models.load_model(keras_path, compile=False)
    preds = []
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            xs.append(clf.preprocess_float(rgb))
        preds.append(np.asarray(model(np.stack(xs), training=False)))
    float_top1 = np.concatenate(preds).argmax(axis=1)
    return float_top1, float(np.mean(float_top1 == int8_top1))


def ref_parity(manifest: ModelManifest, paths, n: int) -> dict:
    fast = TFLiteClassifierTTA(manifest, reference_kernels=False)
    ref = TFLiteClassifierTTA(manifest, reference_kernels=True)
    diffs, agree = [], 0
    for p in paths[:n]:
        bgr = cv2.imread(str(p))
        a, b = fast.view_logits(bgr, tta=False)[0], ref.view_logits(
            bgr, tta=False)[0]
        diffs.append(float(np.max(np.abs(a - b))))
        agree += int(a.argmax() == b.argmax())
    return {"images": min(n, len(paths)), "top1_agreement": agree / max(
        min(n, len(paths)), 1), "max_abs_logit_diff": max(diffs or [0.0])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(model_dir: Path, args) -> dict:
    manifest = ModelManifest.load(model_dir)
    label = model_dir.name
    if args.input_range:
        manifest.input_range = args.input_range
        label += f" [input_range forced to {args.input_range}]"
    clf = TFLiteClassifierTTA(manifest, reference_kernels=args.ref_kernels)
    print(f"\n=== {label} ===\n  {clf.describe()}", flush=True)
    latency_ms = invoke_latency_ms(clf)

    neg_idx = {clf.class_names.index(c) for c in clf.negative_classes}
    sha = _sha256(manifest.model_path)[:16]
    tag = (f"{sha}_{manifest.input_range}_ref{int(args.ref_kernels)}"
           f"_lim{args.limit_per_class}")

    splits = {}
    for split in ("val", "test"):
        paths, labels = list_split(args.splits / split, clf.class_names,
                                   args.limit_per_class)
        print(f"  {split}: {len(paths)} images", flush=True)
        logits, valid = compute_logits(
            clf, paths, args.cache_dir / f"{tag}_{split}.npz", args)
        splits[split] = {"paths": [p for p, v in zip(paths, valid) if v],
                         "labels": labels[valid], "logits": logits[valid],
                         "ms": latency_ms}

    t_old = float(manifest.temperature)
    t_fit = fit_temperature(splits["val"]["logits"], splits["val"]["labels"])
    result = {"model_dir": str(model_dir).replace("\\", "/"),
              "describe": clf.describe(), "temperature_before": t_old,
              "temperature_fit_val": t_fit, "splits": {}}

    val_probs_fit = tta_probs(splits["val"]["logits"], t_fit)
    chosen = choose_threshold(val_probs_fit, splits["val"]["labels"],
                              neg_idx, args.target_precision)
    result["id_threshold"] = chosen["threshold"]

    for split, s in splits.items():
        y, lg = s["labels"], s["logits"]
        single = softmax(lg[:, 0, :])
        tta_old = tta_probs(lg, t_old)
        tta_fit = tta_probs(lg, t_fit)
        r = {
            "n": int(len(y)),
            "ms_per_invoke": round(s["ms"], 2),
            "single_top1": topk_acc(single, y, 1),
            "tta_top1": topk_acc(tta_fit, y, 1),
            "tta_top5": topk_acc(tta_fit, y, 5),
            "nll_T_before": nll(tta_old, y), "ece_T_before": ece(tta_old, y),
            "nll_T_fit": nll(tta_fit, y), "ece_T_fit": ece(tta_fit, y),
            "policy_before": policy(tta_old, y, neg_idx,
                                    float(manifest.id_threshold)),
            "policy_chosen": policy(tta_fit, y, neg_idx, chosen["threshold"]),
        }
        result["splits"][split] = r

    if args.keras:
        test = splits["test"]
        int8_top1 = tta_probs(test["logits"][:, :1, :], 1.0).argmax(axis=1)
        float_top1, agree = keras_agreement(clf, args.keras, test["paths"],
                                            int8_top1)
        result["keras"] = {
            "path": str(args.keras).replace("\\", "/"),
            "float_single_top1_test": float(np.mean(
                float_top1 == test["labels"])),
            "tflite_vs_float_top1_agreement_test": agree,
        }
    if args.ref_parity:
        result["ref_parity"] = ref_parity(manifest, splits["test"]["paths"],
                                          args.ref_parity)

    print_report(result, args.target_precision)

    if args.write and not args.input_range:
        test = result["splits"]["test"]
        manifest.temperature = t_fit
        manifest.id_threshold = chosen["threshold"]
        manifest.eval = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "splits": str(args.splits).replace("\\", "/"),
            "runtime": "ai-edge-litert, "
                       + ("reference kernels" if args.ref_kernels
                          else "default kernels"),
            "temperature_fit_on": "val, TTA-averaged softmax of this exported model (min NLL)",
            "id_threshold_rule": f"lowest threshold with val precision >= "
                                 f"{args.target_precision}",
            "test_single_top1": round(test["single_top1"], 4),
            "test_tta_top1": round(test["tta_top1"], 4),
            "test_tta_top5": round(test["tta_top5"], 4),
            "test_ece": round(test["ece_T_fit"], 4),
            "test_precision": round(test["policy_chosen"]["precision"], 4),
            "test_coverage": round(test["policy_chosen"]["coverage"], 4),
            "test_negative_false_accept": round(
                test["policy_chosen"]["negative_false_accept"], 4),
            **({"tflite_vs_float_top1_agreement_test": round(
                result["keras"]["tflite_vs_float_top1_agreement_test"], 4)}
               if "keras" in result else {}),
        }
        manifest.source["notes"] = (
            "temperature and id_threshold fit by tests/eval_tflite.py on "
            "this model's TTA-averaged val output; see eval")
        manifest.save()
        print(f"  wrote temperature={t_fit} id_threshold="
              f"{chosen['threshold']} to {manifest.path}")
    return result


def _fmt(x) -> str:
    return "   n/a" if x != x else f"{100 * x:5.1f}%"  # NaN-safe


def print_report(r: dict, target: float) -> None:
    print(f"  temperature: manifest {r['temperature_before']:.2f} -> "
          f"fit on val {r['temperature_fit_val']:.2f}")
    print(f"  {'split':5s} {'single':>7s} {'TTA':>7s} {'top5':>7s} "
          f"{'ECE old':>8s} {'ECE fit':>8s}  ms/inv")
    for split, s in r["splits"].items():
        print(f"  {split:5s} {_fmt(s['single_top1']):>7s} "
              f"{_fmt(s['tta_top1']):>7s} {_fmt(s['tta_top5']):>7s} "
              f"{_fmt(s['ece_T_before']):>8s} {_fmt(s['ece_T_fit']):>8s}  "
              f"{s['ms_per_invoke']:.1f}")
    print(f"  identification policy (target val precision {target:.0%}):")
    print(f"  {'split':5s} {'policy':22s} {'prec':>7s} {'coverage':>9s} "
          f"{'neg FA':>7s}")
    for split, s in r["splits"].items():
        for name, key in (("before (manifest T/thr)", "policy_before"),
                          ("chosen", "policy_chosen")):
            p = s[key]
            label = f"{name[:14]} thr={p['threshold']:.2f}"
            print(f"  {split:5s} {label:22s} {_fmt(p['precision']):>7s} "
                  f"{_fmt(p['coverage']):>9s} "
                  f"{_fmt(p['negative_false_accept']):>7s}")
    if "keras" in r:
        k = r["keras"]
        print(f"  float Keras single-view top-1 (test): "
              f"{_fmt(k['float_single_top1_test'])}; tflite agrees with float "
              f"on {_fmt(k['tflite_vs_float_top1_agreement_test'])}")
    if "ref_parity" in r:
        p = r["ref_parity"]
        print(f"  reference-kernel parity on {p['images']} images: top-1 "
              f"agreement {_fmt(p['top1_agreement'])}, max |logit diff| "
              f"{p['max_abs_logit_diff']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, action="append", required=True,
                    help="directory with model_manifest.json (repeatable)")
    ap.add_argument("--splits", type=Path,
                    default=Path("species_identification/cnn/_splits"))
    ap.add_argument("--keras", type=Path, default=None,
                    help="float Keras checkpoint for int8 agreement "
                         "(single --model-dir only)")
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes to shard images across")
    ap.add_argument("--threads", type=int, default=2,
                    help="interpreter threads per worker (with --workers > 1)")
    ap.add_argument("--ref-kernels", action="store_true",
                    help="use the device's reference kernels for everything "
                         "(slow)")
    ap.add_argument("--ref-parity", type=int, default=0,
                    help="compare default vs reference kernels on N images")
    ap.add_argument("--input-range", choices=INPUT_RANGES, default=None,
                    help="force a (wrong) input domain to prove the contract "
                         "matters; never written")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("species_identification/outputs/"
                                 ".eval_cache"))
    ap.add_argument("--json", type=Path, default=None,
                    help="write all results to this JSON file")
    ap.add_argument("--write", action="store_true",
                    help="store fitted temperature/id_threshold/eval summary "
                         "in each manifest")
    args = ap.parse_args()
    if args.keras and len(args.model_dir) != 1:
        ap.error("--keras needs exactly one --model-dir")

    results = [evaluate(d, args) for d in args.model_dir]

    if len(results) > 1:
        print("\n=== comparison (test split, tflite + TTA, refit T, chosen "
              "threshold) ===")
        print(f"  {'model':28s} {'top1':>7s} {'top5':>7s} {'ECE':>7s} "
              f"{'prec':>7s} {'cover':>7s} {'negFA':>7s} {'thr':>5s} "
              f"{'T':>5s} {'ms':>6s}")
        for r in results:
            s = r["splits"]["test"]
            p = s["policy_chosen"]
            print(f"  {Path(r['model_dir']).name:28s} "
                  f"{_fmt(s['tta_top1']):>7s} {_fmt(s['tta_top5']):>7s} "
                  f"{_fmt(s['ece_T_fit']):>7s} {_fmt(p['precision']):>7s} "
                  f"{_fmt(p['coverage']):>7s} "
                  f"{_fmt(p['negative_false_accept']):>7s} "
                  f"{r['id_threshold']:5.2f} {r['temperature_fit_val']:5.2f} "
                  f"{s['ms_per_invoke']:6.1f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
