"""
Diagnostic script for the RoboRanger species classifier.

Loads an existing trained checkpoint and produces:
    1. Overall top-1 and top-5 accuracy on the test split.
    2. Per-class precision, recall, F1, support.
    3. A full confusion matrix (saved as CSV + PNG).
    4. Top-K "most confused pairs" — which species are getting mixed up.
    5. A list of the worst-performing classes ranked by recall.
    6. Per-class top-1 + top-5 accuracy (top-5 reveals "model knew the answer
       was in there somewhere", which is useful — it means more data on those
       classes will help, vs. classes where the model has no clue).

Reads the same _splits/ directory the training script created, so splits are
identical to what the model was trained on. If _splits/ has been deleted,
rebuild it with the same SEED before running this.

Usage:
    python diagnose.py
    python diagnose.py --model ../outputs/best.keras --splits _splits
"""

import argparse
import json
import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Config — matches the training script defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "../outputs/best.keras"
DEFAULT_SPLITS   = "_splits"
DEFAULT_CLASSES  = "../outputs/class_names.json"
DEFAULT_OUT_DIR  = "../outputs/diagnostics"
DEFAULT_TEMP     = "../outputs/temperature.json"
IMG_SIZE         = 320
BATCH_SIZE       = 16


# ---------------------------------------------------------------------------
# Data loading — must match training-time preprocessing exactly
# ---------------------------------------------------------------------------
def load_test_set(splits_root, class_names):
    test_ds = keras.utils.image_dataset_from_directory(
        pathlib.Path(splits_root) / "test",
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return test_ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Inference — collect predictions and labels
# ---------------------------------------------------------------------------
def collect_predictions(model, ds, temperature=1.0):
    """Run inference and return (y_true, y_pred_top1, y_pred_top5, probs).

    `probs` is the softmax-normalized prediction matrix (N, C). Temperature
    scaling is applied to the logits before softmax — matches what the
    inference pipeline does on the Uno Q.
    """
    all_logits, all_labels = [], []
    for imgs, labels in ds:
        logits = model(imgs, training=False).numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Apply temperature scaling and softmax — same as inference time.
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)

    y_true = labels.argmax(axis=1)
    # Top-5 predictions, sorted descending by prob.
    top5 = np.argsort(-probs, axis=1)[:, :5]
    y_pred = top5[:, 0]
    return y_true, y_pred, top5, probs


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------
def per_class_metrics(y_true, y_pred, top5, num_classes):
    """Return a dict per class: precision, recall, F1, top-1, top-5, support."""
    metrics = {}
    for c in range(num_classes):
        true_mask = y_true == c
        pred_mask = y_pred == c
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        support = int(np.sum(true_mask))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Top-5 accuracy for this class: of examples whose true label is c,
        # how many had c in their top-5 predictions?
        if support > 0:
            top5_hits = np.sum(np.any(top5[true_mask] == c, axis=1))
            top5_acc = float(top5_hits) / support
            top1_acc = recall  # by definition
        else:
            top5_acc = 0.0
            top1_acc = 0.0

        metrics[c] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "top1":      top1_acc,
            "top5":      top5_acc,
            "support":   support,
            "tp": tp, "fp": fp, "fn": fn,
        }
    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def most_confused_pairs(cm, class_names, top_k=20):
    """Find the top-K (true, predicted) off-diagonal pairs by count.

    These are the species the model conflates most often. Often points at
    visually similar pairs that need either more training data or higher
    image resolution to distinguish.
    """
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    return [
        {
            "count": int(count),
            "true":  class_names[i],
            "pred":  class_names[j],
        }
        for count, i, j in pairs[:top_k]
    ]


# ---------------------------------------------------------------------------
# Plotting — optional, gracefully degrade if matplotlib isn't installed
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, out_path, normalize=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping confusion matrix PNG.")
        return

    if normalize:
        # Row-normalize: each row sums to 1. Reveals patterns even with
        # class imbalance.
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        data = cm.astype(np.float64) / row_sums
        title = "Confusion matrix (row-normalized)"
        fmt = ".2f"
    else:
        data = cm
        title = "Confusion matrix (counts)"
        fmt = "d"

    # Scale figure size to class count so labels stay readable.
    n = len(class_names)
    size = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix plot: {out_path}")


# ---------------------------------------------------------------------------
# Report formatting — text output meant to be scanned quickly
# ---------------------------------------------------------------------------
def print_report(metrics, class_names, y_true, y_pred, top5):
    n = len(class_names)
    total = len(y_true)
    top1_correct = int(np.sum(y_true == y_pred))
    top5_correct = int(np.sum([t in row for t, row in zip(y_true, top5)]))

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"Total test examples: {total}")
    print(f"Classes:             {n}")
    print(f"Top-1 accuracy:      {top1_correct / total:.4f}  "
          f"({top1_correct}/{total})")
    print(f"Top-5 accuracy:      {top5_correct / total:.4f}  "
          f"({top5_correct}/{total})")

    # Macro vs. weighted: macro = unweighted class average (every class equal),
    # weighted = support-weighted. Big gap between them = class imbalance issue.
    supports = np.array([metrics[c]["support"] for c in range(n)])
    f1s      = np.array([metrics[c]["f1"]      for c in range(n)])
    precs    = np.array([metrics[c]["precision"] for c in range(n)])
    recalls  = np.array([metrics[c]["recall"]  for c in range(n)])

    print(f"\nMacro    precision/recall/F1: "
          f"{precs.mean():.3f} / {recalls.mean():.3f} / {f1s.mean():.3f}")
    if supports.sum() > 0:
        w = supports / supports.sum()
        print(f"Weighted precision/recall/F1: "
              f"{(precs * w).sum():.3f} / {(recalls * w).sum():.3f} / "
              f"{(f1s * w).sum():.3f}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS (sorted by recall, worst first)")
    print("=" * 70)
    print(f"{'class':<35} {'top1':>6} {'top5':>6} {'prec':>6} "
          f"{'rec':>6} {'F1':>6} {'n':>4}")
    print("-" * 70)
    by_recall = sorted(range(n), key=lambda c: metrics[c]["recall"])
    for c in by_recall:
        m = metrics[c]
        name = class_names[c][:34]
        print(f"{name:<35} {m['top1']:>6.3f} {m['top5']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>4}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS HINTS")
    print("=" * 70)
    # Classes where top-5 is high but top-1 is low — model can see the answer
    # but is confusing it with a similar class. More data on the confused
    # pair will help.
    confused = [
        (c, metrics[c]["top5"] - metrics[c]["top1"])
        for c in range(n) if metrics[c]["support"] >= 2
    ]
    confused.sort(key=lambda x: -x[1])
    print("\nClasses where top-5 >> top-1 (model is close but picking wrong):")
    print("  Likely fix: more data on this class + its lookalikes.")
    for c, gap in confused[:10]:
        if gap < 0.1:
            break
        print(f"  {class_names[c]:<35} top1={metrics[c]['top1']:.3f}  "
              f"top5={metrics[c]['top5']:.3f}  gap={gap:.3f}")

    # Classes where even top-5 is bad — model has no representation for
    # these. Architecture or pretraining won't save you. Need more data.
    lost = [c for c in range(n)
            if metrics[c]["support"] >= 2 and metrics[c]["top5"] < 0.5]
    if lost:
        print("\nClasses with top-5 < 50% (model has no clue):")
        print("  Likely fix: substantially more diverse training images, "
              "or check label quality.")
        for c in lost:
            print(f"  {class_names[c]:<35} top1={metrics[c]['top1']:.3f}  "
                  f"top5={metrics[c]['top5']:.3f}  n={metrics[c]['support']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=DEFAULT_MODEL)
    ap.add_argument("--splits",  default=DEFAULT_SPLITS)
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--temp",    default=DEFAULT_TEMP,
                    help="path to temperature.json; ignored if missing")
    ap.add_argument("--out",     default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-confused", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading class names from {args.classes}")
    with open(args.classes) as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"  {num_classes} classes")

    # Temperature is optional — defaults to 1.0 if not found.
    T = 1.0
    if os.path.exists(args.temp):
        with open(args.temp) as f:
            T = float(json.load(f)["temperature"])
        print(f"Loaded temperature T = {T:.3f}")
    else:
        print("No temperature.json found, using T = 1.0 (raw logits).")

    print(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model, compile=False)

    print(f"Loading test split from {args.splits}/test")
    test_ds = load_test_set(args.splits, class_names)

    print("Running inference on test set...")
    y_true, y_pred, top5, probs = collect_predictions(model, test_ds, T)

    print("Computing per-class metrics...")
    metrics = per_class_metrics(y_true, y_pred, top5, num_classes)

    print("Building confusion matrix...")
    cm = build_confusion_matrix(y_true, y_pred, num_classes)

    # Print everything to stdout.
    print_report(metrics, class_names, y_true, y_pred, top5)

    # Top confused pairs.
    pairs = most_confused_pairs(cm, class_names, top_k=args.top_confused)
    print("\n" + "=" * 70)
    print(f"TOP {args.top_confused} CONFUSED PAIRS (true -> predicted)")
    print("=" * 70)
    print("These pairs are the highest-leverage place to add training data.")
    print(f"{'count':>5}  {'true':<35} -> {'predicted':<35}")
    print("-" * 80)
    for p in pairs:
        print(f"{p['count']:>5}  {p['true'][:34]:<35} -> {p['pred'][:34]:<35}")

    # Save everything to disk for later reference.
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    cm_csv = pathlib.Path(args.out) / "confusion_matrix.csv"
    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    print(f"  {cm_csv}")

    metrics_json = pathlib.Path(args.out) / "per_class_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump({class_names[c]: metrics[c] for c in range(num_classes)},
                  f, indent=2)
    print(f"  {metrics_json}")

    pairs_json = pathlib.Path(args.out) / "confused_pairs.json"
    with open(pairs_json, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"  {pairs_json}")

    cm_png = pathlib.Path(args.out) / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_png, normalize=True)


if __name__ == "__main__":
    main()