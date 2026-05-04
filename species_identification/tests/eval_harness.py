"""
Eval harness for RoboRanger retrieval quality including blurbs.

Loads eval_retrieval.yaml, runs each query through the retriever in
test_corpus.py, and emits both a per-item CSV and a console summary
showing pass/fail counts plus the cosine-similarity distribution
across the three expected categories (retrieve / refuse / blurb).

The point of this harness is NOT to get green checkmarks - it's to
expose where the retriever lives in similarity-space so you can pick
a refusal threshold that lives in the gap between "should retrieve"
and "should refuse" cases.

Usage:
    python eval_harness.py corpus.db eval_retrieval.yaml
    python eval_harness.py corpus.db eval_retrieval.yaml --threshold 0.60
    python eval_harness.py corpus.db eval_retrieval.yaml --out results.csv

Output CSV columns:
    id, query, species_id, expected, top1_cosine, top1_category,
    top1_keywords_hit, top1_text_preview, top2_cosine, top3_cosine,
    verdict, notes

Verdict logic (top-1 cosine sim vs threshold):
    expected=retrieve:
        PASS  if top1 >= threshold AND any expected_keyword appears
        SOFT  if top1 >= threshold but no keywords (semantic match only)
        FAIL  if top1 < threshold (would have been refused)
    expected=refuse:
        PASS  if top1 < threshold
        FAIL  if top1 >= threshold (would have been answered)
    expected=blurb:
        We don't run retrieval-pass/fail on these, since the intent
        router should intercept them before retrieval. We DO log the
        top1 cosine sim - useful diagnostic for whether retrieval
        would have answered "correctly enough" if the router missed.
        Verdict is always 'BLURB_PATH' (informational).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

# Reuse the retriever from test_corpus.py rather than reimplementing.
# This guarantees the harness measures the same code path the runtime
# will use - if the harness shows good numbers but the runtime doesn't,
# the divergence is somewhere else.
from test_corpus import EMBED_MODEL, open_db, retrieve, l2_to_cosine


DEFAULT_THRESHOLD = 0.5  # placeholder; the whole point is to TUNE this


def load_eval_set(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("eval_set", [])
    if not items:
        sys.exit(f"[FATAL] {path} has no eval_set entries")
    # Light validation - catch typos in the YAML before we run 50+ queries.
    valid_expected = {"retrieve", "refuse", "blurb"}
    for item in items:
        if item.get("expected") not in valid_expected:
            sys.exit(f"[FATAL] item {item.get('id')} has bad expected: "
                     f"{item.get('expected')}")
        if item["expected"] == "retrieve" and not item.get("expected_keywords"):
            print(f"[WARN] {item['id']} is `retrieve` but has no "
                  f"expected_keywords - keyword check will be skipped",
                  file=sys.stderr)
    return items


def keyword_hit(text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match. Cheap proxy for relevance."""
    if not keywords:
        return False
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def evaluate_item(conn, embedder, item: dict, threshold: float) -> dict:
    """Run one query, score it, return a row dict for the CSV."""
    species_id = item["species_id"]
    query = item["query"]
    expected = item["expected"]

    # `seashore` won't have chunks (it's the negative class).
    # Retrieving against it will return empty - that's the correct
    # behavior, and means the runtime should refuse before this point.
    results = retrieve(conn, embedder, species_id, query, k=3)

    if not results:
        # No chunks retrieved - either species has no corpus or the query
        # produced nothing. For `refuse` and `blurb` items this is fine;
        # for `retrieve` items it's a fail.
        top1_cos = top2_cos = top3_cos = float("nan")
        top1_cat = ""
        top1_text = ""
        top1_kw = False
    else:
        cos_sims = [l2_to_cosine(r[2]) for r in results]
        top1_cos = cos_sims[0]
        top2_cos = cos_sims[1] if len(cos_sims) > 1 else float("nan")
        top3_cos = cos_sims[2] if len(cos_sims) > 2 else float("nan")
        top1_cat = results[0][0]
        top1_text = results[0][1]
        top1_kw = keyword_hit(top1_text, item.get("expected_keywords") or [])

    # Verdict
    if expected == "blurb":
        verdict = "BLURB_PATH"
    elif expected == "retrieve":
        if top1_cos != top1_cos:  # NaN check
            verdict = "FAIL"
        elif top1_cos < threshold:
            verdict = "FAIL"
        elif top1_kw:
            verdict = "PASS"
        elif not item.get("expected_keywords"):
            # No keywords specified - pure threshold pass
            verdict = "PASS"
        else:
            verdict = "SOFT"
    elif expected == "refuse":
        if top1_cos != top1_cos:  # NaN, no chunks - correctly refused
            verdict = "PASS"
        elif top1_cos < threshold:
            verdict = "PASS"
        else:
            verdict = "FAIL"
    else:
        verdict = "?"

    return {
        "id":                  item["id"],
        "query":               query,
        "species_id":          species_id,
        "expected":            expected,
        "top1_cosine":         f"{top1_cos:.3f}" if top1_cos == top1_cos else "",
        "top1_category":       top1_cat,
        "top1_keywords_hit":   "yes" if top1_kw else "no",
        "top1_text_preview":   (top1_text[:200] + "...") if len(top1_text) > 200 else top1_text,
        "top2_cosine":         f"{top2_cos:.3f}" if top2_cos == top2_cos else "",
        "top3_cosine":         f"{top3_cos:.3f}" if top3_cos == top3_cos else "",
        "verdict":             verdict,
        "notes":               (item.get("notes") or "").replace("\n", " ").strip(),
    }


def print_summary(rows: list[dict], threshold: float) -> None:
    """
    Group results by expected category, show pass/fail counts and the
    cosine-similarity distribution. The distribution is the actually-
    useful output - it tells you whether your threshold lives in a
    real gap or whether retrieve and refuse cases overlap.
    """
    print("\n" + "=" * 72)
    print(f"SUMMARY  (threshold = {threshold:.3f})")
    print("=" * 72)

    by_expected = defaultdict(list)
    for r in rows:
        by_expected[r["expected"]].append(r)

    for expected in ("retrieve", "blurb", "refuse"):
        items = by_expected.get(expected, [])
        if not items:
            continue
        verdicts = Counter(r["verdict"] for r in items)
        cos_values = [float(r["top1_cosine"]) for r in items
                      if r["top1_cosine"]]

        print(f"\n  {expected.upper():9s}  n={len(items)}  verdicts={dict(verdicts)}")

        if cos_values:
            cos_values.sort()
            n = len(cos_values)
            mn = cos_values[0]
            mx = cos_values[-1]
            med = cos_values[n // 2]
            print(f"             top1 cosine sim:  "
                  f"min={mn:.3f}  median={med:.3f}  max={mx:.3f}")

            # Histogram-ish: how many items in each cosine band
            bands = [(0.0, 0.30, "weak    "),
                     (0.30, 0.50, "low     "),
                     (0.50, 0.65, "marginal"),
                     (0.65, 0.80, "good    "),
                     (0.80, 1.01, "strong  ")]
            print("             distribution:")
            for lo, hi, label in bands:
                n_in = sum(1 for v in cos_values if lo <= v < hi)
                bar = "#" * n_in
                print(f"               [{lo:.2f}-{hi:.2f}) {label} {n_in:3d}  {bar}")

    # Overlap check: where do retrieve and refuse cases live?
    retrieve_cos = sorted(float(r["top1_cosine"]) for r in by_expected.get("retrieve", [])
                          if r["top1_cosine"])
    refuse_cos = sorted(float(r["top1_cosine"]) for r in by_expected.get("refuse", [])
                        if r["top1_cosine"])
    if retrieve_cos and refuse_cos:
        retrieve_min = retrieve_cos[0]
        refuse_max = refuse_cos[-1]
        print()
        print("  THRESHOLD-BAND ANALYSIS")
        print(f"    Lowest 'retrieve' cosine: {retrieve_min:.3f}")
        print(f"    Highest 'refuse'  cosine: {refuse_max:.3f}")
        if retrieve_min > refuse_max:
            mid = (retrieve_min + refuse_max) / 2
            print(f"    -> Clean gap! Any threshold in "
                  f"[{refuse_max:.3f}, {retrieve_min:.3f}] separates them.")
            print(f"    -> Suggested threshold: {mid:.3f}")
        else:
            overlap_lo = retrieve_min
            overlap_hi = refuse_max
            n_retrieve_below = sum(1 for v in retrieve_cos if v <= overlap_hi)
            n_refuse_above = sum(1 for v in refuse_cos if v >= overlap_lo)
            print(f"    -> OVERLAP from {overlap_lo:.3f} to {overlap_hi:.3f}")
            print(f"       {n_retrieve_below} retrieve items fall in the overlap")
            print(f"       {n_refuse_above} refuse items fall in the overlap")
            print(f"    -> No clean threshold exists. Inspect the overlap items "
                  f"in the CSV to see whether the eval set or the retriever "
                  f"is at fault.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="path to corpus.db")
    ap.add_argument("eval_yaml", help="path to eval_retrieval.yaml")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"refusal threshold on top-1 cosine sim "
                         f"(default {DEFAULT_THRESHOLD})")
    ap.add_argument("--out", default="eval_results.csv",
                    help="output CSV path (default eval_results.csv)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    db_path = Path(args.db)
    eval_path = Path(args.eval_yaml)
    out_path = Path(args.out)

    if not db_path.exists():
        sys.exit(f"[FATAL] corpus DB not found: {db_path}")
    if not eval_path.exists():
        sys.exit(f"[FATAL] eval YAML not found: {eval_path}")

    items = load_eval_set(eval_path)
    print(f"Loaded {len(items)} eval items from {eval_path}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    conn = open_db(db_path)
    print(f"Opened corpus: {db_path}")
    print(f"Threshold: {args.threshold:.3f}\n")

    rows = []
    for i, item in enumerate(items, 1):
        row = evaluate_item(conn, embedder, item, args.threshold)
        rows.append(row)
        # One-line per item so you can watch it run
        cos = row["top1_cosine"] or "  -  "
        print(f"  [{i:3d}/{len(items)}]  {row['verdict']:11s} "
              f"cos={cos:>5}  {row['expected']:8s}  "
              f"{row['species_id'][:25]:25s}  {row['query'][:50]}")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    print_summary(rows, args.threshold)

    conn.close()


if __name__ == "__main__":
    main()