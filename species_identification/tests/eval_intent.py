"""
Evaluation harness for the intent classifier.

Run against a labeled set of realistic queries. Reports:
  - per-intent accuracy
  - confusion matrix
  - score / margin distributions (use these to tune thresholds)
  - mismatches with full score breakdown

Usage:
    # with the real bge-small embedder (recommended):
    python eval_intent.py

    # or with a custom labeled file:
    python eval_intent.py --queries my_queries.json

The labeled set below is the starting fixture. Add real queries from your
own usage as you go — the more, the better the threshold tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

sys.path.insert(1, "../species_identification/llm-tuning")
from intent import Intent, IntentClassifier, PROTOTYPES
from wildlife_gate import WildlifeGate


# ---------------------------------------------------------------------------
# Labeled queries. Keep these phrased the way real visitors would phrase them
# (informal, sometimes ambiguous). Add edge cases: short queries, typos,
# off-topic queries, queries that span multiple intents.
# ---------------------------------------------------------------------------
LABELED_QUERIES: list[tuple[str, Intent]] = [
    # DESCRIPTION
    ("what is this?", Intent.DESCRIPTION),
    ("tell me about this animal", Intent.DESCRIPTION),
    ("describe it", Intent.DESCRIPTION),
    ("what kind of bird is that", Intent.DESCRIPTION),
    ("give me an overview", Intent.DESCRIPTION),
    ("what does it look like", Intent.DESCRIPTION),

    # DANGER
    ("is it venomous", Intent.DANGER),
    ("will this thing bite me", Intent.DANGER),
    ("is it safe to get closer", Intent.DANGER),
    ("dangerous?", Intent.DANGER),
    ("can it hurt my dog", Intent.DANGER),
    ("should I be worried", Intent.DANGER),
    ("is this poisonous to touch", Intent.DANGER),

    # DIET
    ("what does it eat", Intent.DIET),
    ("what does it hunt", Intent.DIET),
    ("does it eat insects", Intent.DIET),
    ("is it a predator", Intent.DIET),
    ("what's its food", Intent.DIET),

    # HABITAT
    ("where does it live", Intent.HABITAT),
    ("is it from around here", Intent.HABITAT),
    ("what's its range", Intent.HABITAT),
    ("where would I find one", Intent.HABITAT),
    ("native to california?", Intent.HABITAT),

    # BEHAVIOR
    ("is it nocturnal", Intent.BEHAVIOR),
    ("is it aggressive", Intent.BEHAVIOR),
    ("how does it hunt", Intent.BEHAVIOR),
    ("does it live in groups", Intent.BEHAVIOR),
    ("what does it do during the day", Intent.BEHAVIOR),

    # IDENTIFICATION
    ("how do I tell it apart from a coyote", Intent.IDENTIFICATION),
    ("what makes this one different", Intent.IDENTIFICATION),
    ("key features to look for", Intent.IDENTIFICATION),
    ("how can I be sure it's this one", Intent.IDENTIFICATION),

    # CONSERVATION
    ("is it endangered", Intent.CONSERVATION),
    ("is this rare", Intent.CONSERVATION),
    ("is it a protected species", Intent.CONSERVATION),
    ("what's the conservation status", Intent.CONSERVATION),

    # OTHER — these should classify as OTHER (low confidence)
    ("what time does the park close", Intent.OTHER),
    ("where is the bathroom", Intent.OTHER),
    ("how do I get back to the trailhead", Intent.OTHER),
    ("asdfasdf", Intent.OTHER),
    ("the weather is nice today", Intent.OTHER),
]


# ---------------------------------------------------------------------------
# Embedder loading. Try bge-small via sentence-transformers; fall back with a
# clear error if not available.
# ---------------------------------------------------------------------------
def load_real_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.", file=sys.stderr)
        print("Install with: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    print("Loading BAAI/bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    class _Wrapper:
        def encode(self, texts: Sequence[str]):
            return model.encode(list(texts), normalize_embeddings=True)

    return _Wrapper()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def evaluate_gate(gate: WildlifeGate,
                  labeled: Sequence[tuple[str, Intent]]) -> dict:
    """Evaluate the wildlife gate. OTHER queries should be rejected; all
    other intents should be accepted."""
    results = []
    for query, expected in labeled:
        r = gate.check(query)
        should_accept = expected != Intent.OTHER
        correct = r.in_domain == should_accept
        results.append({
            "query": query,
            "expected_in_domain": should_accept,
            "predicted_in_domain": r.in_domain,
            "confidence": r.confidence,
            "wildlife_score": r.wildlife_score,
            "off_topic_score": r.off_topic_score,
            "margin": r.margin,
            "correct": correct,
        })
    return results


def print_gate_report(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\n=== WildlifeGate: {correct}/{total} ({100*correct/total:.1f}%) ===")

    # Two error types:
    #   - false reject: in-domain query rejected (lost user)
    #   - false accept: off-topic query accepted (model will hallucinate)
    fr = [r for r in results
          if r["expected_in_domain"] and not r["predicted_in_domain"]]
    fa = [r for r in results
          if not r["expected_in_domain"] and r["predicted_in_domain"]]

    print(f"False rejects (in-domain queries refused): {len(fr)}")
    for r in fr:
        print(f"  {r['query']!r}")
        print(f"    wild={r['wildlife_score']:+.3f}  "
              f"off={r['off_topic_score']:+.3f}  "
              f"margin={r['margin']:+.3f}  ({r['confidence']})")

    print(f"\nFalse accepts (off-topic queries let through): {len(fa)}")
    for r in fa:
        print(f"  {r['query']!r}")
        print(f"    wild={r['wildlife_score']:+.3f}  "
              f"off={r['off_topic_score']:+.3f}  "
              f"margin={r['margin']:+.3f}  ({r['confidence']})")

    # Margin distribution — useful for retuning ACCEPT_MARGIN / AMBIG_MARGIN.
    in_margins = sorted(r["margin"] for r in results
                        if r["expected_in_domain"])
    off_margins = sorted(r["margin"] for r in results
                         if not r["expected_in_domain"])
    if in_margins and off_margins:
        print("\nMargin distribution (wildlife - off_topic):")
        print(f"  in-domain queries:  min={in_margins[0]:+.3f}  "
              f"median={in_margins[len(in_margins)//2]:+.3f}  "
              f"max={in_margins[-1]:+.3f}")
        print(f"  off-topic queries:  min={off_margins[0]:+.3f}  "
              f"median={off_margins[len(off_margins)//2]:+.3f}  "
              f"max={off_margins[-1]:+.3f}")
        # If the worst in-domain margin is greater than the best off-topic
        # margin, you have perfect separation — you can lock in a threshold
        # between them.
        if in_margins[0] > off_margins[-1]:
            print(f"  -> perfect separation; any threshold in "
                  f"({off_margins[-1]:+.3f}, {in_margins[0]:+.3f}) works")


def evaluate(clf: IntentClassifier,
             labeled: Sequence[tuple[str, Intent]]) -> dict:
    results = []
    for query, expected in labeled:
        r = clf.classify(query)
        results.append({
            "query": query,
            "expected": expected,
            "predicted": r.intent,
            "confidence": r.confidence,
            "score": r.score,
            "margin": r.margin,
            "all_scores": r.all_scores,
            "correct": r.intent == expected,
        })
    return results


def print_report(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\n=== Overall: {correct}/{total} ({100*correct/total:.1f}%) ===\n")

    # Per-intent accuracy
    by_intent: dict[Intent, list[dict]] = defaultdict(list)
    for r in results:
        by_intent[r["expected"]].append(r)

    print("Per-intent accuracy:")
    for intent in Intent:
        rs = by_intent.get(intent, [])
        if not rs:
            continue
        c = sum(1 for r in rs if r["correct"])
        print(f"  {intent.value:15s} {c}/{len(rs)}")

    # Confusion matrix
    print("\nConfusion (rows=expected, cols=predicted):")
    intents = list(Intent)
    col_w = 6
    header = " " * 16 + "".join(f"{i.value[:col_w-1]:>{col_w}}" for i in intents)
    print(header)
    for exp in intents:
        rs = by_intent.get(exp, [])
        if not rs:
            continue
        counts = {pred: 0 for pred in intents}
        for r in rs:
            counts[r["predicted"]] += 1
        row = f"  {exp.value:14s}" + "".join(f"{counts[p]:>{col_w}}" for p in intents)
        print(row)

    # Score distribution per expected intent (when correct)
    print("\nScore distribution on correct predictions (use to tune thresholds):")
    print(f"  {'intent':15s} {'n':>3s}  {'min':>6s}  {'med':>6s}  {'max':>6s}  "
          f"{'min_margin':>10s}")
    for intent in Intent:
        rs = [r for r in by_intent.get(intent, []) if r["correct"]]
        if not rs:
            continue
        scores = sorted(r["score"] for r in rs)
        margins = sorted(r["margin"] for r in rs)
        med = scores[len(scores) // 2]
        print(f"  {intent.value:15s} {len(rs):>3d}  "
              f"{scores[0]:>+6.3f}  {med:>+6.3f}  {scores[-1]:>+6.3f}  "
              f"{margins[0]:>+10.3f}")

    # Score distribution for OTHER (these should be LOW)
    other_rs = by_intent.get(Intent.OTHER, [])
    if other_rs:
        print("\nOFF-TOPIC queries — top scores (these should stay below 0.55):")
        for r in other_rs:
            top_intent, top_score = max(
                r["all_scores"].items(), key=lambda kv: kv[1]
            )
            flag = "OK " if r["predicted"] == Intent.OTHER else "BAD"
            print(f"  [{flag}] {r['query']!r:45s} -> "
                  f"top={top_intent.value} score={top_score:+.3f}")

    # Mismatches
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n{len(wrong)} mismatch(es):")
        for r in wrong:
            print(f"  {r['query']!r}")
            print(f"    expected={r['expected'].value}  "
                  f"predicted={r['predicted'].value}  "
                  f"({r['confidence']}, score={r['score']:+.3f}, "
                  f"margin={r['margin']:+.3f})")
            top3 = sorted(r["all_scores"].items(), key=lambda kv: -kv[1])[:3]
            print("    top3: " + ", ".join(
                f"{i.value}={s:+.3f}" for i, s in top3
            ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path,
                    help="JSON file with [{query, intent}] entries (overrides default)")
    args = ap.parse_args()

    labeled = LABELED_QUERIES
    if args.queries:
        data = json.loads(args.queries.read_text())
        labeled = [(d["query"], Intent(d["intent"])) for d in data]

    embedder = load_real_embedder()

    print("\n" + "#" * 60)
    print("# WildlifeGate evaluation")
    print("#" * 60)
    gate = WildlifeGate(embedder)
    gate_results = evaluate_gate(gate, labeled)
    print_gate_report(gate_results)

    print("\n" + "#" * 60)
    print("# IntentClassifier evaluation")
    print("# (run on the SAME labeled set; OTHER queries will be miscategorized")
    print("#  here because the gate is what's supposed to catch them)")
    print("#" * 60)
    clf = IntentClassifier(embedder)
    results = evaluate(clf, labeled)
    print_report(results)


if __name__ == "__main__":
    main()