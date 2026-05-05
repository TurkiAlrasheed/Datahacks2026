"""
Diff two saved runs from eval_e2e.py.

The point: identify cases where laptop and Uno Q diverge in ways that matter
(pass/fail mismatch, large length deltas, big latency gaps), so you don't
have to eyeball 5+ pairs of paragraphs.

Usage:
    python diff_runs.py runs/laptop.json runs/unoq.json
    python diff_runs.py runs/laptop.json runs/unoq.json --show-text

Exit code:
    0  no pass/fail divergence
    1  one or more cases passed in one run and failed in the other
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _kw_overlap(a: str, b: str) -> float:
    """Cheap token-overlap as a stand-in for semantic similarity. Not great,
    but good enough to flag 'these answers said totally different things'."""
    aw = set(w.lower().strip(".,!?:;") for w in a.split() if len(w) > 3)
    bw = set(w.lower().strip(".,!?:;") for w in b.split() if len(w) > 3)
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--show-text", action="store_true",
                    help="print full response text for diverging cases")
    args = ap.parse_args()

    a = json.loads(Path(args.run_a).read_text())
    b = json.loads(Path(args.run_b).read_text())

    print(f"A: {a['backend']} ({a.get('model')}) on {a.get('host')}")
    print(f"B: {b['backend']} ({b.get('model')}) on {b.get('host')}")

    if a.get("sampling") != b.get("sampling"):
        print("\nWARNING: sampling params differ between runs:")
        print(f"  A: {a.get('sampling')}")
        print(f"  B: {b.get('sampling')}")
        print("  Differences below may be sampler artifacts, not real drift.")

    by_case_a = {r["case"]: r for r in a["results"]}
    by_case_b = {r["case"]: r for r in b["results"]}
    cases = sorted(set(by_case_a) | set(by_case_b))

    pa = sum(1 for r in a["results"] if r["passed"])
    pb = sum(1 for r in b["results"] if r["passed"])
    print(f"\nPass rate: A={pa}/{len(a['results'])}  "
          f"B={pb}/{len(b['results'])}")

    la = [r["latency_s"] for r in a["results"]
          if r["latency_s"] is not None and r["error"] is None]
    lb = [r["latency_s"] for r in b["results"]
          if r["latency_s"] is not None and r["error"] is None]
    if la and lb:
        print(f"Median latency: A={sorted(la)[len(la)//2]:.2f}s  "
              f"B={sorted(lb)[len(lb)//2]:.2f}s")

    print("\nPer-case comparison:")
    print(f"  {'case':45s} {'A':>4s} {'B':>4s} {'overlap':>8s} "
          f"{'lat_A':>6s} {'lat_B':>6s}")

    pf_divergence = 0
    semantic_drift = []

    for c in cases:
        ra = by_case_a.get(c)
        rb = by_case_b.get(c)
        if ra is None or rb is None:
            tag = "ONLY-A" if rb is None else "ONLY-B"
            print(f"  {c:45s} {tag}")
            continue

        a_pass = "P" if ra["passed"] else "F"
        b_pass = "P" if rb["passed"] else "F"
        overlap = _kw_overlap(ra["response"], rb["response"])

        flag = ""
        if ra["passed"] != rb["passed"]:
            flag = "  <-- DIVERGE"
            pf_divergence += 1
        elif overlap < 0.4:
            flag = "  <-- low overlap"
            semantic_drift.append(c)

        print(f"  {c:45s} {a_pass:>4s} {b_pass:>4s} {overlap:>8.2f} "
              f"{ra['latency_s']:>6.2f} {rb['latency_s']:>6.2f}{flag}")

    if pf_divergence or semantic_drift or args.show_text:
        print("\n--- Diverging cases ---")
        for c in cases:
            ra = by_case_a.get(c)
            rb = by_case_b.get(c)
            if ra is None or rb is None:
                continue
            diverges = (ra["passed"] != rb["passed"]
                        or _kw_overlap(ra["response"], rb["response"]) < 0.4)
            if not (diverges or args.show_text):
                continue
            print(f"\n[{c}]  query: {ra['query']!r}")
            print(f"  A ({a['backend']}, "
                  f"{'PASS' if ra['passed'] else 'FAIL'}): "
                  f"{ra['response'].strip()}")
            if ra["issues"]:
                print(f"    issues: {ra['issues']}")
            print(f"  B ({b['backend']}, "
                  f"{'PASS' if rb['passed'] else 'FAIL'}): "
                  f"{rb['response'].strip()}")
            if rb["issues"]:
                print(f"    issues: {rb['issues']}")

    print(f"\n=== Summary: {pf_divergence} pass/fail divergence(s), "
          f"{len(semantic_drift)} low-overlap case(s) ===")
    sys.exit(1 if pf_divergence else 0)


if __name__ == "__main__":
    main()