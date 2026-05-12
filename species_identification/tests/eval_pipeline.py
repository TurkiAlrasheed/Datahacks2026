"""
Eval harness for the RoboRanger pipeline.

Runs a list of (species_id, query, expected_path, expected_intent) cases
through pipeline.answer() and reports:
  - per-case pass/fail on expected path
  - intent mismatches (soft signal, not a hard fail)
  - latency by stage and by path
  - chunks_dropped distribution (signal that threshold may be too strict)
  - failures listed at the bottom with full context

Run from the same directory you run run_pipeline.py from:

    python eval_pipeline.py --cases eval_cases.yaml --db corpus.db
    python eval_pipeline.py --cases eval_cases.yaml --db corpus.db --verbose
    python eval_pipeline.py --cases eval_cases.yaml --db corpus.db --filter direct

The --filter flag matches a substring against the case `tag` so you can
iterate on one bucket at a time (e.g. tag=direct, tag=refusal).

Cases file format (YAML):

    cases:
      - tag: direct_diet
        species_id: Buteo_lineatus
        query: what does it eat
        expected_path: blurb_direct
        expected_intent: DIET     # optional; mismatch is a soft warning

      - tag: refusal_offtopic
        species_id: Buteo_lineatus
        query: what's the weather today
        expected_path: gate_rejected
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(1, "../species_identification/pipeline")
sys.path.insert(2, "../species_identification/llm-tuning")
import yaml

from pipeline import RoboRangerPipeline, Response
from intent import Intent


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

@dataclass
class Case:
    tag: str
    species_id: str
    query: str
    expected_path: str
    expected_intent: Intent | None = None


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("cases", [])
    if not items:
        sys.exit(f"[FATAL] {path} has no cases.")

    cases: list[Case] = []
    for i, item in enumerate(items):
        try:
            intent_name = item.get("expected_intent")
            expected_intent = Intent[intent_name] if intent_name else None
            cases.append(Case(
                tag=item["tag"],
                species_id=item["species_id"],
                query=item["query"],
                expected_path=item["expected_path"],
                expected_intent=expected_intent,
            ))
        except (KeyError, ValueError) as e:
            sys.exit(f"[FATAL] case #{i} malformed ({path}): {e}")
    return cases


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: Case
    response: Response
    path_ok: bool
    intent_ok: bool | None   # None when no expected_intent set

    @property
    def passed(self) -> bool:
        return self.path_ok


@dataclass
class Summary:
    results: list[CaseResult] = field(default_factory=list)

    def add(self, r: CaseResult) -> None:
        self.results.append(r)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def intent_mismatches(self) -> list[CaseResult]:
        return [r for r in self.results if r.intent_ok is False]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def latencies_by_path(self) -> dict[str, list[float]]:
        by_path: dict[str, list[float]] = {}
        for r in self.results:
            t = r.response.latency.get("total")
            if t is not None:
                by_path.setdefault(r.response.path, []).append(t)
        return by_path

    def chunks_dropped(self) -> list[int]:
        return [r.response.chunks_dropped for r in self.results
                if r.response.chunks_dropped]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(pipeline: RoboRangerPipeline, case: Case) -> CaseResult:
    response = pipeline.answer(case.species_id, case.query)
    path_ok = response.path == case.expected_path
    if case.expected_intent is None:
        intent_ok = None
    else:
        got = response.intent.intent if response.intent else None
        intent_ok = (got == case.expected_intent)
    return CaseResult(case=case, response=response,
                      path_ok=path_ok, intent_ok=intent_ok)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_latency(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = min(int(len(xs_sorted) * p), len(xs_sorted) - 1)
    return xs_sorted[idx]


def print_progress_line(r: CaseResult) -> None:
    mark = "PASS" if r.passed else "FAIL"
    score_note = ""
    if r.response.intent is not None:
        score_note = f" intent_score={r.response.intent.score:.3f}"
    intent_note = ""
    if r.intent_ok is False:
        got = r.response.intent.intent.name if r.response.intent else "None"
        intent_note = f" [intent: expected {r.case.expected_intent.name}, got {got}]"
    print(f"  [{mark}] {r.case.tag:30s} "
          f"path={r.response.path:20s} "
          f"({_fmt_latency(r.response.latency.get('total', 0))})"
          f"{score_note}{intent_note}")


def print_summary(summary: Summary, verbose: bool) -> None:
    print()
    print("=" * 78)
    print(f"Summary: {summary.passed}/{summary.total} passed "
          f"({summary.passed / summary.total * 100:.0f}%)")

    # Intent mismatches — soft warnings, surface them so they don't hide.
    mismatches = summary.intent_mismatches
    if mismatches:
        print(f"\n{len(mismatches)} intent mismatch(es) "
              f"(path correct, intent wrong):")
        for r in mismatches:
            got = r.response.intent.intent.name if r.response.intent else "None"
            print(f"  - {r.case.tag}: expected {r.case.expected_intent.name}, "
                  f"got {got}")

    # Latency by path
    by_path = summary.latencies_by_path()
    if by_path:
        print("\nLatency by path (median / p95):")
        for path, lats in sorted(by_path.items()):
            print(f"  {path:22s} n={len(lats):3d}  "
                  f"median={_fmt_latency(statistics.median(lats))}  "
                  f"p95={_fmt_latency(_percentile(lats, 0.95))}")

    # Threshold signal
    drops = summary.chunks_dropped()
    if drops:
        print(f"\nchunks_dropped (above-threshold filter): "
              f"{sum(drops)} total across {len(drops)} queries, "
              f"max {max(drops)} in one query")
        if max(drops) >= 4:
            print("  Note: high drop counts may indicate threshold is too strict.")

    # Failures last — most important, easiest to find at the bottom.
    if summary.failures:
        print(f"\n{len(summary.failures)} failure(s):")
        for r in summary.failures:
            print(f"\n  [FAIL] {r.case.tag}")
            print(f"    species:  {r.case.species_id}")
            print(f"    query:    {r.case.query!r}")
            print(f"    expected: {r.case.expected_path}")
            print(f"    got:      {r.response.path}")
            if r.response.intent:
                print(f"    intent:   {r.response.intent.intent.name} "
                      f"(confidence={r.response.intent.confidence})")
            if r.response.error:
                print(f"    error:    {r.response.error}")
            if verbose:
                print(f"    answer:   {r.response.text!r}")

    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pipeline_for_eval(
    *,
    db_path: str,
    backend: str,
    model: str,
    backend_host: str | None,
    retrieval_threshold: float | None,
) -> RoboRangerPipeline:
    """
    Construct the pipeline using the shared factory so the eval, CLI, and
    Uno Q runtime stay in lockstep. If the factory's interface changes,
    only this thin wrapper needs to follow.
    """
    from pipeline_factory import build_pipeline as _build

    kwargs: dict = dict(
        db_path=db_path,
        backend=backend,
        model=model,
        backend_host=backend_host,
    )
    if retrieval_threshold is not None:
        kwargs["retrieval_threshold"] = retrieval_threshold
    return _build(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="path to eval_cases.yaml")
    ap.add_argument("--db", required=True, help="path to corpus.db")
    ap.add_argument("--backend", choices=("ollama", "llama-cpp"),
                    default="ollama",
                    help="LLM backend (default: ollama for laptop)")
    ap.add_argument("--model", default="smollm2:360m",
                    help="model name as backend understands it")
    ap.add_argument("--backend-host", default=None,
                    help="override backend host URL")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine retrieval threshold "
                         "(default: from pipeline.py)")
    ap.add_argument("--filter", help="run only cases whose tag contains this substring")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="show answer text on failures")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.filter:
        cases = [c for c in cases if args.filter in c.tag]
        if not cases:
            sys.exit(f"No cases matched filter '{args.filter}'.")

    print(f"Building pipeline (backend={args.backend}, model={args.model}) "
          f"against {args.db}...")
    t0_build = time.perf_counter()
    pipeline = build_pipeline_for_eval(
        db_path=args.db,
        backend=args.backend,
        model=args.model,
        backend_host=args.backend_host,
        retrieval_threshold=args.threshold,
    )
    print(f"  built in {time.perf_counter() - t0_build:.1f}s")

    print(f"Running {len(cases)} case(s)...\n")
    summary = Summary()
    t0 = time.perf_counter()
    for case in cases:
        result = run_case(pipeline, case)
        summary.add(result)
        print_progress_line(result)
    total_elapsed = time.perf_counter() - t0

    print_summary(summary, verbose=args.verbose)
    print(f"Total wall time: {_fmt_latency(total_elapsed)}")

    # Exit nonzero on any failure — handy for CI / pre-commit.
    return 0 if summary.passed == summary.total else 1


if __name__ == "__main__":
    sys.exit(main())