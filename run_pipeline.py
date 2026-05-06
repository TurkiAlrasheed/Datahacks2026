"""
RoboRanger CLI driver.

Single entry point for manually testing the end-to-end pipeline on either
the laptop (Ollama backend) or the Uno Q (llama.cpp backend). Three modes:

  one-shot:  --species X --query "..."        -> answer + latency table
  repl:      --species X                      -> interactive prompt loop
  profile:   --profile queries.txt --species X -> p50/p95 per stage

Examples:
  # Laptop iteration
  python run_pipeline.py --species Marah_macrocarpa \\
      --query "is it venomous?"

  # Uno Q deployment
  python run_pipeline.py --backend llama-cpp --db /opt/roboranger/corpus.db \\
      --species Marah_macrocarpa --repl

  # Latency profile (50 queries from a file, p50/p95 per stage)
  python run_pipeline.py --backend llama-cpp --species Marah_macrocarpa \\
      --profile test_queries.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(1, "species_identification/pipeline")
sys.path.insert(2, "species_identification/llm-tuning")
sys.path.insert(3, "species_identification/tests")
from pipeline_factory import build_pipeline
from pipeline import RoboRangerPipeline, Response


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

# Stage order for the latency table — matches the order they run in
# pipeline.answer(). Listed explicitly so the table is stable regardless
# of dict insertion order, and so we can print "—" for skipped stages.
STAGE_ORDER = ("gate", "blurb", "intent", "retrieval", "prompt", "llm", "total")


def fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "    —"
    return f"{seconds * 1000:6.1f}ms"


def print_response(resp: Response, *, verbose: bool = False) -> None:
    """Print the answer + a compact latency/diagnostic table."""
    print()
    print("─" * 60)
    print(resp.text)
    print("─" * 60)
    print(f"  path:    {resp.path}")
    if resp.intent is not None:
        print(f"  intent:  {resp.intent.intent.name} ({resp.intent.confidence})")
    if resp.chunks_used or resp.chunks_dropped:
        print(f"  chunks:  {len(resp.chunks_used)} used, "
              f"{resp.chunks_dropped} dropped (below threshold)")
    if resp.error:
        print(f"  error:   {resp.error}")

    print("  latency:")
    for stage in STAGE_ORDER:
        marker = "  └─" if stage == "total" else "  │ "
        print(f"  {marker} {stage:10s} {fmt_ms(resp.latency.get(stage))}")

    if verbose and resp.chunks_used:
        print("  retrieved chunks:")
        for i, c in enumerate(resp.chunks_used, 1):
            preview = c.text[:80].replace("\n", " ")
            print(f"    {i}. [{c.source}] cos={c.score:.3f}  {preview}…")
    print()


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warmup(pipeline: RoboRangerPipeline, species_id: str) -> None:
    """
    Send a throwaway query through the full pipeline so the first real
    query isn't artificially slow. Warms:
      - sentence-transformer (first .encode() is slower than steady state)
      - sqlite-vec (first vector query)
      - llama.cpp prefix cache (system + blurb prefix gets cached for
        subsequent calls with the same species)
    """
    print("warming up...", end=" ", flush=True)
    t = time.perf_counter()
    pipeline.answer(species_id, "warmup query, ignore")
    print(f"({(time.perf_counter() - t) * 1000:.0f}ms)")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_one_shot(pipeline: RoboRangerPipeline, species_id: str,
                 query: str, *, verbose: bool) -> int:
    resp = pipeline.answer(species_id, query)
    print_response(resp, verbose=verbose)
    return 0 if resp.path != "llm_error" else 1


def run_repl(pipeline: RoboRangerPipeline, species_id: str,
             *, verbose: bool) -> int:
    """Interactive loop. Ctrl-C or empty line to exit."""
    print(f"REPL mode — species: {species_id}")
    print("Empty line or Ctrl-C to exit. Prefix with '/species <id>' to switch.")
    print()
    current_species = species_id
    try:
        while True:
            try:
                query = input(f"[{current_species}] > ").strip()
            except EOFError:
                print()
                return 0
            if not query:
                return 0
            if query.startswith("/species "):
                current_species = query.split(maxsplit=1)[1].strip()
                print(f"  switched to {current_species}")
                continue
            resp = pipeline.answer(current_species, query)
            print_response(resp, verbose=verbose)
    except KeyboardInterrupt:
        print()
        return 0


def run_profile(pipeline: RoboRangerPipeline, species_id: str,
                queries_path: Path) -> int:
    """Run each line of `queries_path` and print percentile timings per stage."""
    queries = [
        line.strip() for line in queries_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not queries:
        print(f"no queries found in {queries_path}", file=sys.stderr)
        return 1

    print(f"profiling {len(queries)} queries against {species_id}...")
    # stage -> list of seconds. path counts tracked separately.
    timings: dict[str, list[float]] = {s: [] for s in STAGE_ORDER}
    paths: dict[str, int] = {}
    for i, q in enumerate(queries, 1):
        resp = pipeline.answer(species_id, q)
        paths[resp.path] = paths.get(resp.path, 0) + 1
        for stage, t in resp.latency.items():
            timings.setdefault(stage, []).append(t)
        # one-line progress so a stalled run is obvious
        print(f"  [{i:3d}/{len(queries)}] {resp.path:20s} "
              f"total={fmt_ms(resp.latency.get('total'))}  {q[:50]}")

    print()
    print("─" * 60)
    print(f"  paths: " + ", ".join(f"{p}={n}" for p, n in sorted(paths.items())))
    print()
    print(f"  {'stage':10s} {'p50':>10s} {'p95':>10s} {'max':>10s}  n")
    for stage in STAGE_ORDER:
        ts = timings.get(stage) or []
        if not ts:
            continue
        # statistics.quantiles needs n>=2; fall back to max for tiny runs.
        if len(ts) >= 2:
            qs = statistics.quantiles(ts, n=20)  # 5%-step quantiles
            p50, p95 = statistics.median(ts), qs[18]  # 95th percentile
        else:
            p50 = p95 = ts[0]
        print(f"  {stage:10s} {fmt_ms(p50)} {fmt_ms(p95)} "
              f"{fmt_ms(max(ts))}  {len(ts)}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", 
                   default="species_identification/offline-info/corpus.db",
                   help="path to corpus.db (default: corpus.db)")
    p.add_argument("--backend", choices=("ollama", "llama-cpp"),
                   default="ollama",
                   help="LLM backend (default: ollama for laptop)")
    p.add_argument("--model", default="smollm2:360m",
                   help="model name as backend understands it")
    p.add_argument("--backend-host", default=None,
                   help="override backend host URL")
    p.add_argument("--threshold", type=float, default=None,
                   help="cosine retrieval threshold (default: from pipeline.py)")
    p.add_argument("--species", required=True,
                   help="species_id (e.g. Crotalus_oreganus_helleri)")
    p.add_argument("--query", default=None,
                   help="one-shot query (omit for --repl or --profile)")
    p.add_argument("--repl", action="store_true",
                   help="interactive mode")
    p.add_argument("--profile", type=Path, default=None,
                   help="path to a file of newline-separated test queries")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip warm-up query")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show retrieved chunks")
    args = p.parse_args()

    # Validate mode selection — exactly one of query/repl/profile.
    modes = sum(bool(x) for x in (args.query, args.repl, args.profile))
    if modes != 1:
        p.error("exactly one of --query, --repl, --profile is required")

    # Build the pipeline. This is the only place that touches the heavy
    # imports (sentence-transformers, sqlite-vec).
    print(f"building pipeline (backend={args.backend}, model={args.model})...",
          flush=True)
    t = time.perf_counter()
    kwargs = dict(
        db_path=args.db,
        backend=args.backend,
        model=args.model,
        backend_host=args.backend_host,
    )
    if args.threshold is not None:
        kwargs["retrieval_threshold"] = args.threshold
    pipeline = build_pipeline(**kwargs)
    print(f"  built in {time.perf_counter() - t:.1f}s")

    if not args.no_warmup:
        warmup(pipeline, args.species)

    if args.query:
        return run_one_shot(pipeline, args.species, args.query,
                            verbose=args.verbose)
    if args.repl:
        return run_repl(pipeline, args.species, verbose=args.verbose)
    if args.profile:
        return run_profile(pipeline, args.species, args.profile)
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())