"""
Interactive pipeline REPL — same shape as test_corpus.py but runs the
whole orchestrator (gate -> blurb -> intent -> retrieval -> LLM).

Usage:
    # laptop / Ollama
    python pipeline_repl.py corpus.db --backend ollama --model smollm2-q8

    # Uno Q / llama.cpp
    python pipeline_repl.py corpus.db --backend llama-cpp \\
        --host http://uno-q.local:8080

Then type queries like:
    > Crotalus_oreganus_helleri | is this snake dangerous
    > Apis_mellifera | what does it eat
    > Procyon_lotor | where is the bathroom

Special commands:
    list                  - show species in the corpus
    threshold 0.6         - set the retrieval cosine threshold live
    verbose on / off      - toggle full diagnostic output
    quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_factory import build_pipeline


def _print_response(resp, verbose: bool) -> None:
    """Format a Response for the terminal."""
    print()
    print(f"  [{resp.path}]  {resp.text}")
    print()
    if not verbose:
        return

    # Detailed diagnostic block
    print("  --- diagnostic ---")
    if resp.gate is not None:
        g = resp.gate
        verdict = "ACCEPT" if g.in_domain else "REJECT"
        print(f"  gate:       {verdict} ({g.confidence})  "
              f"wild={g.wildlife_score:+.3f}  "
              f"off={g.off_topic_score:+.3f}  "
              f"margin={g.margin:+.3f}")
    if resp.intent is not None:
        i = resp.intent
        print(f"  intent:     {i.intent.value} ({i.confidence})  "
              f"score={i.score:+.3f}  margin={i.margin:+.3f}")
    if resp.blurb is not None:
        print(f"  blurb:      {resp.blurb.common_name}")
    if resp.chunks_used or resp.chunks_dropped:
        print(f"  chunks:     {len(resp.chunks_used)} kept, "
              f"{resp.chunks_dropped} dropped")
        for i, c in enumerate(resp.chunks_used, 1):
            preview = c.text if len(c.text) < 120 else c.text[:120] + "..."
            print(f"    [{i}] cos={c.score:.3f}  cat={c.source!r}")
            print(f"        {preview}")
    if resp.error:
        print(f"  error:      {resp.error}")
    if resp.latency:
        parts = [f"{k}={v*1000:.0f}ms" for k, v in resp.latency.items()]
        print(f"  latency:    {'  '.join(parts)}")
    print()


def _list_species(conn) -> None:
    rows = conn.execute(
        "SELECT species_id, species_name, common_name, "
        "       (SELECT COUNT(*) FROM chunks "
        "         WHERE chunks.species_id = species.species_id) "
        "FROM species ORDER BY species_id"
    ).fetchall()
    print(f"\n{len(rows)} species in corpus:")
    for sid, sname, cname, n in rows:
        label = sname + (f" ({cname})"
                         if cname and cname.lower() != sname.lower() else "")
        print(f"  {sid:35s} {n:3d} chunks  {label}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="path to corpus.db")
    ap.add_argument("--backend", choices=["ollama", "llama-cpp"],
                    default="ollama")
    ap.add_argument("--model", default="smollm2:360m")
    ap.add_argument("--host", default=None)
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="initial retrieval cosine threshold")
    ap.add_argument("--verbose", action="store_true",
                    help="show full diagnostic output by default")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"Corpus not found: {db_path}")

    print(f"Building pipeline...")
    print(f"  embedder:  loading bge-small-en-v1.5...")
    pipeline = build_pipeline(
        db_path=db_path,
        backend=args.backend,
        model=args.model,
        backend_host=args.host,
        retrieval_threshold=args.threshold,
    )
    print(f"  backend:   {pipeline.llm.name} ({pipeline.llm.model})")
    print(f"  threshold: {pipeline.retrieval_threshold:.3f}")
    print()
    print("Type 'list' for species, 'threshold N' to retune, "
          "'verbose on/off' to toggle, 'quit' to exit.")
    print("Format:  species_id | your question\n")

    verbose = args.verbose

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "list":
            _list_species(pipeline.db_conn)
            continue
        if line.startswith("threshold "):
            try:
                new_t = float(line.split()[1])
                pipeline.retrieval_threshold = new_t
                print(f"  retrieval threshold = {new_t:.3f}\n")
            except (IndexError, ValueError):
                print("  usage: threshold 0.55\n")
            continue
        if line.startswith("verbose"):
            verbose = "on" in line.lower()
            print(f"  verbose = {verbose}\n")
            continue
        if "|" not in line:
            print("  format: species_id | question  (or 'list' / 'quit')\n")
            continue

        species_id, _, query = line.partition("|")
        species_id = species_id.strip()
        query = query.strip()
        if not species_id or not query:
            print("  both species_id and question are required\n")
            continue

        resp = pipeline.answer(species_id=species_id, query=query)
        _print_response(resp, verbose=verbose)


if __name__ == "__main__":
    main()