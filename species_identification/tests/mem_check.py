"""
Memory check helper for RoboRanger phase-6 testing.

Two functions:
  - print_total_rss(): one-shot print of voice_loop + ollama RSS combined
  - growth_check(): run N pipeline queries, print RSS before/after to
    catch leaks

Designed to drop into voice_loop.py without other changes. Uses psutil
because /proc parsing is Linux-only and `ps` parsing across Win/Mac/Linux
is more annoying than just adding the dep.

Usage in voice_loop.py:

    from mem_check import print_total_rss, growth_check

    # after warmup, before the REPL loop:
    print_total_rss(label="after warmup")

    # optional leak check — runs 12 queries through pipeline.answer:
    growth_check(pipeline, args.species, n=12)
"""

from __future__ import annotations

import os
import time

try:
    import psutil
except ImportError:
    raise SystemExit(
        "psutil not installed — pip install psutil"
    )


def _self_rss_mb() -> float:
    """Resident set size of the current process, in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _ollama_rss_mb() -> tuple[float, int]:
    """
    Total RSS of every ollama process currently running, in MB. Ollama
    typically runs at least two: a server (small) and a per-model runner
    (big — this is where SmolLM2 actually lives). Sum them.

    Returns (total_mb, process_count). Returns (0.0, 0) if no ollama
    process is running, e.g. when using the llama-cpp backend on the
    Uno Q where the LLM is in-process.
    """
    total_kb = 0
    count = 0
    for proc in psutil.process_iter(attrs=("name", "memory_info")):
        try:
            name = (proc.info["name"] or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # On Windows the name is "ollama.exe"; on Linux/macOS it's
        # "ollama" or "ollama-runner". `in` covers all of them.
        if "ollama" not in name:
            continue
        try:
            total_kb += proc.info["memory_info"].rss / 1024
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total_kb / 1024, count


def print_total_rss(label: str = "") -> None:
    """
    Print self RSS, ollama RSS, and combined total. Tag with `label`
    so multiple readouts in the same run are distinguishable.
    """
    self_mb = _self_rss_mb()
    ollama_mb, n_ollama = _ollama_rss_mb()
    tag = f" [{label}]" if label else ""
    print(f"  memory{tag}:")
    print(f"    self (voice_loop):     {self_mb:7.1f} MB")
    if n_ollama:
        print(f"    ollama ({n_ollama} procs):       {ollama_mb:7.1f} MB")
        print(f"    combined total:        {self_mb + ollama_mb:7.1f} MB")
    else:
        # No ollama running — either using llama-cpp backend, or ollama
        # hasn't started its worker yet because no LLM-route query has
        # been issued. Worth flagging because a "self only" reading
        # under-predicts the device footprint.
        print(f"    ollama:                not running")
        print(f"    (run an LLM-route query first to load the model)")


def growth_check(pipeline, species_id: str, n: int = 12) -> None:
    """
    Run `n` representative queries through pipeline.answer() and print
    RSS before/after. If RSS grows by more than ~20-30 MB, you probably
    have a leak — most likely a sqlite cursor not closing or an
    embedding cache without an eviction policy. Easier to find on the
    laptop than on the Uno Q.

    Mixes direct-route and LLM-route queries so both code paths are
    exercised. A leak that only shows up on the LLM path won't be
    visible if you only test direct queries.
    """
    queries = [
        # Direct-route — exercises gate, intent, blurb fetch, formatter.
        "is it dangerous",
        "what does it eat",
        "how big is it",
        "where does it live",
        "is it nocturnal",
        "what does it look like",
        # LLM-route — exercises retrieval, prompt build, llm.generate.
        "tell me about its nesting habits",
        "tell me about its courtship",
        # Mix in a gate rejection — different code path, different
        # allocations.
        "what's the weather like today",
    ]

    print(f"\n  growth check: {n} queries...")
    print_total_rss(label="before")

    t0 = time.perf_counter()
    for i in range(n):
        q = queries[i % len(queries)]
        pipeline.answer(species_id, q)
    elapsed = time.perf_counter() - t0
    print(f"  {n} queries in {elapsed:.1f}s ({elapsed / n:.2f}s avg)")

    print_total_rss(label="after")
    print()