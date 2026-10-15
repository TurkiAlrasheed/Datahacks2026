#!/usr/bin/env python3
"""
retrieval_recall_check.py — measure how much recall species retrieval loses,
before and after the partition-key fix.

The ORIGINAL production query (schema v1) looked like this:

    WHERE v.embedding MATCH ?      <- KNN over EVERY chunk in the corpus
      AND c.species_id = ?         <- species filter applied AFTER the join
      AND k = 50                   <- global candidate pool
    ORDER BY v.distance
    LIMIT ?                        <- pipeline asks for RETRIEVAL_K = 5

chunk_vectors had no species_id column, so the vector search could not scope
itself. It returned the 50 globally-nearest chunks, and only then did the
join drop everything belonging to other species. If fewer than 5 of those 50
belonged to the species you asked about, you silently got a thinner prompt —
no error, no warning, just a worse answer. It could also hand back the
species' own blurb chunk, which the prompt already contains as SPECIES FACTS.

Schema v2 (corpus_schema.py) makes species_id a vec0 PARTITION KEY and
category a metadata column, so production retrieval (test_corpus.retrieve /
knn_for_species) is:

    WHERE embedding MATCH ? AND k = ? AND species_id = ? AND category != 'blurb'

Modes:
    legacy       the v1 global-pool + post-filter query (works on v1 and v2)
    partitioned  the production v2 query (v2 only)
    both         legacy and partitioned on the SAME vectors — a direct
                 before/after (default on a v2 corpus)

Ground truth for both modes is an exact brute-force scan
(vec_distance_l2 over the species' non-blurb chunks), independent of vec0's
KNN implementation.

Run from the repo root:

    python species_identification/tests/retrieval_recall_check.py \
        --db species_identification/offline-info/corpus.db

Options:
    --mode M        legacy | partitioned | both (default: both on v2, legacy on v1)
    --k-pool N      legacy candidate pool (default 50, matching the old query)
    --want N        chunks the pipeline asks for (default 5, = RETRIEVAL_K)
    --sweep         legacy only: sweep k to find the pool needed for full recall
    --no-rewrite    also measure without the common-name grounding, to see
                    how much rewrite_query was masking the problem
    --csv PATH      write per-(mode, species, probe) rows for further analysis
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Import the PRODUCTION retrieval helpers rather than reimplementing them,
# so this measures the real code path. pipeline.py imports these same
# functions from test_corpus.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parent))
sys.path.insert(2, str(_HERE.parent / "llm-tuning"))

from corpus_schema import (  # noqa: E402
    BLURB_CATEGORY,
    EMBED_DIM,
    MIGRATION_HINT,
    read_meta,
    schema_version,
)
from prompt_builder import MAX_CHUNKS  # noqa: E402
from test_corpus import (  # noqa: E402
    EMBED_MODEL,
    _get_common_name,
    embed_query,
    knn_for_species,
    l2_to_cosine,
    open_db,
    rewrite_query,
)

# Cosine floor from pipeline.DEFAULT_RETRIEVAL_THRESHOLD.
COSINE_FLOOR = 0.55

# Stock visitor questions. Deliberately generic — these are the ones most
# likely to match another species' chunks just as well as their own, which
# is exactly the case the post-filter mishandles.
PROBES = [
    "what does it eat",
    "is it dangerous",
    "where does it live",
    "how big does it get",
    "what does it look like",
    "how does it behave",
    "is it endangered",
    "tell me about it",
]


# ---------------------------------------------------------------------------
# Corpus inspection
# ---------------------------------------------------------------------------

def corpus_stats(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    per_species = dict(conn.execute(
        "SELECT species_id, COUNT(*) FROM chunks GROUP BY species_id"
    ).fetchall())
    retrievable = dict(conn.execute(
        "SELECT species_id, COUNT(*) FROM chunks WHERE category != ? "
        "GROUP BY species_id", (BLURB_CATEGORY,)
    ).fetchall())
    counts = list(per_species.values())
    return {
        "total_chunks": total,
        "n_species": len(per_species),
        "min_chunks": min(counts) if counts else 0,
        "max_chunks": max(counts) if counts else 0,
        "mean_chunks": (sum(counts) / len(counts)) if counts else 0.0,
        "per_species": per_species,
        "retrievable": retrievable,
    }


def list_species_ids(conn) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT species_id FROM chunks ORDER BY species_id"
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Retrieval variants — each returns [(chunk_id, category, l2_distance)]
# ---------------------------------------------------------------------------

def legacy_retrieve(conn, q_bytes: bytes, species_id: str, k_pool: int,
                    want: int) -> list[tuple]:
    """The v1 production query: global top-k_pool, then species post-filter."""
    return conn.execute(
        """
        SELECT c.id, c.category, v.distance
        FROM chunk_vectors v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ?
          AND c.species_id = ?
          AND k = ?
        ORDER BY v.distance
        LIMIT ?
        """,
        (q_bytes, species_id, k_pool, want),
    ).fetchall()


def partitioned_retrieve(conn, q_bytes: bytes, species_id: str,
                         want: int) -> list[tuple]:
    """The v2 production query (test_corpus.knn_for_species)."""
    return [(cid, cat, dist) for cid, cat, _, dist
            in knn_for_species(conn, q_bytes, species_id, want)]


def exact_truth(conn, q_bytes: bytes, species_id: str, want: int) -> list[tuple]:
    """
    What retrieval SHOULD return: the `want` nearest non-blurb chunks of this
    species by an exact scan. Uses vec_distance_l2 rather than a vec0 KNN so
    the ground truth doesn't depend on the index being measured.
    """
    return conn.execute(
        """
        SELECT c.id, c.category, vec_distance_l2(v.embedding, ?) AS d
        FROM chunks c
        JOIN chunk_vectors v ON v.rowid = c.id
        WHERE c.species_id = ?
          AND c.category != ?
        ORDER BY d
        LIMIT ?
        """,
        (q_bytes, species_id, BLURB_CATEGORY, want),
    ).fetchall()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

# Query text -> embedding bytes. Embeddings don't depend on the DB or the
# retrieval mode, so every mode/sweep step reuses them.
_EMBED_CACHE: dict[str, bytes] = {}


def measure(conn, embedder, species_ids, probes, mode, k_pool, want,
            retrievable, use_rewrite=True):
    """
    For each (species, probe): what the given retrieval mode returned versus
    the exact top-`wanted` non-blurb chunks, where
    wanted = min(want, retrievable chunks for that species) — a species with
    only 3 chunks can't be "short" of 5.
    """
    rows = []
    for sid in species_ids:
        common = _get_common_name(conn, sid)
        wanted = min(want, retrievable.get(sid, 0))
        for probe in probes:
            query = rewrite_query(probe, common) if use_rewrite else probe
            if query not in _EMBED_CACHE:
                _EMBED_CACHE[query] = embed_query(embedder, query)
            q_bytes = _EMBED_CACHE[query]

            t = time.perf_counter()
            if mode == "legacy":
                got = legacy_retrieve(conn, q_bytes, sid, k_pool, want)
            else:
                got = partitioned_retrieve(conn, q_bytes, sid, want)
            latency_ms = (time.perf_counter() - t) * 1000

            nonblurb = [r for r in got if r[1] != BLURB_CATEGORY]
            # The prompt builder keeps only the first MAX_CHUNKS snippets.
            in_prompt = got[:MAX_CHUNKS]
            truth_ids = {r[0] for r in exact_truth(conn, q_bytes, sid, wanted)}
            got_ids = {r[0] for r in nonblurb[:wanted]}

            rows.append({
                "mode": mode,
                "rewrite": use_rewrite,
                "species_id": sid,
                "probe": probe,
                "wanted": wanted,
                "returned": len(got),
                "returned_nonblurb": len(nonblurb),
                "shortfall": max(0, wanted - len(nonblurb)),
                "blurb_in_prompt": any(r[1] == BLURB_CATEGORY
                                       for r in in_prompt),
                "missed_vs_truth": len(truth_ids - got_ids),
                "above_cosine_floor": sum(
                    1 for r in nonblurb[:wanted]
                    if l2_to_cosine(r[2]) >= COSINE_FLOOR),
                "best_cosine": (round(l2_to_cosine(nonblurb[0][2]), 4)
                                if nonblurb else None),
                "latency_ms": round(latency_ms, 4),
            })
    return rows


def sweep_k(conn, embedder, species_ids, probes, want, retrievable, ladder):
    """Legacy only: the candidate pool needed before every query is whole."""
    out = []
    for k in ladder:
        rows = measure(conn, embedder, species_ids, probes, "legacy", k,
                       want, retrievable)
        out.append({
            "k": k,
            "queries": len(rows),
            "short_queries": sum(1 for r in rows if r["shortfall"] > 0),
            "missed_queries": sum(1 for r in rows if r["missed_vs_truth"] > 0),
            "total_missing": sum(r["missed_vs_truth"] for r in rows),
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def report(rows, stats, label):
    n = len(rows)
    short = [r for r in rows if r["shortfall"] > 0]
    zero = [r for r in rows if r["wanted"] > 0 and r["returned_nonblurb"] == 0]
    missed = [r for r in rows if r["missed_vs_truth"] > 0]
    blurb = [r for r in rows if r["blurb_in_prompt"]]
    lat = sorted(r["latency_ms"] for r in rows)

    head = f"RESULTS — {label}"
    print(f"\n{head}")
    print("=" * len(head))
    print(f"queries run                 {n}")
    print(f"short of wanted chunks      {len(short):>5}  "
          f"({_pct(len(short), n):.1f}%)")
    print(f"returned ZERO chunks        {len(zero):>5}  "
          f"({_pct(len(zero), n):.1f}%)")
    print(f"missed a better chunk       {len(missed):>5}  "
          f"({_pct(len(missed), n):.1f}%)")
    print(f"blurb chunk in prompt       {len(blurb):>5}  "
          f"({_pct(len(blurb), n):.1f}%)   <- duplicates SPECIES FACTS")
    print(f"total chunks lost           "
          f"{sum(r['missed_vs_truth'] for r in rows):>5}")
    if lat:
        print(f"retrieval latency           p50 {statistics.median(lat):.3f} ms"
              f"   p95 {lat[max(0, int(0.95 * len(lat)) - 1)]:.3f} ms")

    if missed:
        print("\nWorst offenders (species losing the most chunks):")
        by_species = defaultdict(int)
        for r in rows:
            by_species[r["species_id"]] += r["missed_vs_truth"]
        worst = sorted(by_species.items(), key=lambda x: -x[1])[:10]
        for sid, lost in worst:
            if lost == 0:
                continue
            have = stats["per_species"].get(sid, 0)
            print(f"  {lost:>3} chunks lost   {sid:<40} "
                  f"({have} chunks in corpus)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="species_identification/offline-info/corpus.db")
    ap.add_argument("--mode", choices=("legacy", "partitioned", "both"),
                    default=None)
    ap.add_argument("--k-pool", type=int, default=50,
                    help="legacy candidate pool, matching the old hardcoded k")
    ap.add_argument("--want", type=int, default=5,
                    help="chunks requested, = pipeline.RETRIEVAL_K")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="also measure without common-name grounding")
    ap.add_argument("--limit-species", type=int, default=0,
                    help="only test the first N species (faster smoke run)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"corpus not found: {db_path}", file=sys.stderr)
        return 1

    conn = open_db(db_path)
    version = schema_version(conn)
    mode = args.mode or ("both" if version >= 2 else "legacy")
    if mode in ("partitioned", "both") and version < 2:
        print(f"--mode {mode} needs a schema v2 corpus; {db_path} is "
              f"v{version}.\n{MIGRATION_HINT}", file=sys.stderr)
        return 1
    modes = ["legacy", "partitioned"] if mode == "both" else [mode]

    from sentence_transformers import SentenceTransformer

    stats = corpus_stats(conn)
    species_ids = list_species_ids(conn)
    if args.limit_species:
        species_ids = species_ids[:args.limit_species]

    meta = read_meta(conn)
    print("CORPUS")
    print("======")
    print(f"file               {db_path} ({db_path.stat().st_size / 1e6:.2f} MB)")
    print(f"schema             v{version}"
          + (f"  corpus_version {meta['corpus_version']}"
             if meta.get("corpus_version") else ""))
    print(f"species            {stats['n_species']}")
    print(f"chunks             {stats['total_chunks']}")
    print(f"chunks/species     min {stats['min_chunks']}, "
          f"mean {stats['mean_chunks']:.1f}, max {stats['max_chunks']}")
    print(f"vector bytes       "
          f"{stats['total_chunks'] * EMBED_DIM * 4 / 1e6:.1f} MB "
          f"(float[{EMBED_DIM}])")

    print(f"\nloading {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    all_rows = []
    variants = [True] + ([False] if args.no_rewrite else [])
    for use_rewrite in variants:
        for m in modes:
            rows = measure(conn, embedder, species_ids, PROBES, m,
                           args.k_pool, args.want, stats["retrievable"],
                           use_rewrite=use_rewrite)
            grounding = ("with rewrite_query grounding" if use_rewrite
                         else "WITHOUT grounding")
            pool = f", k={args.k_pool} global pool" if m == "legacy" else ""
            report(rows, stats, f"{m}{pool}, {grounding}")
            all_rows.extend(rows)

    if args.sweep:
        ladder = [25, 50, 100, 200, 500, max(1000, stats["total_chunks"])]
        print("\nLEGACY K SWEEP (with rewrite)")
        print("=============================")
        print(f"{'k':>8}  {'queries short':>14}  {'missed better':>14}  "
              f"{'chunks lost':>12}")
        for s in sweep_k(conn, embedder, species_ids, PROBES, args.want,
                         stats["retrievable"], ladder):
            print(f"{s['k']:>8}  {s['short_queries']:>14}  "
                  f"{s['missed_queries']:>14}  {s['total_missing']:>12}")
        print("\nThe k at which losses hit 0 is the global pool this corpus "
              "needs today.\nIt grows with species count; the partition key "
              "removes the dependency entirely.")

    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {args.csv}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
