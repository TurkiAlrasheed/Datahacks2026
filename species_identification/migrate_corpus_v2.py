"""
Migrate a v1 corpus.db (global vec0 index) to schema v2 (species_id
partition key + category metadata column + corpus_meta/species_manifest).

No Wikipedia re-scrape and no re-embedding: species rows, chunk ids, chunk
text and the exact vector bytes are copied across. That keeps the migration
fast and offline, and makes before/after retrieval measurements an
apples-to-apples comparison (same vectors, different index layout).

Usage (from the repo root):
    python species_identification/migrate_corpus_v2.py \\
        species_identification/offline-info/corpus.db \\
        species_identification/offline-info/corpus_v2.db

    # Compare DB size + partition-scoped KNN latency across chunk sizes first:
    python species_identification/migrate_corpus_v2.py SRC DST --measure

The output is verified before it is moved into place: same species/chunk
counts, byte-identical vectors per rowid, vector metadata matching the
chunks table, and identical per-species content hashes computed from the
source and from the result.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

import sqlite_vec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_schema import (  # noqa: E402
    VEC_CHUNK_SIZE,
    create_schema,
    insert_vector,
    load_content,
    refresh_manifest,
    schema_version,
    species_hashes,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def build_v2(content: dict, dst: Path, chunk_size: int) -> dict:
    """Write `content` (from load_content on the source) into a fresh v2 DB."""
    if dst.exists():
        dst.unlink()
    conn = _connect(dst)
    try:
        create_schema(conn, chunk_size=chunk_size)
        conn.executemany(
            "INSERT INTO species(species_id, species_name, common_name, "
            "source, blurb_text, blurb_json) VALUES (?, ?, ?, ?, ?, ?)",
            [(r["species_id"], r["species_name"], r["common_name"],
              r["source"], r["blurb_text"], r["blurb_json"])
             for r in content["species"].values()],
        )
        for sid, rows in content["chunks"].items():
            for cid, cat, text in rows:
                conn.execute(
                    "INSERT INTO chunks(id, species_id, category, text) "
                    "VALUES (?, ?, ?, ?)", (cid, sid, cat, text))
                vec = content["vectors"].get(cid)
                if vec is None:
                    raise RuntimeError(f"chunk {cid} ({sid}) has no vector "
                                       f"in the source DB")
                insert_vector(conn, cid, sid, cat, vec)
        conn.commit()
        meta = refresh_manifest(conn)
        conn.execute("VACUUM")
        return meta
    finally:
        conn.close()


def verify(src_content: dict, dst: Path) -> list[str]:
    """Return a list of problems; empty means the migration is faithful."""
    problems = []
    conn = _connect(dst)
    try:
        if schema_version(conn) != 2:
            problems.append("destination is not schema v2")
        dst_content = load_content(conn)

        for key in ("species", "vectors"):
            a, b = len(src_content[key]), len(dst_content[key])
            if a != b:
                problems.append(f"{key} count {a} -> {b}")
        a = sum(len(v) for v in src_content["chunks"].values())
        b = sum(len(v) for v in dst_content["chunks"].values())
        if a != b:
            problems.append(f"chunk count {a} -> {b}")

        diff_vec = [r for r, v in src_content["vectors"].items()
                    if dst_content["vectors"].get(r) != v]
        if diff_vec:
            problems.append(f"{len(diff_vec)} vectors differ "
                            f"(first rowid {diff_vec[0]})")

        # Partition/metadata columns must agree with the chunks table,
        # otherwise species-scoped KNN would silently return wrong rows.
        chunk_meta = {cid: (sid, cat)
                      for sid, rows in dst_content["chunks"].items()
                      for cid, cat, _ in rows}
        mismatched = [
            rowid for rowid, sid, cat in conn.execute(
                "SELECT rowid, species_id, category FROM chunk_vectors")
            if chunk_meta.get(rowid) != (sid, cat)
        ]
        if mismatched:
            problems.append(f"{len(mismatched)} vector rows have "
                            f"species_id/category not matching chunks")

        src_h, dst_h = species_hashes(src_content), species_hashes(dst_content)
        changed = sorted(s for s in src_h if src_h[s] != dst_h.get(s))
        if changed:
            problems.append(f"species content hash changed: {changed[:5]}")
        manifest = dict(conn.execute(
            "SELECT species_id, sha256 FROM species_manifest"))
        if manifest != {s: sha for s, (_, sha) in dst_h.items()}:
            problems.append("species_manifest does not match content")
    finally:
        conn.close()
    return problems


def measure(content: dict, sizes: list[int], n_queries: int = 400) -> None:
    """Build a throwaway v2 DB per chunk_size; report file size and latency."""
    rng = random.Random(0)
    by_species = {sid: [cid for cid, _, _ in rows]
                  for sid, rows in content["chunks"].items()}
    probes = []
    for _ in range(n_queries):
        sid = rng.choice(sorted(by_species))
        probes.append((sid, content["vectors"][rng.choice(by_species[sid])]))

    print(f"{'chunk_size':>10}  {'file MB':>8}  {'knn p50 ms':>10}  "
          f"{'knn p95 ms':>10}")
    with tempfile.TemporaryDirectory() as tmp:
        for cs in sizes:
            path = Path(tmp) / f"corpus_cs{cs}.db"
            build_v2(content, path, cs)
            conn = _connect(path)
            times = []
            for sid, vec in probes:
                t = time.perf_counter()
                conn.execute(
                    "SELECT rowid, distance FROM chunk_vectors "
                    "WHERE embedding MATCH ? AND k = 5 AND species_id = ? "
                    "AND category != 'blurb'", (vec, sid)).fetchall()
                times.append((time.perf_counter() - t) * 1000)
            conn.close()
            times.sort()
            print(f"{cs:>10}  {path.stat().st_size / 1e6:>8.2f}  "
                  f"{statistics.median(times):>10.3f}  "
                  f"{times[int(0.95 * len(times)) - 1]:>10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="existing corpus.db (v1)")
    ap.add_argument("dst", type=Path, help="output path for the v2 corpus")
    ap.add_argument("--chunk-size", type=int, default=VEC_CHUNK_SIZE,
                    help=f"vec0 chunk_size (default {VEC_CHUNK_SIZE})")
    ap.add_argument("--measure", action="store_true",
                    help="compare file size + KNN latency across chunk sizes")
    ap.add_argument("--force", action="store_true",
                    help="overwrite dst if it exists")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 1
    if args.src.resolve() == args.dst.resolve():
        print("src and dst must differ; migrate to a new file, verify, "
              "then swap", file=sys.stderr)
        return 1
    if args.dst.exists() and not args.force:
        print(f"{args.dst} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    src = _connect(args.src)
    try:
        version = schema_version(src)
        if version != 1:
            print(f"source is schema v{version}, expected v1", file=sys.stderr)
            return 1
        content = load_content(src)
    finally:
        src.close()

    n_chunks = sum(len(v) for v in content["chunks"].values())
    print(f"source: {args.src} ({args.src.stat().st_size / 1e6:.2f} MB) — "
          f"{len(content['species'])} species, {n_chunks} chunks, "
          f"{len(content['vectors'])} vectors")

    if args.measure:
        measure(content, [8, 16, 32, 64, 128])

    # Build next to the destination, verify, then move into place so a
    # failed or interrupted run never leaves a half-written corpus at dst.
    tmp = args.dst.with_name(args.dst.name + ".tmp")
    meta = build_v2(content, tmp, args.chunk_size)
    problems = verify(content, tmp)
    if problems:
        print("VERIFY FAILED — destination not written:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return 1
    os.replace(tmp, args.dst)

    print(f"\nwrote {args.dst} ({args.dst.stat().st_size / 1e6:.2f} MB, "
          f"chunk_size={args.chunk_size}) — verified: counts, vector bytes, "
          f"partition metadata, per-species hashes")
    for key in ("schema_version", "corpus_version", "content_sha256",
                "sqlite_vec_version", "species_count", "chunk_count"):
        print(f"  {key:20s} {meta[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
