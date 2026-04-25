"""
Interactive retrieval tester for the species corpus.

Run this on your laptop to verify that vector search returns sensible chunks
before we wire in the LLM. If retrieval is bad here, the LLM cannot fix it.

Usage:
    python test_retrieval.py species-id/corpus.db

Then type queries like:
    > Apis_mellifera | what do they eat
    > Procyon_lotor | are they dangerous
    > Quercus_agrifolia | how tall does it grow
    > Canis_latrans | habitat

Type 'list' to see all species, 'quit' to exit.
"""

from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
TOP_K = 3


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def list_species(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT species_id, species_name, common_name, "
        "       (SELECT COUNT(*) FROM chunks WHERE chunks.species_id = species.species_id) "
        "FROM species ORDER BY species_id"
    ).fetchall()
    print(f"\n{len(rows)} species in corpus:")
    for sid, sname, cname, n in rows:
        label = f"{sname}" + (f" ({cname})" if cname and cname.lower() != sname.lower() else "")
        print(f"  {sid:35s} {n:3d} chunks  {label}")
    print()


def retrieve(conn: sqlite3.Connection, embedder: SentenceTransformer,
             species_id: str, query: str, k: int = TOP_K) -> list[tuple]:
    """
    Retrieve top-k chunks for a species, filtered by species_id.

    sqlite-vec's vec0 uses L2 distance by default. Since our embeddings are
    L2-normalized, L2 distance d and cosine similarity c are related by:
        d^2 = 2 - 2c    ->    c = 1 - d^2/2
    So L2 distance of 0.5 -> cosine sim 0.875 (very relevant)
       L2 distance of 0.7 -> cosine sim 0.755 (relevant)
       L2 distance of 1.0 -> cosine sim 0.50  (marginal)
       L2 distance of 1.4 -> cosine sim ~0.0  (unrelated / orthogonal)
    """
    # bge models are recommended to be used with a query prefix for retrieval
    query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
    q_emb = embedder.encode(
        [query_with_prefix],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    q_bytes = struct.pack(f"{EMBED_DIM}f", *q_emb.tolist())

    rows = conn.execute("""
        SELECT c.category, c.text, v.distance
        FROM chunk_vectors v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ?
          AND c.species_id = ?
          AND k = 50
        ORDER BY v.distance
        LIMIT ?
    """, (q_bytes, species_id, k)).fetchall()
    return rows


def l2_to_cosine(d: float) -> float:
    """Convert L2 distance between unit vectors to cosine similarity."""
    return 1.0 - (d * d) / 2.0


def main(db_path: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    db = Path(db_path)
    if not db.exists():
        print(f"Corpus not found: {db}")
        sys.exit(1)

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    conn = open_db(db)
    print(f"Corpus: {db}")
    print("Type 'list' to see species, 'quit' to exit.")
    print("Format:  species_id | your question\n")

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
            list_species(conn)
            continue
        if "|" not in line:
            print("  Format: species_id | question")
            continue

        species_id, _, query = line.partition("|")
        species_id = species_id.strip()
        query = query.strip()
        if not species_id or not query:
            print("  Both species_id and question are required.")
            continue

        # Verify the species exists
        exists = conn.execute(
            "SELECT species_name, common_name FROM species WHERE species_id = ?",
            (species_id,),
        ).fetchone()
        if not exists:
            print(f"  Unknown species '{species_id}'. Type 'list' to see options.")
            continue

        sname, cname = exists
        display = sname + (f" ({cname})" if cname and cname.lower() != sname.lower() else "")
        print(f"\n  Species: {display}")
        print(f"  Query:   {query}\n")

        results = retrieve(conn, embedder, species_id, query)
        if not results:
            print("  No results.\n")
            continue

        for i, (category, text, dist) in enumerate(results, 1):
            cos_sim = l2_to_cosine(dist)
            # Relevance heuristic based on cosine similarity
            if cos_sim >= 0.80:
                verdict = "STRONG"
            elif cos_sim >= 0.70:
                verdict = "GOOD"
            elif cos_sim >= 0.55:
                verdict = "MARGINAL"
            else:
                verdict = "WEAK"
            snippet = text if len(text) < 400 else text[:400] + "..."
            print(f"  [{i}] L2={dist:.3f}  cosine_sim={cos_sim:.3f}  [{verdict}]  category={category}")
            print(f"      {snippet}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_retrieval.py <corpus.db>")
        sys.exit(1)
    main(sys.argv[1])