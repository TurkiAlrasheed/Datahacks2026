"""
Interactive retrieval tester for the species corpus.

Run this on your laptop to verify that vector search returns sensible chunks
before we wire in the LLM. If retrieval is bad here, the LLM cannot fix it.

Usage:
    python species_identification/tests/test_corpus.py \
        species_identification/offline-info/corpus.db

Then type queries like:
    > Apis_mellifera | what do they eat
    > Procyon_lotor | are they dangerous
    > Quercus_agrifolia | how tall does it grow
    > Canis_latrans | habitat

Type 'list' to see all species, 'quit' to exit.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_schema import (  # noqa: E402
    BLURB_CATEGORY,
    EMBED_DIM,
    EMBED_MODEL,
    check_corpus_schema,
)

TOP_K = 3
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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

# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------
# Visitors phrase questions with pronouns ("are they smart", "can it swim")
# that don't carry the species name into the embedding. bge-small-en-v1.5
# struggles to bridge that to articles that name the species in nearly every
# sentence. We measured a +0.16 cosine sim delta from rephrasing the same
# question with the species name; this rewriter captures most of that gain
# at zero inference cost.
#
# Cache the common name lookup so we don't hit the DB on every retrieve.
_COMMON_NAME_CACHE: dict[str, str] = {}

# Pronouns that signal an "ungrounded" query - one that doesn't name its
# subject. Padded with spaces so we don't match inside other words.
_UNGROUNDED_MARKERS = (
    " they ", " them ", " their ", " they're ", " theyre ",
    " it ", " its ", " it's ",
    " these ", " those ", " this ", " that ",
)


def _get_common_name(conn: sqlite3.Connection, species_id: str) -> str:
    """
    Look up the common name for a species, cached. Prefers the blurb's
    common_name (the species.common_name column is only set when the
    Wikipedia title differs from the binomial, so it's empty for about half
    the corpus), then the column, then the binomial, then the species_id
    with underscores replaced.
    """
    if species_id in _COMMON_NAME_CACHE:
        return _COMMON_NAME_CACHE[species_id]
    row = conn.execute(
        "SELECT blurb_json, common_name, species_name FROM species "
        "WHERE species_id = ?",
        (species_id,),
    ).fetchone()
    name = species_id.replace("_", " ")
    if row:
        blurb_json, common, binomial = row
        blurb_common = None
        if blurb_json:
            try:
                blurb_common = json.loads(blurb_json).get("common_name")
            except (json.JSONDecodeError, AttributeError):
                blurb_common = None
        name = blurb_common or common or binomial or name
    _COMMON_NAME_CACHE[species_id] = name
    return name


def rewrite_query(query: str, common_name: str) -> str:
    """
    If the query uses pronouns or generic determiners ("they", "it",
    "this", etc.), prepend the species name so the embedding has
    something to ground on. Otherwise return the query unchanged.
    """
    padded = f" {query.lower().strip()} "
    if any(marker in padded for marker in _UNGROUNDED_MARKERS):
        return f"{common_name}: {query}"
    return query

def embed_query(embedder, text: str) -> bytes:
    """Embed a query with the bge retrieval prefix; returns vec0 float32 bytes."""
    q_emb = embedder.encode(
        [BGE_QUERY_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return struct.pack(f"{EMBED_DIM}f", *q_emb.tolist())


def knn_for_species(conn: sqlite3.Connection, q_bytes: bytes,
                    species_id: str, k: int) -> list[tuple]:
    """
    Exact top-k chunks for ONE species, nearest first, as
    (chunk_id, category, text, l2_distance).

    `species_id` is the vec0 partition key, so the KNN only scans that
    species' vectors and always returns its true top-k (or all of them, if
    it has fewer). The blurb chunk is excluded inside the KNN: it is always
    injected into the prompt as SPECIES FACTS, so retrieving it would only
    duplicate it and take a snippet slot.

    The KNN runs in a CTE so the partition/metadata constraints are handed
    to vec0 directly rather than depending on how the planner treats a join.
    """
    return conn.execute("""
        WITH knn AS (
            SELECT rowid, distance
            FROM chunk_vectors
            WHERE embedding MATCH ?
              AND k = ?
              AND species_id = ?
              AND category != ?
        )
        SELECT c.id, c.category, c.text, knn.distance
        FROM knn
        JOIN chunks c ON c.id = knn.rowid
        ORDER BY knn.distance
    """, (q_bytes, k, species_id, BLURB_CATEGORY)).fetchall()


def retrieve(conn: sqlite3.Connection, embedder,
             species_id: str, query: str, k: int = TOP_K) -> list[tuple]:
    """
    Retrieve the top-k non-blurb chunks for a species as
    (category, text, l2_distance) tuples, nearest first.

    sqlite-vec's vec0 uses L2 distance by default. Since our embeddings are
    L2-normalized, L2 distance d and cosine similarity c are related by:
        d^2 = 2 - 2c    ->    c = 1 - d^2/2
    So L2 distance of 0.5 -> cosine sim 0.875 (very relevant)
       L2 distance of 0.7 -> cosine sim 0.755 (relevant)
       L2 distance of 1.0 -> cosine sim 0.50  (marginal)
       L2 distance of 1.4 -> cosine sim ~0.0  (unrelated / orthogonal)
    """
    # Inject the species name when the query uses pronouns. This recovers
    # most of the cosine-sim gap measured between colloquial and encyclopedic
    # phrasing of the same question.
    rewritten = rewrite_query(query, _get_common_name(conn, species_id))
    q_bytes = embed_query(embedder, rewritten)
    return [(cat, text, dist) for _, cat, text, dist
            in knn_for_species(conn, q_bytes, species_id, k)]


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

    from sentence_transformers import SentenceTransformer

    conn = open_db(db)
    check_corpus_schema(conn)
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
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
        print("Usage: python test_corpus.py <corpus.db>")
        sys.exit(1)
    main(sys.argv[1])