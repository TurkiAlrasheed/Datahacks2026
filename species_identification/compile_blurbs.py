"""
Compile reviewed blurbs from blurbs.yaml into the corpus DB built by
build_corpus.py.

Run order (this matters):
    1. python build_corpus.py species.json corpus.db
    2. python build_blurbs.py species.json blurbs.yaml   # draft
    3. (review blurbs.yaml by hand, set reviewed: true)
    4. python compile_blurbs.py blurbs.yaml corpus.db    # this script

This script does three things to the existing corpus.db:

  1. Adds two columns to the `species` table (idempotently): `blurb_text`
     holds the rendered string the runtime speaks for the Phase 1 intro,
     and `blurb_json` holds the structured fields as JSON for any
     field-specific lookup.

  2. Inserts one new row per species into `chunks` with `category = 'blurb'`,
     containing the rendered blurb. This row is also embedded and inserted
     into `chunk_vectors` so the RAG retriever can find it via cosine
     similarity along with the Wikipedia chunks. The runtime should also
     pin this row by species_id so it's guaranteed in context regardless
     of similarity score.

  3. Refuses to do anything if any blurb has reviewed: false, or if any
     blurb references a species_id missing from the corpus and the blurb
     itself doesn't carry enough info to create a species row.

Idempotent: re-running deletes existing blurb chunks for each species
before re-inserting, so you can iterate on blurbs.yaml without rebuilding
the whole corpus.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Must match build_corpus.py. If you change the embedding model there,
# change it here too — mismatched embeddings ruin retrieval silently.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Order in which fields appear in the rendered blurb text. Stable order
# matters: it's both what the LLM sees as RAG context and (potentially)
# what the runtime speaks. Skip fields that aren't present.
FIELD_ORDER = [
    ("common_name",         "Common name"),
    ("size",                "Size"),
    ("appearance",          "Appearance"),
    ("habitat",             "Habitat"),
    ("diet",                "Diet"),
    ("behavior",            "Behavior"),
    ("dangerous_to_humans", "Danger to humans"),
    ("dangerous_to_pets",   "Danger to pets"),
    ("notable",             "Notable"),
]

# Fields that are part of the user-facing blurb. `reviewed` and `source`
# are metadata and stay out of both rendered text and JSON payload.
METADATA_FIELDS = {"reviewed", "source"}


def render_blurb(species_id: str, fields: dict) -> str:
    """
    Render a blurb dict into a labeled-line text blob.

    Labeled lines are deliberate — SmolLM2-360M reads "Diet: insects" far
    more reliably than the same fact buried in prose. The species_id
    (binomial) goes on the first line for both retrieval keying and so
    the LLM has the scientific name when answering.
    """
    binomial = species_id.replace("_", " ")
    lines = [f"Species: {binomial}"]
    for key, label in FIELD_ORDER:
        value = fields.get(key)
        if value is None or value == "":
            continue
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def load_blurbs(path: Path) -> dict[str, dict]:
    """Load and validate the blurbs YAML. Hard-fails on any unreviewed entry."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    blurbs = raw.get("blurbs", {})
    if not isinstance(blurbs, dict) or not blurbs:
        sys.exit(f"[FATAL] {path} has no blurbs to compile.")

    unreviewed = [sid for sid, b in blurbs.items()
                  if not b.get("reviewed", False)]
    if unreviewed:
        print(f"[FATAL] {len(unreviewed)} blurb(s) still have reviewed: false.",
              file=sys.stderr)
        print("Review them by hand and flip the flag before compiling:",
              file=sys.stderr)
        for sid in unreviewed:
            print(f"  - {sid}", file=sys.stderr)
        sys.exit(1)

    return blurbs


def open_corpus(db_path: Path) -> sqlite3.Connection:
    """Open the corpus DB built by build_corpus.py and load sqlite-vec."""
    if not db_path.exists():
        sys.exit(f"[FATAL] {db_path} does not exist. "
                 f"Run build_corpus.py first.")
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def ensure_blurb_columns(conn: sqlite3.Connection) -> None:
    """
    Add blurb_text and blurb_json columns to the species table if missing.
    Safe to call repeatedly; the corpus rebuild will recreate the table
    without these columns, and we'll add them back here.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(species)")}
    if "blurb_text" not in cols:
        conn.execute("ALTER TABLE species ADD COLUMN blurb_text TEXT")
    if "blurb_json" not in cols:
        conn.execute("ALTER TABLE species ADD COLUMN blurb_json TEXT")
    conn.commit()


def upsert_species_row(conn: sqlite3.Connection, species_id: str,
                       blurb: dict) -> None:
    """
    Make sure the species row exists. If build_corpus.py already inserted
    it (the common case), leave species_name/common_name/source alone. If
    it doesn't exist (Wikipedia missing for this species), create it from
    the blurb metadata so retrieval has something to key against.
    """
    row = conn.execute(
        "SELECT species_id FROM species WHERE species_id = ?",
        (species_id,),
    ).fetchone()
    if row is None:
        binomial = species_id.replace("_", " ")
        conn.execute(
            "INSERT INTO species(species_id, species_name, common_name, source) "
            "VALUES (?, ?, ?, ?)",
            (
                species_id,
                binomial,
                blurb.get("common_name", ""),
                blurb.get("source", "manual"),
            ),
        )


def delete_existing_blurb_chunks(conn: sqlite3.Connection,
                                 species_id: str) -> None:
    """
    Remove any prior blurb chunk for this species (and its vector row), so
    re-running the compile step replaces rather than duplicates.
    """
    rows = conn.execute(
        "SELECT id FROM chunks WHERE species_id = ? AND category = 'blurb'",
        (species_id,),
    ).fetchall()
    for (rowid,) in rows:
        conn.execute("DELETE FROM chunk_vectors WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (rowid,))


def insert_blurb_chunk(conn: sqlite3.Connection, species_id: str,
                       text: str, embedding) -> None:
    """Insert the blurb as a chunk + matching vector row."""
    cur = conn.execute(
        "INSERT INTO chunks(species_id, category, text) VALUES (?, 'blurb', ?)",
        (species_id, text),
    )
    rowid = cur.lastrowid
    emb_bytes = struct.pack(f"{EMBED_DIM}f", *embedding.tolist())
    conn.execute(
        "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
        (rowid, emb_bytes),
    )


def update_species_blurb_payload(conn: sqlite3.Connection, species_id: str,
                                 rendered: str, fields_json: str) -> None:
    """Stash the rendered text and parsed JSON on the species row."""
    conn.execute(
        "UPDATE species SET blurb_text = ?, blurb_json = ? WHERE species_id = ?",
        (rendered, fields_json, species_id),
    )


def main(blurbs_path: str, db_path: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    blurbs_file = Path(blurbs_path)
    db_file = Path(db_path)

    blurbs = load_blurbs(blurbs_file)
    print(f"Loaded {len(blurbs)} reviewed blurbs from {blurbs_file}.")

    conn = open_corpus(db_file)
    ensure_blurb_columns(conn)

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Render everything first so we can embed in one batch — sentence-
    # transformers is meaningfully faster batched than per-call.
    rendered_by_id: dict[str, str] = {}
    payload_by_id: dict[str, str] = {}
    for species_id, blurb in blurbs.items():
        fields = {k: v for k, v in blurb.items() if k not in METADATA_FIELDS}
        rendered_by_id[species_id] = render_blurb(species_id, fields)
        payload_by_id[species_id] = json.dumps(fields, ensure_ascii=False)

    species_ids = list(rendered_by_id.keys())
    texts = [rendered_by_id[sid] for sid in species_ids]

    print(f"Embedding {len(texts)} blurbs...")
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Writing to DB...")
    for species_id, embedding in tqdm(
        zip(species_ids, embeddings),
        total=len(species_ids),
        desc="Compiling",
    ):
        blurb = blurbs[species_id]
        rendered = rendered_by_id[species_id]
        fields_json = payload_by_id[species_id]

        upsert_species_row(conn, species_id, blurb)
        delete_existing_blurb_chunks(conn, species_id)
        insert_blurb_chunk(conn, species_id, rendered, embedding)
        update_species_blurb_payload(conn, species_id, rendered, fields_json)

    conn.commit()

    # Sanity check: how many species rows now have blurbs vs total?
    total_species = conn.execute(
        "SELECT COUNT(*) FROM species"
    ).fetchone()[0]
    with_blurbs = conn.execute(
        "SELECT COUNT(*) FROM species WHERE blurb_text IS NOT NULL"
    ).fetchone()[0]
    blurb_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE category = 'blurb'"
    ).fetchone()[0]

    conn.close()

    print(f"\nDone.")
    print(f"  Species in corpus:      {total_species}")
    print(f"  Species with blurb:     {with_blurbs}")
    print(f"  Blurb chunks indexed:   {blurb_chunks}")
    if with_blurbs < total_species:
        missing = total_species - with_blurbs
        print(f"\n  Note: {missing} species in the corpus have no blurb. "
              f"The runtime will fall back to RAG over Wikipedia chunks "
              f"for those.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compile_blurbs.py <blurbs.yaml> <corpus.db>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])