"""
Shared schema for corpus.db (schema v2) — one place for the DDL, the vector
insert, the version/manifest bookkeeping, and the runtime schema check.

Imported by build_corpus.py, compile_blurbs.py, migrate_corpus_v2.py, the
retriever in tests/test_corpus.py, and the integration test fixture, so the
builder and the runtime can't drift apart. Deliberately stdlib-only (plus the
sqlite-vec connection it is handed) so the on-device runtime can import it
without pulling in sentence-transformers.

Schema v2 vs v1
---------------
v1 had `chunk_vectors USING vec0(embedding float[384])` with no species
column. Retrieval had to run a GLOBAL k=50 KNN and drop other species' rows
after the join, which silently starved species whose chunks weren't in the
global top 50 (measured: 20% of grounded queries missed a better chunk,
52% of ungrounded queries got zero chunks).

v2 makes species_id a PARTITION KEY on the vec0 table, so a KNN query scoped
with `species_id = ?` only scans that species' vectors and always returns its
true top-k. `category` is a metadata column so the always-injected blurb
chunk can be excluded inside the KNN itself (`category != 'blurb'`).

chunk_size: vec0 allocates one fixed-size vector chunk per partition value
(default 1024 slots = ~1.5 MB of float[384]). With ~17 chunks per species the
default inflates the DB, so v2 pins a small chunk_size. Measured with
`migrate_corpus_v2.py --measure` on the 50-species / 871-chunk corpus
(v1 file: 2.87 MB):

    chunk_size    file MB   partition KNN p50
         8          3.04        0.060 ms
        16          3.37        0.061 ms   <- chosen
        32          4.36        0.089 ms
       128         11.46        0.149 ms
      1024         81.35        1.246 ms   (vec0 default)

Versioning (OTA foundations)
----------------------------
`corpus_meta` holds schema_version, corpus_version, embed model/dim, the
sqlite-vec version that built the file, and content_sha256. `species_manifest`
holds a per-species content hash, so two corpora can be diffed species by
species (the unit a partition-scoped delta update would ship).
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import datetime, timezone

SCHEMA_VERSION = 2

# Must match the model the vectors were embedded with. build_corpus.py and
# compile_blurbs.py embed with this; test_corpus.py queries with it.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Vector slots per partition chunk. Must be a multiple of 8. See the module
# docstring for the size/latency measurements behind this value.
VEC_CHUNK_SIZE = 16

# Chunk category written by compile_blurbs.py. The blurb is always injected
# into the prompt as SPECIES FACTS, so retrieval excludes it.
BLURB_CATEGORY = "blurb"

MIGRATION_HINT = (
    "Migrate it with:\n"
    "    python species_identification/migrate_corpus_v2.py "
    "<old corpus.db> <new corpus.db>"
)


class CorpusSchemaError(RuntimeError):
    """corpus.db is missing, too old, or was built with a different embedder."""


# ---------------------------------------------------------------------------
# DDL + writes
# ---------------------------------------------------------------------------

def create_schema(conn: sqlite3.Connection,
                  chunk_size: int = VEC_CHUNK_SIZE) -> None:
    """Create an empty v2 corpus. Caller must have loaded sqlite-vec."""
    if chunk_size <= 0 or chunk_size % 8:
        raise ValueError(f"chunk_size must be a positive multiple of 8, "
                         f"got {chunk_size}")
    conn.executescript(f"""
        CREATE TABLE species (
            species_id   TEXT PRIMARY KEY,
            species_name TEXT NOT NULL,
            common_name  TEXT,
            source       TEXT,
            blurb_text   TEXT,
            blurb_json   TEXT
        );

        CREATE TABLE chunks (
            id           INTEGER PRIMARY KEY,
            species_id   TEXT NOT NULL REFERENCES species(species_id),
            category     TEXT NOT NULL,
            text         TEXT NOT NULL
        );

        CREATE INDEX idx_chunks_species ON chunks(species_id);
        CREATE INDEX idx_chunks_species_cat ON chunks(species_id, category);

        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            species_id TEXT PARTITION KEY,
            category   TEXT,
            embedding  float[{EMBED_DIM}],
            chunk_size = {chunk_size}
        );

        CREATE TABLE corpus_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE species_manifest (
            species_id TEXT PRIMARY KEY,
            n_chunks   INTEGER NOT NULL,
            sha256     TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO corpus_meta(key, value) VALUES "
                 "('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()


def pack_embedding(embedding) -> bytes:
    """float32 little-endian bytes, the format vec0 stores."""
    values = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    if len(values) != EMBED_DIM:
        raise ValueError(f"expected {EMBED_DIM}-dim embedding, "
                         f"got {len(values)}")
    return struct.pack(f"{EMBED_DIM}f", *values)


def insert_vector(conn: sqlite3.Connection, rowid: int, species_id: str,
                  category: str, embedding_bytes: bytes) -> None:
    """Insert one chunk vector. rowid must equal chunks.id."""
    conn.execute(
        "INSERT INTO chunk_vectors(rowid, species_id, category, embedding) "
        "VALUES (?, ?, ?, ?)",
        (rowid, species_id, category, embedding_bytes),
    )


# ---------------------------------------------------------------------------
# Content hashing + manifest
# ---------------------------------------------------------------------------

_SPECIES_FIELDS = ("species_id", "species_name", "common_name", "source",
                   "blurb_text", "blurb_json")


def _field_bytes(value) -> bytes:
    """Length-prefixed encoding so ('ab','c') and ('a','bc') hash differently."""
    if value is None:
        return b"\x00"
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return b"\x01" + len(raw).to_bytes(8, "little") + raw


def load_content(conn: sqlite3.Connection) -> dict:
    """
    Read everything that defines corpus content, schema-agnostic (works on
    v1 and v2, since both keep vectors keyed by chunks.id).

    Returns {"species": {sid: row_dict}, "chunks": {sid: [(id, cat, text)]},
             "vectors": {rowid: bytes}}.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(species)")}
    select = [c if c in cols else f"NULL AS {c}" for c in _SPECIES_FIELDS]
    species = {}
    for row in conn.execute(f"SELECT {', '.join(select)} FROM species"):
        species[row[0]] = dict(zip(_SPECIES_FIELDS, row))

    chunks: dict[str, list[tuple[int, str, str]]] = {}
    for cid, sid, cat, text in conn.execute(
            "SELECT id, species_id, category, text FROM chunks ORDER BY id"):
        chunks.setdefault(sid, []).append((cid, cat, text))

    vectors = {rowid: bytes(emb) for rowid, emb in conn.execute(
        "SELECT rowid, embedding FROM chunk_vectors")}
    return {"species": species, "chunks": chunks, "vectors": vectors}


def species_hashes(content: dict) -> dict[str, tuple[int, str]]:
    """
    Per-species (n_chunks, sha256). Independent of chunk ids, so the same
    content rebuilt with different rowids hashes identically — what a
    species-level delta update needs to decide "changed or not".
    """
    out = {}
    for sid, row in content["species"].items():
        h = hashlib.sha256()
        for f in _SPECIES_FIELDS:
            h.update(_field_bytes(row[f]))
        rows = content["chunks"].get(sid, [])
        items = sorted(
            (cat, text, content["vectors"].get(cid, b""))
            for cid, cat, text in rows
        )
        for cat, text, vec in items:
            h.update(_field_bytes(cat))
            h.update(_field_bytes(text))
            h.update(_field_bytes(vec))
        out[sid] = (len(rows), h.hexdigest())
    return out


def refresh_manifest(conn: sqlite3.Connection,
                     corpus_version: str | None = None) -> dict:
    """
    Recompute species_manifest + corpus_meta from current content. Call after
    any write (build, blurb compile, migration). Returns the meta dict.
    """
    content = load_content(conn)
    hashes = species_hashes(content)

    conn.execute("DELETE FROM species_manifest")
    conn.executemany(
        "INSERT INTO species_manifest(species_id, n_chunks, sha256) "
        "VALUES (?, ?, ?)",
        [(sid, n, sha) for sid, (n, sha) in sorted(hashes.items())],
    )

    top = hashlib.sha256()
    top.update(f"schema={SCHEMA_VERSION};model={EMBED_MODEL};"
               f"dim={EMBED_DIM}\n".encode())
    for sid, (_, sha) in sorted(hashes.items()):
        top.update(f"{sid}:{sha}\n".encode())
    content_sha = top.hexdigest()

    now = datetime.now(timezone.utc)
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "corpus_version": corpus_version
                          or f"{now:%Y.%m.%d}+{content_sha[:8]}",
        "embed_model": EMBED_MODEL,
        "embed_dim": str(EMBED_DIM),
        "sqlite_vec_version": conn.execute(
            "SELECT vec_version()").fetchone()[0],
        "built_at": now.isoformat(timespec="seconds"),
        "content_sha256": content_sha,
        "species_count": str(len(hashes)),
        "chunk_count": str(sum(n for n, _ in hashes.values())),
    }
    conn.executemany(
        "INSERT INTO corpus_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        list(meta.items()),
    )
    conn.commit()
    return meta


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------

def schema_version(conn: sqlite3.Connection) -> int:
    """2 for a partitioned corpus, 1 for the legacy layout, 0 if not a corpus."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_vectors'"
    ).fetchone()
    if row is None:
        return 0
    ddl = " ".join((row[0] or "").lower().split())
    return 2 if "partition key" in ddl else 1


def read_meta(conn: sqlite3.Connection) -> dict:
    try:
        return dict(conn.execute("SELECT key, value FROM corpus_meta"))
    except sqlite3.OperationalError:
        return {}


def check_corpus_schema(conn: sqlite3.Connection) -> dict:
    """
    Fail fast (at pipeline build, before the visitor is waiting) if the DB
    isn't a v2 corpus built with our embedder. Returns corpus_meta.
    """
    version = schema_version(conn)
    if version == 0:
        raise CorpusSchemaError(
            "corpus.db has no chunk_vectors table — is this the right file?")
    if version < SCHEMA_VERSION:
        raise CorpusSchemaError(
            f"corpus.db uses schema v{version} (global vector index, no "
            f"species partition). Retrieval requires v{SCHEMA_VERSION}.\n"
            + MIGRATION_HINT)
    meta = read_meta(conn)
    if meta.get("schema_version") != str(SCHEMA_VERSION):
        raise CorpusSchemaError(
            f"corpus_meta.schema_version is {meta.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}.\n" + MIGRATION_HINT)
    model = meta.get("embed_model")
    if model and model != EMBED_MODEL:
        raise CorpusSchemaError(
            f"corpus.db was embedded with {model!r} but the runtime queries "
            f"with {EMBED_MODEL!r}; similarities would be meaningless.")
    return meta
