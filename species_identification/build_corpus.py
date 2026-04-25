"""
Corpus builder for GameExplorer species agent.

Builds a SQLite + sqlite-vec database of Wikipedia-sourced species information,
chunked and embedded with bge-small-en-v1.5, for offline RAG on the UNO Q.

Run on your laptop, ship the output .db file to the device.

Usage:
    pip install requests sentence-transformers sqlite-vec tqdm
    python build_corpus.py species.json corpus.db

Outputs:
    corpus.db          - SQLite file with chunks + vectors, ready to ship
    corpus_dump.json   - Human-readable dump of every chunk for review
    build_report.txt   - Summary of coverage issues to review manually
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import sqlite_vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "GameExplorer-CorpusBuilder/1.0 (educational project)"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

CHUNK_TOKENS = 200  # approx; we use word count as a cheap proxy
CHUNK_OVERLAP = 30
MIN_CHUNK_WORDS = 20  # drop trivially short chunks

# Wikipedia section headers -> canonical category.
# Keys are matched case-insensitively as substrings of the header.
SECTION_MAP = {
    "description": "description",
    "identification": "description",
    "appearance": "description",
    "morphology": "description",
    "anatomy": "description",
    "distribution": "habitat",
    "habitat": "habitat",
    "range": "habitat",
    "ecology": "habitat",
    "diet": "diet",
    "feeding": "diet",
    "food": "diet",
    "foraging": "diet",
    "behavior": "behavior",
    "behaviour": "behavior",
    "reproduction": "behavior",
    "breeding": "behavior",
    "life cycle": "behavior",
    "lifecycle": "behavior",
    "conservation": "conservation",
    "status": "conservation",
    "threats": "conservation",
    "relationship with humans": "humans",
    "cultural": "humans",
    "uses": "humans",
}

# Special non-species classes the classifier can output.
# These are skipped during corpus building.
NON_SPECIES_CLASSES = {"seashore"}

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class Chunk:
    species_id: str       # e.g. "Apis_mellifera"
    species_name: str     # e.g. "Apis mellifera"
    common_name: str      # e.g. "Western honey bee" (if available)
    category: str         # "general" | "description" | "habitat" | ...
    text: str
    source: str           # "wikipedia:{page_title}"


@dataclass
class BuildReport:
    ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    short: list[tuple[str, int]] = field(default_factory=list)  # (species, char_count)
    skipped: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        lines = []
        lines.append(f"Built corpus for {len(self.ok)} species.\n")
        if self.skipped:
            lines.append(f"\nSkipped {len(self.skipped)} non-species class(es):")
            for s in self.skipped:
                lines.append(f"  - {s}")
        if self.missing:
            lines.append(f"\nMISSING: {len(self.missing)} species had no Wikipedia content.")
            lines.append("These need manual curation:")
            for s in self.missing:
                lines.append(f"  - {s}")
        if self.short:
            lines.append(f"\nSHORT: {len(self.short)} species had < 1000 chars of content.")
            lines.append("These may benefit from supplemental sources:")
            for s, n in sorted(self.short, key=lambda x: x[1]):
                lines.append(f"  - {s} ({n} chars)")
        path.write_text("\n".join(lines), encoding="utf-8")

# -----------------------------------------------------------------------------
# Wikipedia fetching
# -----------------------------------------------------------------------------

def _wiki_get(params: dict) -> dict:
    """Thin wrapper around the Wikipedia action API with retries."""
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(3):
        try:
            r = requests.get(
                WIKI_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    return {}


def fetch_article(species_binomial: str) -> tuple[str, str, str] | None:
    """
    Fetch the Wikipedia article for a species.

    Returns (page_title, common_name, plain_text) or None if not found.
    The article is requested as plain text (no wiki markup, no HTML).
    """
    # Try the exact binomial first. Wikipedia handles redirects automatically
    # when we pass redirects=1.
    data = _wiki_get({
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": species_binomial,
    })

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        # Fall back to search
        search = _wiki_get({
            "action": "query",
            "list": "search",
            "srsearch": species_binomial,
            "srlimit": "1",
        })
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        data = _wiki_get({
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages or not pages[0].get("extract"):
            return None
        page = pages[0]

    title = page["title"]
    text = page["extract"]

    # Common name heuristic: if the page title isn't the binomial, it's
    # probably the common name (Wikipedia prefers common names for titles).
    common_name = title if title.lower() != species_binomial.lower() else ""

    return title, common_name, text


# -----------------------------------------------------------------------------
# Text processing
# -----------------------------------------------------------------------------

def categorize_section(header: str) -> str:
    h = header.lower()
    for key, category in SECTION_MAP.items():
        if key in h:
            return category
    return "other"


def split_sections(plain_text: str) -> list[tuple[str, str]]:
    """
    Split a Wikipedia plain-text extract into (header, body) sections.

    Wikipedia's explaintext format uses lines like:
        == Description ==
        == Habitat and distribution ==
    Subsections use === and ====. We collapse to top-level sections.
    """
    sections: list[tuple[str, str]] = [("__lead__", "")]
    lines = plain_text.split("\n")
    current_header = "__lead__"
    current_body: list[str] = []

    section_re = re.compile(r"^\s*(={2,})\s*(.+?)\s*\1\s*$")

    for line in lines:
        m = section_re.match(line)
        if m and len(m.group(1)) == 2:  # only top-level ==header==
            # flush previous
            sections[-1] = (current_header, "\n".join(current_body).strip())
            current_header = m.group(2)
            current_body = []
            sections.append((current_header, ""))
        elif m and len(m.group(1)) > 2:
            # subsection - keep content but don't start a new section
            current_body.append("")  # paragraph break
        else:
            current_body.append(line)

    sections[-1] = (current_header, "\n".join(current_body).strip())

    # Drop sections we never want: references, external links, see also, etc.
    blocklist = {"references", "external links", "see also", "notes",
                 "further reading", "bibliography", "gallery"}
    return [(h, b) for h, b in sections
            if b.strip() and h.lower().strip() not in blocklist]


def clean_text(text: str) -> str:
    """Light cleanup of Wikipedia plaintext."""
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop parenthetical citation remnants like "(Smith, 2001)" - rare in explaintext
    # but sometimes leak through
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences with a simple regex.

    Not linguistically perfect (doesn't handle 'Dr. Smith' etc. flawlessly)
    but good enough for Wikipedia prose and much better than word-based
    cutting. We preserve paragraph breaks as their own 'sentence' so the
    rejoined text keeps its structure.
    """
    # First protect common abbreviations that would cause false splits.
    # These are the ones that actually show up in species articles.
    protected = text
    for abbr in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Mt.",
                 "e.g.", "i.e.", "etc.", "cf.", "vs.", "approx.",
                 "subsp.", "var.", "spp.", "sp.", "ca.", "c."]:
        protected = protected.replace(abbr, abbr.replace(".", "<!DOT!>"))

    # Split on sentence boundaries: . ! ? followed by whitespace + capital
    # letter, OR paragraph break.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n\n+", protected)
    sentences = [p.replace("<!DOT!>", ".").strip() for p in parts if p.strip()]
    return sentences


def chunk_text(text: str, target_words: int = CHUNK_TOKENS,
               overlap_sentences: int = 2) -> list[str]:
    """
    Split text into sentence-aware chunks of approximately target_words.

    Strategy: accumulate whole sentences until adding the next would exceed
    the target, then emit a chunk. Overlap is the last N sentences of the
    previous chunk prepended to the next one - this keeps semantic context
    across boundaries without cutting mid-sentence.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush():
        if current and sum(len(s.split()) for s in current) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(current).strip())

    for sent in sentences:
        sent_words = len(sent.split())

        # If a single sentence alone exceeds the target, emit it as its own
        # chunk (better than splitting it mid-sentence).
        if sent_words >= target_words and not current:
            chunks.append(sent.strip())
            continue

        # If adding this sentence would overflow, flush and start a new
        # chunk with the last `overlap_sentences` sentences as preamble.
        if current_words + sent_words > target_words and current:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_words = sum(len(s.split()) for s in current)

        current.append(sent)
        current_words += sent_words

    flush()
    return chunks


# -----------------------------------------------------------------------------
# Per-species build
# -----------------------------------------------------------------------------

def build_chunks_for_species(species_id: str) -> list[Chunk]:
    """
    Fetch + process one species. Returns an empty list if not found.
    """
    binomial = species_id.replace("_", " ")
    result = fetch_article(binomial)
    if not result:
        return []

    page_title, common_name, text = result
    text = clean_text(text)
    if len(text) < 200:
        return []

    source = f"wikipedia:{page_title}"
    chunks: list[Chunk] = []

    sections = split_sections(text)
    for header, body in sections:
        category = "general" if header == "__lead__" else categorize_section(header)
        for piece in chunk_text(body):
            chunks.append(Chunk(
                species_id=species_id,
                species_name=binomial,
                common_name=common_name,
                category=category,
                text=piece,
                source=source,
            ))

    return chunks


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(f"""
        CREATE TABLE species (
            species_id   TEXT PRIMARY KEY,
            species_name TEXT NOT NULL,
            common_name  TEXT,
            source       TEXT
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
            embedding float[{EMBED_DIM}]
        );
    """)
    conn.commit()
    return conn


def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk],
                  embeddings) -> None:
    """Insert chunks and their embeddings in a single transaction."""
    import struct

    # Species row (one per species, from the first chunk)
    species_seen: dict[str, Chunk] = {}
    for c in chunks:
        if c.species_id not in species_seen:
            species_seen[c.species_id] = c

    for c in species_seen.values():
        conn.execute(
            "INSERT OR IGNORE INTO species(species_id, species_name, common_name, source) "
            "VALUES (?, ?, ?, ?)",
            (c.species_id, c.species_name, c.common_name, c.source),
        )

    # Chunks + vectors
    for chunk, emb in zip(chunks, embeddings):
        cur = conn.execute(
            "INSERT INTO chunks(species_id, category, text) VALUES (?, ?, ?)",
            (chunk.species_id, chunk.category, chunk.text),
        )
        rowid = cur.lastrowid
        emb_bytes = struct.pack(f"{EMBED_DIM}f", *emb.tolist())
        conn.execute(
            "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
            (rowid, emb_bytes),
        )

    conn.commit()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(species_file: str, db_path: str) -> None:
    # Force UTF-8 on stdout/stderr so print() can't crash on Unicode
    # content (Windows default is cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7 or unusual stream

    species_list = json.loads(Path(species_file).read_text(encoding="utf-8"))
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    dump_file = db_file.with_name(db_file.stem + "_dump.json")
    report_file = db_file.with_name("build_report.txt")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    conn = init_db(db_file)
    report = BuildReport()
    dump_rows: list[dict] = []

    for species_id in tqdm(species_list, desc="Species"):
        if species_id in NON_SPECIES_CLASSES:
            report.skipped.append(species_id)
            continue

        try:
            chunks = build_chunks_for_species(species_id)
        except Exception as e:
            print(f"\n[ERROR] {species_id}: {e}")
            report.missing.append(species_id)
            continue

        if not chunks:
            report.missing.append(species_id)
            continue

        total_chars = sum(len(c.text) for c in chunks)
        if total_chars < 1000:
            report.short.append((species_id, total_chars))

        # Embed all chunks for this species in one batch
        texts = [c.text for c in chunks]
        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        insert_chunks(conn, chunks, embeddings)

        for c in chunks:
            dump_rows.append({
                "species_id": c.species_id,
                "species_name": c.species_name,
                "common_name": c.common_name,
                "category": c.category,
                "text": c.text,
            })

        report.ok.append(species_id)
        # Be polite to Wikipedia
        time.sleep(0.2)

    conn.close()
    dump_file.write_text(
        json.dumps(dump_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report.write(report_file)

    print(f"\nDone.")
    print(f"  DB:     {db_file}")
    print(f"  Dump:   {dump_file}  ({len(dump_rows)} chunks)")
    print(f"  Report: {report_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_corpus.py <species.json> <corpus.db>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])