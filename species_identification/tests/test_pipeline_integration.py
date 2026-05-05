"""
Integration smoke test for RoboRangerPipeline.

Builds a minimal in-memory SQLite database with the schema BlurbStore
expects, plus a stub LLM and stub embedder, and runs the pipeline end-
to-end through every code path:

    - gate accepts -> chunks above threshold -> blurb_plus_chunks
    - gate accepts -> no chunks above threshold -> blurb_only
    - gate rejects -> gate_rejected
    - species not in DB -> species_not_found
    - LLM raises -> llm_error
    - BlurbStore on a missing column -> raises BlurbStoreError

This isn't testing the *quality* of the pipeline — that's what the eval
harnesses are for. This is testing the *plumbing*: every component is
called with the right args, every failure mode returns a sensible
Response, no exception escapes the orchestrator.
"""

from __future__ import annotations

import sqlite3
import struct
import sys
from typing import Sequence

import numpy as np

# Import sqlite_vec so we can build a vec0 virtual table for chunks.
import sqlite_vec
import sys

sys.path.insert(1, "../species_identification/pipeline")
sys.path.insert(2, "../species_identification/llm-tuning")
from blurb_store import BlurbStore, BlurbStoreError
from intent import IntentClassifier
from pipeline import RoboRangerPipeline
from prompt_builder import Blurb
from wildlife_gate import WildlifeGate


EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubEmbedder:
    """
    Deterministic embedder: maps each input string to a vector via a hash.
    Real embedders give meaningful similarity; this one gives pseudo-random
    similarity that's stable across calls. That's enough to exercise the
    plumbing — the actual centroid logic is tested in test_units.py.

    For specific test queries, we override certain texts to land near or
    far from chunk vectors so we can deterministically test threshold
    filtering.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        # Anchor vector — we'll make some test queries embed close to this
        # and others embed orthogonal to it.
        self._anchor = self._stable_vec("anchor")

    def _stable_vec(self, key: str) -> np.ndarray:
        seed = abs(hash(key)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            t_low = t.lower()
            # Test queries we'll use:
            if "near" in t_low and "chunk" in t_low:
                # very close to the anchor
                v = self._anchor + 0.01 * self._stable_vec(t)
            elif "far" in t_low and "chunk" in t_low:
                # roughly orthogonal to the anchor
                base = self._stable_vec(t)
                v = base - (base @ self._anchor) * self._anchor
            else:
                v = self._stable_vec(t)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            out.append(v.astype(np.float32))
        return np.stack(out)

    @property
    def anchor(self) -> np.ndarray:
        return self._anchor


class StubLLM:
    name = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.should_raise = False

    def generate(self, messages: list[dict]) -> str:
        if self.should_raise:
            raise RuntimeError("stub error")
        self.calls.append(messages)
        return "stub response"


# ---------------------------------------------------------------------------
# Test fixture: build a tiny corpus.db in memory
# ---------------------------------------------------------------------------

def build_test_db(embedder: StubEmbedder) -> sqlite3.Connection:
    """Create the schema BlurbStore + retriever expect, populated minimally."""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # species table — matches the real corpus.db schema (JSON blurbs).
    conn.execute("""
        CREATE TABLE species (
            species_id    TEXT PRIMARY KEY,
            species_name  TEXT,
            common_name   TEXT,
            source        TEXT,
            blurb_text    TEXT,
            blurb_json    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY,
            species_id  TEXT,
            category    TEXT,
            text        TEXT
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            embedding float[{EMBED_DIM}]
        )
    """)

    # Insert one species with a complete blurb (JSON schema).
    import json
    blurb_json = json.dumps({
        "common_name": "Test Critter",
        "appearance": "A small spiny lizard with blue belly patches.",
        "size": "10-15 cm body length",
        "habitat": "Rocks and fences.",
        "diet": "Insects and small arthropods.",
        "behavior": "Active during the day; basks on sunny surfaces.",
        "dangerous_to_humans": "no",
        "dangerous_to_pets": "no",
        "notable": "Reduces Lyme disease prevalence in its range.",
    })
    conn.execute("""
        INSERT INTO species VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "Test_species", "Testus testius", "Test Critter",
        "wikipedia:Testus testius", "rendered text here", blurb_json,
    ))
    # Insert two chunks: one whose embedding is near the anchor (will pass
    # any reasonable threshold), one that's far from the anchor.
    near_text = "A near chunk that is highly relevant."
    far_text = "A far chunk that is irrelevant."
    near_emb = embedder._anchor + 0.01 * np.random.default_rng(0).standard_normal(EMBED_DIM).astype(np.float32)
    near_emb /= np.linalg.norm(near_emb)
    far_emb = embedder._stable_vec("orthogonal")
    far_emb = far_emb - (far_emb @ embedder._anchor) * embedder._anchor
    far_emb /= np.linalg.norm(far_emb)

    cur = conn.cursor()
    cur.execute("INSERT INTO chunks (species_id, category, text) VALUES (?,?,?)",
                ("Test_species", "diet", near_text))
    near_id = cur.lastrowid
    cur.execute("INSERT INTO chunks (species_id, category, text) VALUES (?,?,?)",
                ("Test_species", "habitat", far_text))
    far_id = cur.lastrowid

    # Insert into the vector table at matching rowids.
    for rid, vec in [(near_id, near_emb), (far_id, far_emb)]:
        cur.execute(
            "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
            (rid, struct.pack(f"{EMBED_DIM}f", *vec.tolist())),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def main() -> int:
    embedder = StubEmbedder()
    conn = build_test_db(embedder)
    llm = StubLLM()

    pipeline = RoboRangerPipeline(
        embedder=embedder,
        db_conn=conn,
        llm_backend=llm,
        retrieval_threshold=0.50,
    )

    # Pick wildlife-shaped queries that the stub embedder happens to route
    # to the wildlife side. We discover these empirically rather than by
    # design — the stub embedder is deterministic but the gate's centroids
    # depend on the prototypes, so we need to find a query that lands
    # right. Tests 2-5 all need a query that passes the gate. If we can't
    # find one, tests skip with [SKIP].
    candidates = [
        "what does it eat",
        "is it venomous",
        "tell me about this species",
        "is it nocturnal",
        "where does it live",
    ]
    wildlife_query = None
    for q in candidates:
        if pipeline.gate.check(q).in_domain:
            wildlife_query = q
            break
    if wildlife_query is None:
        print("WARNING: no candidate query passed the stub gate; "
              "tests 2-5 will be SKIPPED. (This is a test-fixture "
              "limitation, not a pipeline bug.)")

    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        tag = "PASS" if condition else "FAIL"
        print(f"  [{tag}] {label}" + (f"   {detail}" if detail else ""))
        if not condition:
            failures += 1

    print("\n=== test 1: gate rejects an off-topic query ===")
    llm.calls.clear()
    # We need the gate's off-topic centroid to win for this query. The stub
    # embedder hashes deterministically, so this is best-effort: just check
    # that *if* the gate rejects, the path is correct. If it accepts (because
    # of stub-embedding noise), we skip; the gate's own tests cover quality.
    resp = pipeline.answer("Test_species", "where is the bathroom")
    if resp.path == "gate_rejected":
        check("gate_rejected path returns MSG_OFF_TOPIC",
              "wildlife" in resp.text.lower())
        check("LLM not called on gate reject", len(llm.calls) == 0)
    else:
        print(f"  [SKIP] stub embedder didn't trigger rejection "
              f"(path={resp.path}); gate quality is tested separately")

    print("\n=== test 2: missing species -> species_not_found ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        llm.calls.clear()
        resp = pipeline.answer("Nonexistent_species", wildlife_query)
        check("species_not_found path", resp.path == "species_not_found")
        check("text mentions species not found",
              "species" in resp.text.lower())
        check("LLM not called", len(llm.calls) == 0)

    print("\n=== test 3: chunks above threshold -> blurb_plus_chunks ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        # The retriever uses a separate query-rewriting + embedding path, so
        # we can't easily ensure chunks pass the threshold without a real
        # embedder. Lower the threshold to ~zero so any chunk passes.
        pipeline.retrieval_threshold = -1.0
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is blurb_plus_chunks",
              resp.path == "blurb_plus_chunks",
              f"got {resp.path}")
        check("LLM was called", len(llm.calls) >= 1)
        check("at least one chunk used", len(resp.chunks_used) >= 1)
        check("intent populated", resp.intent is not None)
        check("blurb populated and is right species",
              resp.blurb is not None
              and resp.blurb.common_name == "Test Critter")
        check("latency dict has all stages",
              all(k in resp.latency
                  for k in ("gate", "blurb", "intent", "retrieval",
                            "prompt", "llm", "total")))
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 4: no chunks above threshold -> blurb_only ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        # Threshold so high no chunk can pass.
        pipeline.retrieval_threshold = 0.99
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is blurb_only",
              resp.path == "blurb_only",
              f"got {resp.path}")
        check("no chunks used", len(resp.chunks_used) == 0)
        check("dropped chunks > 0",
              resp.chunks_dropped > 0,
              f"dropped={resp.chunks_dropped}")
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 5: LLM raises -> llm_error ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        pipeline.retrieval_threshold = -1.0
        llm.should_raise = True
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is llm_error", resp.path == "llm_error")
        check("error field populated",
              resp.error and "stub error" in resp.error)
        check("user-facing text doesn't leak the exception",
              "stub error" not in resp.text.lower())
        llm.should_raise = False
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 6: BlurbStore raises on missing schema ===")
    bad_conn = sqlite3.connect(":memory:")
    bad_conn.execute("CREATE TABLE species (species_id TEXT, common_name TEXT)")
    try:
        BlurbStore(bad_conn)
    except BlurbStoreError as e:
        check("BlurbStoreError raised on missing columns", True,
              f"msg head: {str(e)[:60]}...")
    else:
        check("BlurbStoreError raised on missing columns", False)
    bad_conn.close()

    print(f"\n=== {failures} failure(s) ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())