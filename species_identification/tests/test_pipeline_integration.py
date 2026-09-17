"""
Integration smoke test for RoboRangerPipeline.

Builds a minimal in-memory corpus with the real schema (corpus_schema.py),
plus a stub LLM and stub embedder, and runs the pipeline end-to-end through
every code path:

    - gate rejects -> gate_rejected
    - species not in DB -> species_not_found
    - gate accepts -> chunks above threshold -> blurb_plus_chunks
    - gate accepts -> no chunks above threshold -> blurb_only
    - LLM raises -> llm_error
    - gate accepts but intent OTHER -> intent_unclear
    - BlurbStore on a missing column -> raises BlurbStoreError
    - high-confidence field intent -> blurb_direct (no retrieval, no LLM)
    - field missing from the blurb -> falls through to the LLM, no crash
    - retrieval is scoped to the species and never returns the blurb chunk
    - a v1 (unpartitioned) corpus is refused by check_corpus_schema
    - warmup() reaches retrieval + LLM and reports a failing LLM

This isn't testing the *quality* of the pipeline — that's what the eval
harnesses are for. This is testing the *plumbing*: every component is
called with the right args, every failure mode returns a sensible
Response, no exception escapes the orchestrator.

Run from anywhere:
    python species_identification/tests/test_pipeline_integration.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import sqlite_vec

_SPECIES_DIR = Path(__file__).resolve().parents[1]
for _sub in ("tests", "llm-tuning", "pipeline", "."):
    _path = str((_SPECIES_DIR / _sub).resolve())
    if _path not in sys.path:
        sys.path.insert(1, _path)

from blurb_store import BlurbStore, BlurbStoreError  # noqa: E402
from corpus_schema import (  # noqa: E402
    EMBED_DIM,
    CorpusSchemaError,
    check_corpus_schema,
    create_schema,
    insert_vector,
    pack_embedding,
    refresh_manifest,
)
from intent import Intent, IntentResult  # noqa: E402
from pipeline import RoboRangerPipeline  # noqa: E402
from test_corpus import retrieve  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubEmbedder:
    """
    Deterministic embedder: maps each input string to a vector via sha256.
    (Python's hash() is randomized per process, which made this fixture's
    gate/intent outcomes change between runs.) Real embedders give
    meaningful similarity; this one gives pseudo-random similarity that's
    stable across calls and processes. That's enough to exercise the
    plumbing.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self._anchor = self._stable_vec("anchor")

    def _stable_vec(self, key: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                              "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            v = self._stable_vec(t)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            out.append(v.astype(np.float32))
        return np.stack(out)

    def near_anchor(self, key: str) -> np.ndarray:
        v = self._anchor + 0.01 * self._stable_vec(key)
        return v / np.linalg.norm(v)

    def far_from_anchor(self, key: str) -> np.ndarray:
        v = self._stable_vec(key)
        v = v - (v @ self._anchor) * self._anchor
        return v / np.linalg.norm(v)


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


class FixedClassifier:
    """Intent classifier stub that always returns one result."""

    def __init__(self, intent: Intent, confidence: str = "high",
                 score: float = 0.85) -> None:
        self.result = IntentResult(intent, confidence, score, 0.10, {})

    def classify(self, q):
        return self.result


# ---------------------------------------------------------------------------
# Test fixture: build a tiny v2 corpus in memory
# ---------------------------------------------------------------------------

FULL_BLURB = {
    "common_name": "Test Critter",
    "appearance": "A small spiny lizard with blue belly patches.",
    "size": "10-15 cm body length",
    "habitat": "Rocks and fences.",
    "diet": "Insects and small arthropods.",
    "behavior": "Active during the day; basks on sunny surfaces.",
    "dangerous_to_humans": "no",
    "dangerous_to_pets": "no",
    "notable": "Reduces Lyme disease prevalence in its range.",
}
# Missing diet / habitat / size: direct-route formatters must fall through.
SPARSE_BLURB = {
    "common_name": "Sparse Critter",
    "appearance": "A mostly undocumented beetle.",
    "dangerous_to_humans": "no",
    "dangerous_to_pets": "no",
}

NEAR_TEXT = "A near chunk that is highly relevant."
FAR_TEXT = "A far chunk that is irrelevant."
BLURB_TEXT = "Species: Testus testius\nDiet: Insects"
OTHER_TEXT = "Another species' chunk that sits right on the anchor."


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def build_test_db(embedder: StubEmbedder) -> sqlite3.Connection:
    """Create the production schema, populated minimally."""
    conn = _connect()
    create_schema(conn)

    species = [
        ("Test_species", "Testus testius", "Test Critter", FULL_BLURB),
        ("Sparse_species", "Sparsus sparsus", "", SPARSE_BLURB),
        ("Other_species", "Alius alius", "Other Critter", FULL_BLURB),
    ]
    for sid, name, common, blurb in species:
        conn.execute(
            "INSERT INTO species(species_id, species_name, common_name, "
            "source, blurb_text, blurb_json) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, name, common, f"wikipedia:{name}", "rendered text",
             json.dumps(blurb)))

    # Test_species: one near chunk, one far chunk, and a blurb chunk placed
    # right on the anchor (retrieval must still exclude it). Other_species
    # and Sparse_species also get near-anchor chunks (retrieval for
    # Test_species must never return them).
    rows = [
        ("Test_species", "diet", NEAR_TEXT, embedder.near_anchor("near")),
        ("Test_species", "habitat", FAR_TEXT, embedder.far_from_anchor("far")),
        ("Test_species", "blurb", BLURB_TEXT, embedder.near_anchor("blurb")),
        ("Other_species", "diet", OTHER_TEXT, embedder.near_anchor("other")),
        ("Sparse_species", "general", "Sparse beetle text.",
         embedder.near_anchor("sparse")),
    ]
    for sid, cat, text, vec in rows:
        cur = conn.execute(
            "INSERT INTO chunks(species_id, category, text) VALUES (?, ?, ?)",
            (sid, cat, text))
        insert_vector(conn, cur.lastrowid, sid, cat, pack_embedding(vec))
    conn.commit()
    refresh_manifest(conn)
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

    # Pick a wildlife-shaped query that the stub embedder happens to route
    # to the wildlife side. We discover these empirically rather than by
    # design — the stub embedder is deterministic but the gate's centroids
    # depend on the prototypes, so we need to find a query that lands
    # right. Tests that need one skip with [SKIP] if none is found.
    candidates = [
        "what does it eat",
        "is it venomous",
        "tell me about this species",
        "is it nocturnal",
        "where does it live",
        "how can I identify it",
        "is it endangered",
        "what does it look like",
    ]
    wildlife_query = next(
        (q for q in candidates if pipeline.gate.check(q).in_domain), None)
    if wildlife_query is None:
        print("WARNING: no candidate query passed the stub gate; "
              "gate-dependent tests will be SKIPPED. (This is a test-fixture "
              "limitation, not a pipeline bug.)")

    real_classifier = pipeline.intent_classifier
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        tag = "PASS" if condition else "FAIL"
        print(f"  [{tag}] {label}" + (f"   {detail}" if detail else ""))
        if not condition:
            failures += 1

    print("\n=== test 1: gate rejects an off-topic query ===")
    llm.calls.clear()
    # Best-effort with a stub embedder: check the path only if it rejects.
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

    # Tests 3-5 exercise the retrieval + LLM path, so the stub intent must
    # NOT be direct-routable (DIET/DANGER/SIZE/HABITAT/BEHAVIOR/DESCRIPTION
    # at high confidence are answered from the blurb with no LLM).
    pipeline.intent_classifier = FixedClassifier(Intent.IDENTIFICATION)

    print("\n=== test 3: chunks above threshold -> blurb_plus_chunks ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        llm.calls.clear()
        pipeline.retrieval_threshold = -1.0
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is blurb_plus_chunks",
              resp.path == "blurb_plus_chunks", f"got {resp.path}")
        check("LLM was called", len(llm.calls) >= 1)
        check("at least one chunk used", len(resp.chunks_used) >= 1)
        texts = {c.text for c in resp.chunks_used}
        check("only this species' chunks retrieved",
              texts <= {NEAR_TEXT, FAR_TEXT}, f"got {sorted(texts)}")
        check("blurb chunk not retrieved (already in SPECIES FACTS)",
              BLURB_TEXT not in texts)
        check("intent populated", resp.intent is not None)
        check("blurb populated and is right species",
              resp.blurb is not None
              and resp.blurb.common_name == "Test Critter")
        check("latency dict has all stages",
              all(k in resp.latency
                  for k in ("gate", "blurb", "intent", "retrieval",
                            "prompt", "llm", "total")),
              f"got {sorted(resp.latency)}")
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 4: no chunks above threshold -> blurb_only ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        pipeline.retrieval_threshold = 0.99
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is blurb_only", resp.path == "blurb_only",
              f"got {resp.path}")
        check("no chunks used", len(resp.chunks_used) == 0)
        check("dropped chunks > 0", resp.chunks_dropped > 0,
              f"dropped={resp.chunks_dropped}")
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 5: LLM raises -> llm_error ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        pipeline.retrieval_threshold = -1.0
        llm.should_raise = True
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is llm_error", resp.path == "llm_error",
              f"got {resp.path}")
        check("error field populated",
              bool(resp.error) and "stub error" in resp.error)
        check("user-facing text doesn't leak the exception",
              "stub error" not in resp.text.lower())
        llm.should_raise = False
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 5b: gate accepts but intent OTHER -> intent_unclear ===")
    pipeline.intent_classifier = FixedClassifier(Intent.OTHER, "low", 0.4)
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        llm.calls.clear()
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is intent_unclear", resp.path == "intent_unclear",
              f"got {resp.path}")
        check("LLM not called when intent unclear", len(llm.calls) == 0)
        check("user-facing text guides them",
              "eats" in resp.text.lower() or "lives" in resp.text.lower())
    pipeline.intent_classifier = real_classifier

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

    print("\n=== test 7: high-confidence field intent -> blurb_direct ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        pipeline.intent_classifier = FixedClassifier(Intent.DIET)
        llm.calls.clear()
        resp = pipeline.answer("Test_species", wildlife_query)
        check("path is blurb_direct", resp.path == "blurb_direct",
              f"got {resp.path}")
        check("answer comes from the diet field",
              "insects" in resp.text.lower(), resp.text)
        check("LLM not called on direct route", len(llm.calls) == 0)
        check("retrieval skipped on direct route",
              "retrieval" not in resp.latency)
        pipeline.intent_classifier = real_classifier

    print("\n=== test 8: missing blurb field -> falls through, no crash ===")
    if wildlife_query is None:
        print("  [SKIP]")
    else:
        pipeline.retrieval_threshold = -1.0
        for intent in (Intent.DIET, Intent.HABITAT, Intent.SIZE):
            pipeline.intent_classifier = FixedClassifier(intent)
            llm.calls.clear()
            try:
                resp = pipeline.answer("Sparse_species", wildlife_query)
            except Exception as e:  # the old formatters raised AttributeError
                check(f"{intent.name}: no exception", False,
                      f"{type(e).__name__}: {e}")
                continue
            check(f"{intent.name}: fell through to the LLM path",
                  resp.path in ("blurb_plus_chunks", "blurb_only"),
                  f"got {resp.path}")
            check(f"{intent.name}: no 'None' rendered",
                  "none" not in resp.text.lower(), resp.text)
        pipeline.intent_classifier = real_classifier
        pipeline.retrieval_threshold = 0.50

    print("\n=== test 9: retrieval is partition-scoped and excludes blurb ===")
    rows = retrieve(conn, embedder, "Test_species", "anything", k=10)
    texts = [text for _, text, _ in rows]
    check("returns every non-blurb chunk of the species (k > chunks)",
          sorted(texts) == sorted([NEAR_TEXT, FAR_TEXT]), f"got {texts}")
    check("no category='blurb' rows", all(cat != "blurb" for cat, _, _ in rows))

    print("\n=== test 10: v1 corpus is refused ===")
    v1 = _connect()
    v1.execute(f"CREATE VIRTUAL TABLE chunk_vectors USING vec0("
               f"embedding float[{EMBED_DIM}])")
    try:
        check_corpus_schema(v1)
    except CorpusSchemaError as e:
        check("CorpusSchemaError on v1 schema", True,
              f"msg head: {str(e)[:50]}...")
    else:
        check("CorpusSchemaError on v1 schema", False)
    v1.close()
    try:
        meta = check_corpus_schema(conn)
        check("v2 fixture passes schema check with manifest",
              meta.get("species_count") == "3", str(meta))
    except CorpusSchemaError as e:
        check("v2 fixture passes schema check", False, str(e))

    print("\n=== test 11: warmup reaches retrieval + LLM ===")
    llm.calls.clear()
    timings = pipeline.warmup("Test_species")
    check("warmup ran retrieval and LLM",
          "retrieval" in timings and "llm" in timings and len(llm.calls) == 1,
          str(timings))
    check("no error reported", "error" not in timings, str(timings))
    llm.should_raise = True
    timings = pipeline.warmup("Test_species")
    check("LLM failure reported, not raised", "error" in timings,
          str(timings))
    llm.should_raise = False

    print(f"\n=== {failures} failure(s) ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
