"""
RoboRanger pipeline orchestrator.

Wires the components into a single end-to-end query path:

    query, species_id
        |
        v
    [WildlifeGate]      -> reject path: gate_rejected
        |
        v
    [BlurbStore.get]    -> reject path: species_not_found
        |
        v
    [IntentClassifier]
        |
        v
    [retrieve()]        -> filter by retrieval_threshold (cosine sim)
        |
        v
    [build_messages]
        |
        v
    [LLMBackend.generate] -> reject path: llm_error
        |
        v
    Response

The orchestrator does no I/O beyond what the components do. It does no
logging by default — every fact a caller might want to log is on the
returned Response. Caller decides what to print.

This file contains only orchestration logic. Component construction
(loading the embedder, opening the DB, picking the LLM backend) belongs
to the caller — see pipeline_factory.py for a thin "build everything
from a config" helper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Protocol
import sys

sys.path.insert(1, "../species_identification/llm-tuning")
sys.path.insert(2, "../species_identification/tests")
from blurb_store import BlurbStore
from intent import Intent, IntentClassifier, IntentResult
from prompt_builder import Blurb, Chunk, build_messages
from wildlife_gate import GateResult, WildlifeGate

# `test_corpus` imports sentence_transformers at module level, which is a
# heavy dep we don't actually need in this file. Import the two pure
# functions we use lazily so importing pipeline.py is cheap.
def _retrieve_fns():
    from test_corpus import retrieve, l2_to_cosine
    return retrieve, l2_to_cosine


# ---------------------------------------------------------------------------
# Constants — visible at the top so they're easy to find and tune.
# ---------------------------------------------------------------------------

# Cosine similarity below this means "the chunk is not relevant enough,
# drop it." Tune from your eval_harness output. The harness emits a
# suggested threshold when retrieve and refuse cases separate cleanly;
# 0.55 is a reasonable starting default based on the harness bands.
DEFAULT_RETRIEVAL_THRESHOLD = 0.55

# How many chunks to ask the retriever for. We may keep fewer after the
# threshold filter, and the prompt builder caps further at MAX_CHUNKS.
RETRIEVAL_K = 5

# User-facing fallback messages. Kept short and visitor-friendly. These
# are what the user sees when the pipeline can't or won't call the LLM.
MSG_OFF_TOPIC = (
    "I can only help with questions about the wildlife you're looking at."
)
MSG_SPECIES_NOT_FOUND = (
    "I don't have information about this species yet."
)
# Used when the wildlife gate accepted the query but the intent classifier
# couldn't confidently route it. This catches edge cases the gate's binary
# decision misses — e.g. "tell me a joke about it", "what should I name
# it", "how rare is this in movies" — which are wildlife-adjacent enough
# to pass the gate but not real field-guide questions.
MSG_INTENT_UNCLEAR = (
    "I'm not sure I understand, I am limited to questions about the wildlife."
    "Try asking what it eats, where it lives, "
    "whether it's dangerous, or something along those lines."
)
MSG_LLM_ERROR = (
    "Sorry, I had trouble answering that. Please try again."
)

# Intents whose answer can be served verbatim from a blurb field, no LLM
# needed. 
DIRECT_ROUTE_INTENTS = frozenset({
    Intent.DANGER,
    Intent.DIET,
    Intent.SIZE,
    Intent.HABITAT,
    Intent.BEHAVIOR,
    Intent.DESCRIPTION,   # -> appearance + notable
})

# ---------------------------------------------------------------------------
# Protocols (structural typing — any object with these methods works)
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """Anything with .name and .generate(messages) -> str."""
    name: str
    def generate(self, messages: list[dict]) -> str: ...


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

ResponsePath = Literal[
    "gate_rejected",
    "species_not_found",
    "intent_unclear",
    "blurb_direct",
    "blurb_only",
    "blurb_plus_chunks",
    "llm_error",
]


@dataclass
class Response:
    """Everything the caller might want to log or display."""
    text: str
    path: ResponsePath
    species_id: str
    query: str
    # Optional diagnostic fields — None when the corresponding stage was
    # skipped (e.g. intent is None when the gate rejected the query).
    gate: GateResult | None = None
    intent: IntentResult | None = None
    blurb: Blurb | None = None
    chunks_used: list[Chunk] = field(default_factory=list)
    chunks_dropped: int = 0      # how many were filtered by threshold
    error: str | None = None
    latency: dict[str, float] = field(default_factory=dict)

    @property
    def used_llm(self) -> bool:
        return self.path in ("blurb_only", "blurb_plus_chunks")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RoboRangerPipeline:
    """
    Orchestrates the full query path. Stateless beyond the components it
    holds — safe to call .answer() concurrently from multiple threads as
    long as the underlying components are thread-safe (sqlite-vec and
    SentenceTransformer both are for read-only use).
    """

    def __init__(
        self,
        *,
        embedder,                       # SentenceTransformer-like
        db_conn,                        # sqlite3.Connection (sqlite-vec loaded)
        llm_backend: LLMBackend,
        gate: WildlifeGate | None = None,
        intent_classifier: IntentClassifier | None = None,
        blurb_store: BlurbStore | None = None,
        retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
        retrieval_k: int = RETRIEVAL_K,
    ) -> None:
        self.embedder = embedder
        self.db_conn = db_conn
        self.llm = llm_backend
        # Build the embedding-based components if not provided. This lets
        # callers either pass pre-built components (for testing or sharing
        # centroids across pipelines) or let the pipeline build them.
        self.gate = gate or WildlifeGate(embedder)
        self.intent_classifier = (
            intent_classifier or IntentClassifier(embedder)
        )
        self.blurb_store = blurb_store or BlurbStore(db_conn)
        self.retrieval_threshold = retrieval_threshold
        self.retrieval_k = retrieval_k

    # -- the main entry point ------------------------------------------------

    def answer(self, species_id: str, query: str) -> Response:
        """Run the full pipeline for one (species_id, query) pair."""
        latency: dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. Wildlife gate
        t = time.perf_counter()
        gate_result = self.gate.check(query)
        print(gate_result)
        latency["gate"] = time.perf_counter() - t
        if gate_result.reject:
            latency["total"] = time.perf_counter() - t0
            return Response(
                text=MSG_OFF_TOPIC,
                path="gate_rejected",
                species_id=species_id,
                query=query,
                gate=gate_result,
                latency=latency,
            )

        # 2. Blurb lookup. If the species isn't in the DB, refuse cleanly —
        # the alternative would be calling the LLM with no grounding, which
        # is exactly what we don't want.
        t = time.perf_counter()
        blurb = self.blurb_store.get(species_id)
        latency["blurb"] = time.perf_counter() - t
        if blurb is None:
            latency["total"] = time.perf_counter() - t0
            return Response(
                text=MSG_SPECIES_NOT_FOUND,
                path="species_not_found",
                species_id=species_id,
                query=query,
                gate=gate_result,
                latency=latency,
            )

        # 3. Intent classification. Drives prompt template + which blurb
        # field leads. Also acts as a second-line filter: if the intent
        # classifier returns OTHER or only low confidence, the question
        # probably isn't a real field-guide query (the gate is binary and
        # misses some edge cases). Short-circuit to a fixed message rather
        # than letting the LLM fabricate an answer.
        t = time.perf_counter()
        intent_result = self.intent_classifier.classify(query)
        latency["intent"] = time.perf_counter() - t

        if (
            intent_result.intent == Intent.OTHER
            or intent_result.confidence == "low"
        ):
            latency["total"] = time.perf_counter() - t0
            return Response(
                text=MSG_INTENT_UNCLEAR,
                path="intent_unclear",
                species_id=species_id,
                query=query,
                gate=gate_result,
                intent=intent_result,
                blurb=blurb,
                latency=latency,
            )
        
        # 3a. Direct-field routing. If the intent is high-confidence and maps
        # to a blurb field, format the field directly and skip retrieval +
        # LLM entirely. Sub-millisecond instead of 12+ seconds. Falls
        # through to the LLM path when the formatter returns None (missing
        # field, or formatter decides the question needs synthesis).
        if (
            intent_result.confidence == "high"
            and intent_result.intent in DIRECT_ROUTE_INTENTS
        ):
            t = time.perf_counter()
            direct_text = self._format_from_blurb(intent_result.intent, blurb)
            latency["format"] = time.perf_counter() - t
            if direct_text is not None:
                latency["total"] = time.perf_counter() - t0
                return Response(
                    text=direct_text,
                    path="blurb_direct",
                    species_id=species_id,
                    query=query,
                    gate=gate_result,
                    intent=intent_result,
                    blurb=blurb,
                    latency=latency,
                )

        # 4. Retrieval + threshold filter.
        t = time.perf_counter()
        chunks, dropped = self._retrieve_filtered(species_id, query)
        latency["retrieval"] = time.perf_counter() - t

        if not chunks and intent_result.confidence != "high":
            latency["total"] = time.perf_counter() - t0
            return Response(
                text=MSG_INTENT_UNCLEAR,
                path="intent_unclear",
                species_id=species_id,
                query=query,
                gate=gate_result,
                intent=intent_result,
                blurb=blurb,
                latency=latency,
            )

        # 5. Prompt assembly.
        t = time.perf_counter()
        messages = build_messages(
            query=query,
            blurb=blurb,
            chunks=chunks,
            intent_result=intent_result,
        )
        latency["prompt"] = time.perf_counter() - t

        # 6. LLM call. Failures here should be visible (logged) but
        # recoverable from the user's perspective.
        t = time.perf_counter()
        try:
            text = self.llm.generate(messages).strip()
        except Exception as e:
            latency["llm"] = time.perf_counter() - t
            latency["total"] = time.perf_counter() - t0
            return Response(
                text=MSG_LLM_ERROR,
                path="llm_error",
                species_id=species_id,
                query=query,
                gate=gate_result,
                intent=intent_result,
                blurb=blurb,
                chunks_used=chunks,
                chunks_dropped=dropped,
                error=f"{type(e).__name__}: {e}",
                latency=latency,
            )
        latency["llm"] = time.perf_counter() - t
        latency["total"] = time.perf_counter() - t0

        path: ResponsePath = "blurb_plus_chunks" if chunks else "blurb_only"
        return Response(
            text=text,
            path=path,
            species_id=species_id,
            query=query,
            gate=gate_result,
            intent=intent_result,
            blurb=blurb,
            chunks_used=chunks,
            chunks_dropped=dropped,
            latency=latency,
        )

    # -- internal helpers ---------------------------------------------------

    def _retrieve_filtered(
        self, species_id: str, query: str,
    ) -> tuple[list[Chunk], int]:
        """
        Call the corpus retriever and convert its (category, text, l2) tuples
        into prompt_builder.Chunk objects, dropping anything below the
        cosine-similarity threshold.

        Returns (kept_chunks, n_dropped).
        """
        retrieve, l2_to_cosine = _retrieve_fns()
        raw = retrieve(
            self.db_conn, self.embedder, species_id, query, k=self.retrieval_k,
        )
        kept: list[Chunk] = []
        dropped = 0
        for category, text, l2_dist in raw:
            cosine = l2_to_cosine(l2_dist)
            if cosine < self.retrieval_threshold:
                dropped += 1
                continue
            kept.append(Chunk(
                text=text,
                score=cosine,
                source=category or "",
            ))
        return kept, dropped
    
    def _format_from_blurb(self, intent: Intent, blurb: Blurb) -> str | None:
        """
        Return a templated answer from blurb fields, or None to fall through
        to the LLM. Each formatter decides for itself whether the blurb has
        enough information to answer directly. Keep these formatters dumb —
        no synthesis, no inference, just field lookup + sentence templating.
        """
        # Stubs for now. Real implementations come next; returning None
        # everywhere means this routing change is a no-op for output but
        # exercises the new code path.
        formatters = {
            Intent.DANGER:      self._format_danger,
            Intent.DIET:        self._format_diet,
            Intent.SIZE:        self._format_size,
            Intent.HABITAT:     self._format_habitat,
            Intent.BEHAVIOR:    self._format_behavior,
            Intent.DESCRIPTION: self._format_description,
        }
        fn = formatters.get(intent)
        return fn(blurb) if fn else None

    # Stubs — return None so everything falls through to the LLM. Replace
    # one at a time and watch the path counts shift in --profile output.
    def _format_danger(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct danger"

    def _format_diet(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct diet"
    
    def _format_size(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct size"

    def _format_habitat(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct habitat"

    def _format_behavior(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct behavior"

    def _format_description(self, blurb: Blurb) -> str | None:
        return "DEBUG: routed direct description"