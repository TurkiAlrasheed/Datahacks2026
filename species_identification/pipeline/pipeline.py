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
    [LLMBackend.generate | .stream] -> reject path: llm_error
        |
        v
    Response

The orchestrator does no I/O beyond what the components do. It does no
logging by default — every fact a caller might want to log is on the
returned Response. Caller decides what to print.

Answers always end on a complete sentence: max_tokens cuts generation off
mid-word, so a trailing fragment is dropped. A caller that wants to speak
while the model is still generating passes on_sentence= to answer(); each
finished sentence is handed over as soon as it is complete (the caller owns
the audio, so this file still does no audio I/O).

This file contains only orchestration logic. Component construction
(loading the embedder, opening the DB, picking the LLM backend) belongs
to the caller — see pipeline_factory.py for a thin "build everything
from a config" helper.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Protocol

import numpy as np

# Resolve sibling packages from this file's location, not the caller's CWD.
_SPECIES_DIR = Path(__file__).resolve().parents[1]
for _sub in ("tests", "llm-tuning", "pipeline", "."):
    _path = str((_SPECIES_DIR / _sub).resolve())
    if _path not in sys.path:
        sys.path.insert(1, _path)

from blurb_store import BlurbStore  # noqa: E402
from intent import Intent, IntentClassifier, IntentResult  # noqa: E402
from prompt_builder import Blurb, Chunk, build_messages  # noqa: E402
from test_corpus import l2_to_cosine, retrieve  # noqa: E402
from wildlife_gate import GateResult, WildlifeGate  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — visible at the top so they're easy to find and tune.
# ---------------------------------------------------------------------------

# Cosine similarity below this means "the chunk is not relevant enough,
# drop it." Tune from your eval_harness output. The harness emits a
# suggested threshold when retrieve and refuse cases separate cleanly;
# 0.55 is a reasonable starting default based on the harness bands.
DEFAULT_RETRIEVAL_THRESHOLD = 0.55
DIRECT_ROUTE_SCORE_THRESHOLD = 0.65  

# Phrasings that signal the user wants prose, not a field lookup. Even when
# the intent classifier is confident, "tell me about its X" almost always
# refers to a sub-topic the structured field doesn't cover (courtship,
# nesting, migration). Route these to retrieval regardless of confidence.
OPEN_ENDED_PREFIXES = ("tell me about", "tell me more about")

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
    "I'm not sure I understand, I am limited to questions about the wildlife. "
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
# Sentence boundaries
# ---------------------------------------------------------------------------

# Terminal punctuation plus any closing quote/bracket, followed by
# whitespace or the end of the text. "25.4" never matches (no whitespace
# after the period).
_SENTENCE_END = re.compile(r"""[.!?]+["'”’)\]]*(?=\s|$)""")
# The text already ends on a finished sentence.
_TRAILING_END = re.compile(r"""[.!?]["'”’)\]]*$""")
# Words whose period is not a sentence end. Single letters are handled
# separately: genus initials ("M. melodia") and "U.S." show up in answers.
_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "vs", "approx", "ca", "dr", "mr", "mrs", "ms", "st", "mt",
    "jr", "sr",
})


def _is_abbreviation(text: str, dot: int) -> bool:
    """True when the period at text[dot] belongs to an abbreviation."""
    parts = text[:dot].rsplit(None, 1)
    word = parts[-1].lstrip("\"'(“‘[") if parts else ""
    last = word.rsplit(".", 1)[-1]
    return (len(last) == 1 and last.isalpha()) or word.lower() in _ABBREVIATIONS


def _sentence_ends(text: str) -> Iterator[int]:
    """Offsets just past each sentence-ending punctuation run in text."""
    for m in _SENTENCE_END.finditer(text):
        if m.group()[0] != "." or not _is_abbreviation(text, m.start()):
            yield m.end()


def split_complete_sentences(text: str) -> tuple[list[str], str]:
    """
    Split off the sentences that are definitely finished, for streaming.
    Returns (sentences, rest). A boundary at the very end of `text` is left
    in `rest`, because more tokens may still be coming ("25." -> "25.4").
    """
    sentences: list[str] = []
    start = 0
    for end in _sentence_ends(text):
        if end == len(text):
            break
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    return sentences, text[start:]


def trim_incomplete_sentence(text: str) -> str:
    """
    Drop a trailing fragment the max_tokens cap cut off mid-sentence. Text
    with no complete sentence at all is returned whole: a fragment is
    better than silence.
    """
    text = text.strip()
    if _TRAILING_END.search(text):
        return text
    ends = list(_sentence_ends(text))
    return text[:ends[-1]] if ends else text


# ---------------------------------------------------------------------------
# Protocols (structural typing — any object with these methods works)
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """
    Anything with .name and .generate(messages) -> str. A backend may also
    have .stream(messages) -> Iterator[str] (text deltas); answer() uses it
    when the caller asks for sentences as they are generated.
    """
    name: str
    def generate(self, messages: list[dict], *,
                 max_tokens: int | None = None) -> str: ...


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
    # True when `text` was already delivered sentence by sentence through
    # answer(on_sentence=...), so the caller must not speak it again.
    streamed: bool = False

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

    def answer(
        self,
        species_id: str,
        query: str,
        *,
        on_llm_start: Callable[[], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
    ) -> Response:
        """
        Run the full pipeline for one (species_id, query) pair.

        on_llm_start: called once, just before the LLM request, and only on
            the paths that use the LLM (the slow ones). A voice front-end
            plays a "one moment" line from it.
        on_sentence: when given and the backend can stream, the answer is
            streamed and each complete sentence is passed here as soon as
            it is generated. Response.streamed then says the text has
            already been delivered. Every other path returns its text the
            usual way and leaves streamed False.
        """
        latency: dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. Wildlife gate. The gate and the intent classifier embed the
        # same raw query, so encode it once and hand the vector to both
        # (each encode is ~240 ms on the UNO Q). "gate" therefore includes
        # the encode and "intent" is just the centroid scoring.
        t = time.perf_counter()
        q_emb = None
        if query.strip() and (self._shares_embedder(self.gate)
                              or self._shares_embedder(self.intent_classifier)):
            q_emb = np.asarray(self.embedder.encode([query])[0],
                               dtype=np.float32)
        gate_result = (self.gate.check(query, q_emb=q_emb)
                       if self._shares_embedder(self.gate)
                       else self.gate.check(query))
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
        intent_result = (
            self.intent_classifier.classify(query, q_emb=q_emb)
            if self._shares_embedder(self.intent_classifier)
            else self.intent_classifier.classify(query))
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
        if self._should_direct_route(query, intent_result):
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
        # recoverable from the user's perspective. The answer is cut back to
        # its last complete sentence: max_tokens stops generation mid-word.
        if on_llm_start is not None:
            on_llm_start()
        stream = getattr(self.llm, "stream", None) if on_sentence else None
        spoken: list[str] = []   # sentences already passed to on_sentence
        error = None
        t = time.perf_counter()
        try:
            if stream is None:
                text = trim_incomplete_sentence(self.llm.generate(messages))
            else:
                text = self._stream_sentences(stream(messages), on_sentence,
                                              spoken, latency, t)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if not spoken:
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
                    error=error,
                    latency=latency,
                )
            # Part of the answer has already been spoken; that part is the
            # answer the visitor got.
            text = " ".join(spoken)
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
            error=error,
            latency=latency,
            streamed=bool(spoken),
        )

    def warmup_prompt(self, species_id: str
                      ) -> tuple[list[dict] | None, dict[str, float | str]]:
        """
        The cheap half of warmup: gate + intent embedding, the species'
        partition KNN, and prompt assembly. Returns (messages, timings), or
        (None, {"error": ...}) if the species isn't in the corpus.

        Split out from warmup() so a caller can run this on the thread that
        owns the sqlite connection and then push the LLM prefill onto
        another thread (full_roboranger_run.py overlaps it with speaking the
        identification).
        """
        timings: dict[str, float | str] = {}
        blurb = self.blurb_store.get(species_id)
        if blurb is None:
            return None, {"error": f"species {species_id!r} not in corpus"}
        query = f"what does the {self._name(blurb)} look like"

        t = time.perf_counter()
        self.gate.check(query)
        self.intent_classifier.classify(query)
        timings["embed"] = time.perf_counter() - t

        t = time.perf_counter()
        chunks, _ = self._retrieve_filtered(species_id, query)
        timings["retrieval"] = time.perf_counter() - t

        messages = build_messages(
            query=query,
            blurb=blurb,
            chunks=chunks,
            intent_result=IntentResult(Intent.DESCRIPTION, "high", 1.0, 1.0,
                                       {}),
        )
        return messages, timings

    def warm_llm(self, messages: list[dict]) -> str | None:
        """
        Push a prompt through the LLM with max_tokens=1. That pays the
        prompt-processing cost now and leaves llama.cpp's --cache-prompt
        holding the system prompt + SPECIES FACTS prefix, so the first real
        question skips it. Generating a full answer here would only waste
        tokens the visitor never hears.

        Returns an error string, or None on success — a model server that
        isn't up must be reported at startup, not crash it. Touches no
        sqlite, so it is safe to call from another thread.
        """
        try:
            self.llm.generate(messages, max_tokens=1)
        except Exception as e:
            return f"LLM backend: {type(e).__name__}: {e}"
        return None

    def warmup(self, species_id: str) -> dict[str, float | str]:
        """
        Run the stages the first real question will hit, so the visitor
        doesn't pay cold-start costs: gate + intent embedding, the species'
        partition KNN, and the LLM prefill for this species' prompt.

        A throwaway query through answer() doesn't do this: it is rejected at
        the gate or intent stage and never reaches retrieval or the LLM.

        Returns per-stage seconds. LLM failures are recorded under "error"
        instead of raised.
        """
        messages, timings = self.warmup_prompt(species_id)
        if messages is None:
            return timings
        t = time.perf_counter()
        error = self.warm_llm(messages)
        timings["llm"] = time.perf_counter() - t
        if error:
            timings["error"] = error
        return timings

    # -- internal helpers ---------------------------------------------------

    def _shares_embedder(self, component) -> bool:
        """True when a gate/classifier encodes with this pipeline's embedder,
        so a query vector from self.embedder can be handed to it. Test stubs
        and components built on another model don't qualify."""
        return getattr(component, "embedder", None) is self.embedder

    @staticmethod
    def _stream_sentences(deltas: Iterator[str],
                          on_sentence: Callable[[str], None],
                          spoken: list[str],
                          latency: dict[str, float],
                          t_start: float) -> str:
        """
        Accumulate streamed text deltas and pass each complete sentence to
        on_sentence as soon as it is finished, appending it to `spoken` so
        the caller knows what was delivered if the stream fails midway.
        At the end, a trailing fragment cut off by max_tokens is dropped
        (same rule as trim_incomplete_sentence). Returns the full answer.
        """
        buf = ""
        for delta in deltas:
            buf += delta
            sentences, buf = split_complete_sentences(buf)
            for sentence in sentences:
                if not spoken:
                    latency["llm_first_sentence"] = time.perf_counter() - t_start
                spoken.append(sentence)
                on_sentence(sentence)
        tail = buf.strip()
        if tail and (_TRAILING_END.search(tail) or not spoken):
            if not spoken:
                latency["llm_first_sentence"] = time.perf_counter() - t_start
            spoken.append(tail)
            on_sentence(tail)
        return " ".join(spoken)

    def _retrieve_filtered(
        self, species_id: str, query: str,
    ) -> tuple[list[Chunk], int]:
        """
        Call the corpus retriever and convert its (category, text, l2) tuples
        into prompt_builder.Chunk objects, dropping anything below the
        cosine-similarity threshold.

        Returns (kept_chunks, n_dropped).
        """
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
    
    def _should_direct_route(self, query: str, intent_result: IntentResult) -> bool:
        """
        Direct-route is appropriate when:
        - the query isn't open-ended ('tell me about its X' wants retrieval)
        - the intent maps to a structured field we can format
        - the classifier is confident enough:
            * high confidence: always
            * medium confidence: only if the cosine score clears the floor
        """
        if intent_result.intent not in DIRECT_ROUTE_INTENTS:
            return False

        q_lower = query.lower().strip()
        if any(p in q_lower for p in OPEN_ENDED_PREFIXES):
            return False

        if intent_result.confidence == "high":
            return True
        if (
            intent_result.confidence == "medium"
            and intent_result.score >= DIRECT_ROUTE_SCORE_THRESHOLD
        ):
            return True
        return False
    
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
    # ----- direct-routing formatters ---------------------------------------
    #
    # Each formatter takes a Blurb and returns either:
    #   - a complete sentence answer (the field-direct path)
    #   - None (caller falls through to retrieval + LLM)
    #
    # Style: terse, factual, ranger voice — "You'll find X..." / "It eats Y."
    # No "I think" hedges, no apologies, no padding. A blurb field is the
    # source of truth; we're just wrapping it in a sentence shape.
    #
    # Capitalization in source fields is inconsistent (some sentence-cap,
    # most lowercase). _decap() lowercases the first character so phrases
    # paste cleanly into mid-sentence; values that appear at the start of
    # a sentence get manually capitalized in the template. It also drops
    # trailing periods, since every template supplies its own punctuation
    # (otherwise "Insects and small arthropods." renders as "arthropods..").

    @staticmethod
    def _decap(s: str) -> str:
        """Lowercase first character and drop trailing periods so a field
        reads naturally mid-sentence."""
        s = s.rstrip(" .") if s else s
        return s[:1].lower() + s[1:] if s else s

    @staticmethod
    def _name(blurb: Blurb) -> str:
        """Display name for templates. Falls back to scientific name."""
        return blurb.common_name or blurb.species_id.replace("_", " ")

    def _format_danger(self, blurb: Blurb) -> str | None:
        """
        Combine human and pet danger into one sentence. Going field-direct
        means we have to handle the four common combinations explicitly —
        "no humans, yes pets" is a real and important case (e.g. owls,
        hawks) that a single-field answer would miss.
        """
        h = blurb.dangerous_to_humans  # "no" | "mild" | "yes"
        p = blurb.dangerous_to_pets
        name = self._name(blurb)

        # Most common case: harmless to both.
        if h == "no" and p == "no":
            return f"The {name} is not dangerous to humans or pets."

        # Dangerous to one but not the other — the case worth being precise
        # about. A user asking "is it venomous" who has a dog wants this.
        if h == "no" and p == "yes":
            return (f"The {name} is not dangerous to humans, but it can be "
                    f"a threat to small pets.")
        if h == "yes" and p == "no":
            return (f"The {name} can be dangerous to humans. It's not known "
                    f"to threaten pets.")

        # Mild to both — bees, etc.
        if h == "mild" and p == "mild":
            return (f"The {name} can sting or bite if provoked, but it's not "
                    f"seriously dangerous to humans or pets.")

        # Anything with "yes" in it: lead with the strongest warning.
        if h == "yes" or p == "yes":
            who = []
            if h == "yes": who.append("humans")
            if p == "yes": who.append("pets")
            if h == "mild" and "humans" not in who: who.insert(0, "humans (mildly)")
            if p == "mild" and "pets" not in who: who.append("pets (mildly)")
            return f"The {name} can be dangerous to {' and '.join(who)}. Keep your distance."

        # Mild to one, no to the other.
        if h == "mild":
            return (f"The {name} can cause mild harm to humans if provoked. "
                    f"It's not dangerous to pets.")
        if p == "mild":
            return (f"The {name} is not dangerous to humans. It may cause "
                    f"mild harm to pets if they get too close.")

        # Should be unreachable if blurb data is clean. Fall through to LLM.
        return None

    def _format_diet(self, blurb: Blurb) -> str | None:
        if blurb.prose_reviewed and blurb.diet_prose:
            return blurb.diet_prose

        # Missing fields fall through to retrieval + LLM rather than
        # crashing (None.strip()) or rendering "eats ." to the visitor.
        diet = (blurb.diet or "").strip()
        if not diet:
            return None
        name = self._name(blurb)

        if "photosynthesis" in diet.lower():
            return (f"The {name} doesn't eat in the way an animal does — "
                    f"it makes its own food from sunlight, water, and "
                    f"nutrients in the soil.")

        # Animal diet: blurb field is a comma-separated list of foods.
        return f"The {name} eats {self._decap(diet)}."

    def _format_habitat(self, blurb: Blurb) -> str | None:
        habitat = (blurb.habitat or "").strip()
        if not habitat:
            return None
        name = self._name(blurb)
        return f"You'll find the {name} in {self._decap(habitat)}."

    def _format_size(self, blurb: Blurb) -> str | None:
        if blurb.prose_reviewed and blurb.size_prose:
            return blurb.size_prose
        # Fallback while prose isn't reviewed yet.
        size = (blurb.size or "").strip()
        if not size:
            return None
        return f"The {self._name(blurb)} typically measures {self._decap(size)}."

    def _format_behavior(self, blurb: Blurb) -> str | None:
        if blurb.prose_reviewed and blurb.behavior_prose:
            return blurb.behavior_prose
        behavior = (blurb.behavior or "").strip()
        if not behavior:
            return None
        return f"The {self._name(blurb)} is {self._decap(behavior)}."

    def _format_description(self, blurb: Blurb) -> str | None:
        # Description synthesizes appearance + size + notable. Prefer prose
        # for each piece when available; mix-and-match is fine because each
        # prose sentence is self-contained.
        if blurb.prose_reviewed:
            parts = []
            if blurb.appearance_prose:
                parts.append(blurb.appearance_prose)
            if blurb.size_prose:
                parts.append(blurb.size_prose)
            if blurb.notable_prose:
                parts.append(blurb.notable_prose)
            if parts:
                return " ".join(parts)
        # Fallback: structural template across the three fields.
        appearance = (blurb.appearance or "").strip()
        size = (blurb.size or "").strip()
        notable = (blurb.notable or "").strip()
        if not appearance:
            return None
        first = f"The {self._name(blurb)} has {self._decap(appearance)}"
        if size:
            parts = [f"{first} and typically measures {self._decap(size)}."]
        else:
            parts = [f"{first}."]
        if notable:
            parts.append(f"Notably, {self._decap(notable)}.")
        return " ".join(parts)