"""
Build a fully-wired RoboRangerPipeline from a config.

Single entry point so the CLI tool, eval harness, and Uno Q runtime all
construct the pipeline the same way. If you ever swap the embedder model
or the LLM backend, change it here, not in three places.

Two-stage construction, so a caller can hide the slow part behind other
work (full_roboranger_run.py loads these while the visitor is aiming the
camera):

    prewarm_components(...)   slow, no sqlite: the sentence-transformer,
                              the gate/intent centroids, the LLM backend,
                              and optionally a token-sized LLM request that
                              forces the weights resident and caches the
                              system prompt. Safe to run in a thread.
    build_pipeline(..., components=...)
                              fast: opens corpus.db, builds the BlurbStore,
                              assembles the pipeline. Must run on the thread
                              that will use the pipeline, because sqlite3
                              connections are bound to their creating thread.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Resolve sibling packages from this file's location, not the caller's CWD.
_SPECIES_DIR = Path(__file__).resolve().parents[1]
for _sub in ("tests", "llm-tuning", "pipeline", "."):
    _path = str((_SPECIES_DIR / _sub).resolve())
    if _path not in sys.path:
        sys.path.insert(1, _path)

from blurb_store import BlurbStore  # noqa: E402
from corpus_schema import check_corpus_schema  # noqa: E402
from intent import IntentClassifier  # noqa: E402
from pipeline import DEFAULT_RETRIEVAL_THRESHOLD, RoboRangerPipeline  # noqa: E402
from test_corpus import EMBED_MODEL, open_db  # noqa: E402
from wildlife_gate import WildlifeGate  # noqa: E402


@dataclass
class Components:
    """
    The expensive, sqlite-free half of a pipeline. Build it on any thread
    (see prewarm_components) and hand it to build_pipeline.
    """
    embedder: object
    gate: WildlifeGate
    intent_classifier: IntentClassifier
    llm: object
    timings: dict[str, float] = None  # seconds per step, for logging


def make_backend(backend: Literal["ollama", "llama-cpp"] = "ollama",
                 model: str = "smollm2:360m",
                 backend_host: str | None = None):
    """
    Construct an LLM backend adapter (no network traffic yet). The prompt's
    section headings are stop strings, so a model that starts copying its
    prompt ("SPECIES FACTS: ...") is cut off by the server at once.
    """
    from llm_backends import LlamaCppBackend, OllamaBackend
    from prompt_builder import PROMPT_HEADINGS
    if backend == "ollama":
        return OllamaBackend(model=model,
                             host=backend_host or "http://localhost:11434",
                             stop=PROMPT_HEADINGS)
    if backend == "llama-cpp":
        return LlamaCppBackend(host=backend_host or "http://localhost:8080",
                               model=model, stop=PROMPT_HEADINGS)
    raise ValueError(f"unknown backend: {backend!r}")


def prewarm_components(
    *,
    backend: Literal["ollama", "llama-cpp"] = "ollama",
    model: str = "smollm2:360m",
    backend_host: str | None = None,
    embed_model: str = EMBED_MODEL,
    embedder=None,
    warm_llm: bool = True,
) -> Components:
    """
    Load everything a pipeline needs except the corpus connection: the
    sentence-transformer (seconds), the gate and intent prototype centroids
    (dozens of encodes), the LLM backend adapter, and — when warm_llm — a
    1-token request carrying the real system prompt, which makes the model
    server load its weights and cache that prefix.

    Touches no sqlite, so this is safe to run on a background thread while
    the main thread is doing something else (camera identification). LLM
    failures are recorded in `timings["llm_error"]`, never raised: a model
    server that isn't up yet must not stop the session from starting.
    """
    timings: dict[str, float] = {}
    if embedder is None:
        # Importing sentence_transformers pulls in torch, which on the UNO Q
        # costs more than loading the model itself — time it separately so
        # the startup log accounts for the whole wait.
        t = time.perf_counter()
        from sentence_transformers import SentenceTransformer
        timings["import"] = time.perf_counter() - t
        t = time.perf_counter()
        embedder = SentenceTransformer(embed_model)
        timings["embedder"] = time.perf_counter() - t

    t = time.perf_counter()
    gate = WildlifeGate(embedder)
    intent_classifier = IntentClassifier(embedder)
    timings["centroids"] = time.perf_counter() - t

    llm = make_backend(backend, model, backend_host)
    if warm_llm:
        from prompt_builder import SYSTEM_PROMPT
        t = time.perf_counter()
        try:
            llm.generate(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": "Ready?"}],
                max_tokens=1,
            )
            timings["llm_load"] = time.perf_counter() - t
        except Exception as e:
            timings["llm_load"] = time.perf_counter() - t
            timings["llm_error"] = f"{type(e).__name__}: {e}"

    return Components(embedder=embedder, gate=gate,
                      intent_classifier=intent_classifier, llm=llm,
                      timings=timings)


def build_pipeline(
    *,
    db_path: str | Path,
    backend: Literal["ollama", "llama-cpp"] = "ollama",
    model: str = "smollm2:360m",
    backend_host: str | None = None,
    embed_model: str = EMBED_MODEL,
    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    embedder=None,                   # pass in if already loaded (saves time)
    components: Components | None = None,
) -> RoboRangerPipeline:
    """
    Build a ready-to-call pipeline.

    Args:
        db_path:              path to corpus.db (built by build_corpus.py)
        backend:              'ollama' for laptop iteration, 'llama-cpp'
                              for the Uno Q deployment.
        model:                model name as the backend understands it.
        backend_host:         override default host (optional).
        embed_model:          sentence-transformers model. Default matches
                              what the corpus was indexed with.
        retrieval_threshold:  cosine-sim cutoff; chunks below are dropped.
        embedder:             pre-loaded SentenceTransformer. If None, this
                              function loads one — but loading takes ~3s,
                              so reuse if you can.
        components:           result of prewarm_components(), to skip the
                              slow half entirely. When given, backend/model/
                              backend_host/embed_model/embedder are ignored.

    Returns:
        A RoboRangerPipeline. The pipeline owns the DB connection but NOT
        the embedder (caller may want to reuse it).

    Raises:
        CorpusSchemaError: corpus.db is not schema v2 (partitioned vectors)
            or was embedded with a different model. Checked before the
            embedder loads so a stale corpus fails in milliseconds.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"corpus not found: {db_path}")
    # sqlite3 connections are bound to the thread that creates them, so the
    # connection (and anything holding it) is built here, on the caller's
    # thread, not inside prewarm_components.
    conn = open_db(db_path)
    check_corpus_schema(conn)

    if components is None:
        components = prewarm_components(
            backend=backend, model=model, backend_host=backend_host,
            embed_model=embed_model, embedder=embedder, warm_llm=False)

    return RoboRangerPipeline(
        embedder=components.embedder,
        db_conn=conn,
        llm_backend=components.llm,
        gate=components.gate,
        intent_classifier=components.intent_classifier,
        blurb_store=BlurbStore(conn),
        retrieval_threshold=retrieval_threshold,
    )