"""
Build a fully-wired RoboRangerPipeline from a config.

Single entry point so the CLI tool, eval harness, and Uno Q runtime all
construct the pipeline the same way. If you ever swap the embedder model
or the LLM backend, change it here, not in three places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import sys

sys.path.insert(1, "../species_identification/llm-tuning")
from blurb_store import BlurbStore
from intent import IntentClassifier
from pipeline import DEFAULT_RETRIEVAL_THRESHOLD, RoboRangerPipeline
from test_corpus import EMBED_MODEL, open_db
from wildlife_gate import WildlifeGate


# Reuse the LLM backends from eval_e2e.py — same interface.
# Importing lazily inside build_pipeline so this module can be imported
# without eval_e2e.py being on the path (e.g. during tests).


def build_pipeline(
    *,
    db_path: str | Path,
    backend: Literal["ollama", "llama-cpp"] = "ollama",
    model: str = "smollm2:360m",
    backend_host: str | None = None,
    embed_model: str = EMBED_MODEL,
    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    embedder=None,                   # pass in if already loaded (saves time)
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

    Returns:
        A RoboRangerPipeline. The pipeline owns the DB connection but NOT
        the embedder (caller may want to reuse it).
    """
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embed_model)

    conn = open_db(Path(db_path))

    # Build embedding-based components once and pass them in. They cache
    # centroids on construction; reusing them across queries is ~free, but
    # rebuilding them per query would be silly.
    gate = WildlifeGate(embedder)
    intent_classifier = IntentClassifier(embedder)
    blurb_store = BlurbStore(conn)

    # LLM backend
    from llm_backends import LlamaCppBackend, OllamaBackend
    if backend == "ollama":
        llm = OllamaBackend(
            model=model,
            host=backend_host or "http://localhost:11434",
        )
    elif backend == "llama-cpp":
        llm = LlamaCppBackend(
            host=backend_host or "http://localhost:8080",
            model=model,
        )
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    return RoboRangerPipeline(
        embedder=embedder,
        db_conn=conn,
        llm_backend=llm,
        gate=gate,
        intent_classifier=intent_classifier,
        blurb_store=blurb_store,
        retrieval_threshold=retrieval_threshold,
    )