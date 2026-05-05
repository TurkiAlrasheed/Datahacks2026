"""
End-to-end evaluation: intent -> retrieval (mocked or real) -> prompt -> LLM.

The point of this harness is to test the *relevance gate* — does SmolLM2
actually respect the "if snippets don't address the question, use only
species facts" instruction? You can't test that with unit tests; you need
the real model.

Backends:
    --backend ollama        Uses Ollama on the laptop (default model: llama3.1)
    --backend llama-cpp     Uses llama.cpp's HTTP server (your Uno Q deployment)

For the laptop, point Ollama at SmolLM2 if you have it pulled:
    ollama pull smollm2:360m
    python eval_e2e.py --backend ollama --model smollm2:360m

Each test case specifies:
    - a query
    - a species blurb
    - a list of chunks (some are 'irrelevant' to test the gate)
    - an expectation: should the model answer from snippets, from the blurb,
      or refuse?

The harness scores each response with simple heuristics. It's not a
substitute for eyeballing outputs, but it'll flag obvious regressions.

Saving runs and comparing laptop-vs-UnoQ:
    python eval_e2e.py --backend ollama --save runs/laptop.json
    python eval_e2e.py --backend llama-cpp --host http://uno-q.local:8080 \
                       --save runs/unoq.json
    python diff_runs.py runs/laptop.json runs/unoq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable

from intent import IntentClassifier, Intent
from prompt_builder import Blurb, Chunk, build_messages, build_prompt


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
RATTLESNAKE = Blurb(
    common_name="Southern Pacific Rattlesnake",
    scientific_name="Crotalus oreganus helleri",
    description="A medium-sized pit viper with a triangular head and segmented rattle.",
    habitat="Coastal sage scrub, chaparral, and rocky hillsides across Southern California.",
    diet="Small mammals, lizards, and birds, ambushed and subdued with venom.",
    behavior="Mostly crepuscular; coils and rattles when threatened.",
    danger="Venomous. Bites are medical emergencies. Back away slowly; do not attempt to handle.",
    identification="Diamond pattern fading toward the tail, blunt tail with rattle.",
    conservation="Common; not listed.",
)

FENCE_LIZARD = Blurb(
    common_name="Western Fence Lizard",
    scientific_name="Sceloporus occidentalis",
    description="Small spiny lizard with blue belly patches in males.",
    habitat="Rocks, fences, woodpiles across the western US.",
    diet="Insects and small arthropods.",
    behavior="Active in daytime, basks on sunny surfaces.",
    danger="Harmless to humans.",
    identification="Spiny scales, blue belly, gray-brown body.",
    conservation="Common; not listed.",
)


@dataclass
class TestCase:
    name: str
    query: str
    blurb: Blurb
    chunks: list[Chunk]
    expected_intent: Intent
    # heuristic checks on the model's response
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    note: str = ""


CASES: list[TestCase] = [
    TestCase(
        name="answer_from_blurb_when_chunks_irrelevant",
        query="is this snake dangerous?",
        blurb=RATTLESNAKE,
        chunks=[
            Chunk(text="Rattlesnakes have a long fossil record going back "
                       "millions of years.", score=0.58),
            Chunk(text="The taxonomy of rattlesnakes was revised in 2008.",
                  score=0.56),
        ],
        expected_intent=Intent.DANGER,
        must_contain=["venom"],
        note="Chunks are about taxonomy/fossils — model must use blurb's "
             "danger field, not the snippets.",
    ),
    TestCase(
        name="use_chunk_when_relevant",
        query="how dangerous are bites in San Diego specifically?",
        blurb=RATTLESNAKE,
        chunks=[
            Chunk(text="The Southern Pacific rattlesnake is responsible for "
                       "most envenomations in San Diego County.",
                  score=0.78),
        ],
        expected_intent=Intent.DANGER,
        must_contain=["San Diego"],
        note="Chunk is on-point. Model should incorporate it.",
    ),
    TestCase(
        name="no_chunks_falls_back_to_blurb",
        query="what does it eat?",
        blurb=RATTLESNAKE,
        chunks=[],
        expected_intent=Intent.DIET,
        must_contain=["mammal"],  # from the blurb's diet field
        note="Retriever returned nothing above threshold. Use blurb only.",
    ),
    TestCase(
        name="harmless_species_question",
        query="will this lizard hurt me?",
        blurb=FENCE_LIZARD,
        chunks=[],
        expected_intent=Intent.DANGER,
        must_contain=["harmless"],
        must_not_contain=["venom", "bite is a medical emergency"],
        note="Should pull 'harmless to humans' from the blurb, not invent risks.",
    ),
    TestCase(
        name="off_topic_should_say_unknown",
        query="what time does the park close?",
        blurb=FENCE_LIZARD,
        chunks=[],
        expected_intent=Intent.OTHER,
        must_not_contain=["6:00", "8:00", "10:00"],  # no fabricated times
        note="Off-topic. Model should refuse or admit it doesn't know.",
    ),
]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
# Shared sampling params. Pin these on BOTH backends so cross-environment
# differences come from the model runtime, not from sampler defaults.
SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 42,
    "max_tokens": 200,
}


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "smollm2:360m",
                 host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def generate(self, messages: list[dict]) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": SAMPLING["temperature"],
                "top_p": SAMPLING["top_p"],
                "top_k": SAMPLING["top_k"],
                "seed": SAMPLING["seed"],
                "num_predict": SAMPLING["max_tokens"],
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]


class LlamaCppBackend:
    """
    Talks to llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.
    Start the server on the Uno Q with something like:
        ./llama-server -m smollm2-360m-q8_0.gguf --host 0.0.0.0 --port 8080 \\
                       --ctx-size 2048 --threads 4
    """

    name = "llama-cpp"

    def __init__(self, host: str = "http://localhost:8080",
                 model: str = "smollm2") -> None:
        self.host = host
        self.model = model

    def generate(self, messages: list[dict]) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
            "seed": SAMPLING["seed"],
            "max_tokens": SAMPLING["max_tokens"],
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Embedder loading (only needed if you want intent classification scored too)
# ---------------------------------------------------------------------------
def load_embedder():
    from sentence_transformers import SentenceTransformer
    print("Loading BAAI/bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    class _W:
        def encode(self, texts):
            return model.encode(list(texts), normalize_embeddings=True)

    return _W()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def score_response(case: TestCase, response: str) -> tuple[bool, list[str]]:
    """Cheap heuristic checks. Returns (passed, list_of_issues)."""
    issues = []
    lower = response.lower()
    for needle in case.must_contain:
        if needle.lower() not in lower:
            issues.append(f"missing required: {needle!r}")
    for needle in case.must_not_contain:
        if needle.lower() in lower:
            issues.append(f"contains forbidden: {needle!r}")
    return len(issues) == 0, issues


def run(backend, classifier: IntentClassifier | None = None,
        cases: list[TestCase] | None = None,
        save_path: str | None = None) -> dict:
    cases = cases or CASES
    passed = 0
    record = {
        "backend": backend.name,
        "model": getattr(backend, "model", None),
        "host": getattr(backend, "host", None),
        "sampling": SAMPLING,
        "results": [],
    }

    for case in cases:
        print(f"\n=== {case.name} ===")
        print(f"query: {case.query!r}")
        if case.note:
            print(f"note: {case.note}")

        # Intent (classifier optional for offline runs)
        if classifier is not None:
            ir = classifier.classify(case.query)
            intent_ok = ir.intent == case.expected_intent
            tag = "OK " if intent_ok else "MISS"
            print(f"intent: [{tag}] expected={case.expected_intent.value} "
                  f"got={ir.intent.value} ({ir.confidence}, "
                  f"score={ir.score:+.3f})")
            intent_record = {
                "predicted": ir.intent.value,
                "expected": case.expected_intent.value,
                "confidence": ir.confidence,
                "score": ir.score,
                "margin": ir.margin,
                "match": intent_ok,
            }
        else:
            from intent import IntentResult
            ir = IntentResult(case.expected_intent, "high", 0.7, 0.15, {})
            intent_record = None

        # Prompt + LLM
        messages = build_messages(
            query=case.query, blurb=case.blurb,
            chunks=case.chunks, intent_result=ir,
        )

        import time
        start = time.perf_counter()
        try:
            response = backend.generate(messages)
            latency_s = time.perf_counter() - start
            error = None
        except Exception as e:
            latency_s = time.perf_counter() - start
            response = ""
            error = str(e)
            print(f"BACKEND ERROR: {e}")

        if error is None:
            ok, issues = score_response(case, response)
            tag = "PASS" if ok else "FAIL"
            print(f"latency: {latency_s:.2f}s")
            print(f"response [{tag}]: {response.strip()}")
            if issues:
                for i in issues:
                    print(f"  - {i}")
            if ok:
                passed += 1
        else:
            ok, issues = False, [f"backend_error: {error}"]

        record["results"].append({
            "case": case.name,
            "query": case.query,
            "species": case.blurb.common_name,
            "expected_intent": case.expected_intent.value,
            "intent": intent_record,
            "response": response,
            "latency_s": round(latency_s, 3),
            "error": error,
            "passed": ok,
            "issues": issues,
        })

    print(f"\n=== {passed}/{len(cases)} cases passed heuristic checks ===")
    print("Note: heuristics catch obvious regressions but you should still "
          "read every response.")

    if save_path:
        from pathlib import Path
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2))
        print(f"\nSaved to {save_path}")

    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "llama-cpp"],
                    default="ollama")
    ap.add_argument("--model", default="smollm2:360m")
    ap.add_argument("--host", default=None,
                    help="override backend host URL")
    ap.add_argument("--no-intent", action="store_true",
                    help="skip intent classification (don't load embedder)")
    ap.add_argument("--save", default=None,
                    help="path to write a JSON record of this run "
                         "(used by diff_runs.py)")
    args = ap.parse_args()

    if args.backend == "ollama":
        backend = OllamaBackend(
            model=args.model,
            host=args.host or "http://localhost:11434",
        )
    else:
        backend = LlamaCppBackend(
            host=args.host or "http://localhost:8080",
            model=args.model,
        )

    classifier = None
    if not args.no_intent:
        classifier = IntentClassifier(load_embedder())

    run(backend, classifier, save_path=args.save)


if __name__ == "__main__":
    main()