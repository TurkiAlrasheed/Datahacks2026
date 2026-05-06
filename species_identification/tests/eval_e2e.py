"""
Prompt-quality evaluation: tests how SmolLM2 responds to prompts produced
by build_messages(). DOES NOT test the full pipeline — it bypasses the
wildlife gate, intent short-circuit, and orchestrator. Use this when
iterating on the prompt design; use test_pipeline_integration.py and
pipeline_repl.py for orchestrator behavior.

What this harness covers:
    - Does the prompt produce reasonable danger answers?
    - Does the model respect the relevance gate ("if snippets don't
      address the question, use only species facts")?
    - Does polarity routing work? ("safe?" vs "dangerous?" on the same
      species should give consistent answers)

What this harness does NOT cover:
    - Off-topic queries (the gate would block them in real use; here
      they go straight to the LLM and predictably fail). If you want
      to test pipeline behavior end-to-end, use the orchestrator.
    - Intent classification accuracy (use eval_intent.py).

Backends:
    --backend ollama        Uses Ollama on the laptop (default model: llama3.1)
    --backend llama-cpp     Uses llama.cpp's HTTP server (your Uno Q deployment)

For the laptop, point Ollama at SmolLM2 if you have it pulled:
    ollama pull smollm2:360m
    python eval_e2e.py --backend ollama --model smollm2:360m

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

sys.path.insert(1, "../species_identification/pipeline")
sys.path.insert(2, "../species_identification/llm-tuning")
from intent import IntentClassifier, Intent
from prompt_builder import Blurb, Chunk, build_messages, build_prompt


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
RATTLESNAKE = Blurb(
    common_name="Southern Pacific Rattlesnake",
    scientific_name="Crotalus oreganus helleri",
    appearance="A medium-sized pit viper with a triangular head, segmented rattle, "
               "and diamond pattern fading toward the tail.",
    size="80-130 cm long",
    habitat="Coastal sage scrub, chaparral, and rocky hillsides across Southern California.",
    diet="Small mammals, lizards, and birds, ambushed and subdued with venom.",
    behavior="Mostly crepuscular; coils and rattles when threatened.",
    dangerous_to_humans="yes",
    dangerous_to_pets="yes",
    notable="Responsible for most envenomations in San Diego County.",
)

FENCE_LIZARD = Blurb(
    common_name="Western Fence Lizard",
    scientific_name="Sceloporus occidentalis",
    appearance="Small spiny lizard with blue belly patches in males. "
               "Spiny scales, gray-brown body.",
    size="6-9 cm body length",
    habitat="Rocks, fences, woodpiles across the western US.",
    diet="Insects and small arthropods.",
    behavior="Active in daytime, basks on sunny surfaces.",
    dangerous_to_humans="no",
    dangerous_to_pets="no",
    notable="Their blood kills Lyme disease bacteria in tick guts.",
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
        # New design: lead word "Yes" + restated verdict. The model should
        # say "Yes" because rattlesnake humans=yes,pets=yes. The chunks
        # being about taxonomy/fossils means the answer must NOT include
        # those topics.
        must_contain=["yes", "dangerous"],
        must_not_contain=["fossil", "taxonomy"],
        note="Chunks are about taxonomy/fossils — model must use the "
             "VERDICT, not the snippets. Answer should begin with Yes.",
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
        # Note: with the new "restate the VERDICT" directive, the model
        # may not pull the chunk content as strongly as before. We're
        # checking it stays on-topic and gets the polarity right.
        must_contain=["dangerous"],
        note="Chunk is on-point. Model should not contradict the verdict.",
    ),
    TestCase(
        name="no_chunks_falls_back_to_blurb",
        query="what does it eat?",
        blurb=RATTLESNAKE,
        chunks=[],
        expected_intent=Intent.DIET,
        # DIET intent uses the diet blurb field directly. Words "mammal"
        # or "lizard" or "bird" should appear since those are in the diet.
        must_contain=["mammal"],
        note="Retriever returned nothing above threshold. Use blurb only.",
    ),
    TestCase(
        name="harmless_species_question",
        query="will this lizard hurt me?",
        blurb=FENCE_LIZARD,
        chunks=[],
        expected_intent=Intent.DANGER,
        # New design: fence_lizard humans=no,pets=no, query is danger-
        # framed. Lead word "No", verdict "is NOT dangerous".
        must_contain=["no"],
        must_not_contain=["venom", "bite is a medical emergency",
                          "respiratory", "burn"],
        note="Should restate the 'NOT dangerous' verdict, not invent risks.",
    ),
    TestCase(
        name="safety_polarity_dangerous_species",
        query="is it safe to eat?",
        # Reuse rattlesnake as a stand-in dangerous species; in real use
        # this case really matters for poisonous mushrooms.
        blurb=RATTLESNAKE,
        chunks=[],
        expected_intent=Intent.DANGER,
        # Polarity flip: question is "safe?", species is dangerous, so
        # the lead word should be "No" — not safe.
        must_contain=["no"],
        must_not_contain=["yes"],
        note="Polarity flip: 'safe?' on a dangerous species should "
             "answer 'No', not 'Yes'.",
    ),
    # NOTE: there used to be an "off_topic_should_say_unknown" case here
    # that asked "what time does the park close?". It was removed because
    # this harness bypasses the wildlife gate and intent short-circuit,
    # so the LLM will always fabricate an answer to that query — there
    # is no prompt that prevents it. Pipeline-level off-topic handling
    # is tested in test_pipeline_integration.py (test 1, test 5b).
]


# ---------------------------------------------------------------------------
# LLM backends — defined in llm_backends.py so the pipeline can use them
# without dragging in this file's eval fixtures.
# ---------------------------------------------------------------------------
from llm_backends import LlamaCppBackend, OllamaBackend, SAMPLING


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