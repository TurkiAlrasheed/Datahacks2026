"""
Prints the new DANGER-intent prompt for several species so you can eyeball
what SmolLM2 will actually see. Compare against bad_llm_responses.txt to
verify the prompt structure looks right BEFORE running the LLM.

Run:
    python preview_danger_prompts.py corpus.db
"""

from __future__ import annotations

import sqlite3
import sys

import sqlite_vec

sys.path.insert(1, "../species_identification/llm-tuning")
sys.path.insert(1, "../species_identification/pipeline")
from blurb_store import BlurbStore
from intent import Intent, IntentResult
from prompt_builder import build_messages, Chunk


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def stub_intent() -> IntentResult:
    return IntentResult(Intent.DANGER, "high", 0.78, 0.10, {})


# Cases from bad_llm_responses.txt where SmolLM2 hallucinated risk.
PREVIEW_CASES = [
    ("Omphalotus_olivascens",   "are they safe to eat"),
    ("Omphalotus_olivascens",   "are they dangerous"),
    ("Laetiporus_gilbertsonii", "are they safe to eat"),
    ("Laetiporus_gilbertsonii", "are they dangerous to pets"),
    ("Dione_vanillae",          "are they dangerous"),
    ("Sceloporus_occidentalis", "is it safe to touch"),
]


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.db"
    conn = open_db(db_path)
    store = BlurbStore(conn)

    for sid, query in PREVIEW_CASES:
        blurb = store.get(sid)
        if blurb is None:
            print(f"\n--- {sid} (not in DB, skipping) ---\n")
            continue
        msgs = build_messages(
            query=query,
            blurb=blurb,
            chunks=[Chunk(text="(retrieved chunks omitted for clarity)",
                          score=0.7, source="general")],
            intent_result=stub_intent(),
        )
        print("=" * 72)
        print(f"  species:  {sid}")
        print(f"  blurb:    {blurb.common_name}")
        print(f"  humans:   {blurb.dangerous_to_humans}")
        print(f"  pets:     {blurb.dangerous_to_pets}")
        print(f"  query:    {query}")
        print("=" * 72)
        print()
        print("--- system ---")
        print(msgs[0]["content"])
        print()
        print("--- user ---")
        print(msgs[1]["content"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())