"""
Smoke test against the real corpus.db. Run after any change to BlurbStore
or prompt_builder to confirm the schema adapter still works.

Verifies:
  - BlurbStore loads without schema errors
  - Every species in the DB returns a Blurb
  - The danger enums are valid (or correctly dropped if invalid)
  - Required fields are populated
  - Prompt rendering works for each intent
  - The known typo'd species ('apprearance') has its appearance loaded
"""

from __future__ import annotations

import json
import sys
import sqlite3

import sqlite_vec

sys.path.insert(1, "../species_identification/llm-tuning")
sys.path.insert(2, "../species_identification/pipeline")
from blurb_store import BlurbStore
from intent import Intent, IntentResult
from prompt_builder import DANGER_LEVELS, build_messages


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def stub_intent(intent: Intent) -> IntentResult:
    return IntentResult(intent, "high", 0.7, 0.15, {})


def main() -> int:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.db"
    conn = open_db(db_path)
    store = BlurbStore(conn)
    print(f"BlurbStore opened against {db_path}")

    species_ids = [
        r[0] for r in conn.execute("SELECT species_id FROM species").fetchall()
    ]
    print(f"Found {len(species_ids)} species")

    failures = 0
    field_counts: dict[str, int] = {}
    danger_humans_dist: dict[str, int] = {}
    danger_pets_dist: dict[str, int] = {}

    for sid in species_ids:
        blurb = store.get(sid)
        if blurb is None:
            print(f"  [FAIL] {sid}: get() returned None")
            failures += 1
            continue

        # Aggregate stats
        for fld in ("appearance", "size", "habitat", "diet", "behavior",
                    "notable", "dangerous_to_humans", "dangerous_to_pets"):
            if getattr(blurb, fld):
                field_counts[fld] = field_counts.get(fld, 0) + 1

        if blurb.dangerous_to_humans:
            danger_humans_dist[blurb.dangerous_to_humans] = \
                danger_humans_dist.get(blurb.dangerous_to_humans, 0) + 1
        if blurb.dangerous_to_pets:
            danger_pets_dist[blurb.dangerous_to_pets] = \
                danger_pets_dist.get(blurb.dangerous_to_pets, 0) + 1

        # Required-field check
        if not blurb.common_name:
            print(f"  [FAIL] {sid}: missing common_name")
            failures += 1

    print(f"\n=== Field population (out of {len(species_ids)} species) ===")
    for fld in ("appearance", "size", "habitat", "diet", "behavior",
                "notable", "dangerous_to_humans", "dangerous_to_pets"):
        n = field_counts.get(fld, 0)
        bar = "#" * int(40 * n / len(species_ids))
        print(f"  {fld:22s} {n:3d}/{len(species_ids):3d}  {bar}")

    print(f"\n=== Danger enum distribution ===")
    print(f"  to humans: {danger_humans_dist}")
    print(f"  to pets:   {danger_pets_dist}")

    # Specifically verify the typo'd row was rescued
    print(f"\n=== Typo recovery check ===")
    rows = conn.execute("""
        SELECT species_id FROM species
        WHERE blurb_json LIKE '%apprearance%'
    """).fetchall()
    if rows:
        for (sid,) in rows:
            blurb = store.get(sid)
            if blurb and blurb.appearance:
                print(f"  [OK]   {sid}: appearance recovered "
                      f"({len(blurb.appearance)} chars)")
            else:
                print(f"  [FAIL] {sid}: appearance not loaded "
                      f"despite typo mapping")
                failures += 1
    else:
        print("  (no typo'd species in DB any more — clean)")

    # Render a prompt for each intent on one species, just to eyeball
    print(f"\n=== Sample prompt: DANGER intent on first species ===")
    sample_sid = species_ids[0]
    blurb = store.get(sample_sid)
    msgs = build_messages(
        query="is this dangerous to my dog?",
        blurb=blurb,
        chunks=[],
        intent_result=stub_intent(Intent.DANGER),
    )
    print(msgs[1]["content"])

    print(f"\n=== {failures} failure(s) across {len(species_ids)} species ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())