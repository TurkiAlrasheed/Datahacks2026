"""
Test the colloquial-query hypothesis.

For each of the failing dilute items from the last eval run, run THREE
phrasings of the same question against the same corpus chunks and compare
top-1 cosine sim:

  - colloquial: how a visitor would actually ask it (the original failing query)
  - neutral:    plain-language but uses some content vocabulary
  - encyclopedic: phrased the way a Wikipedia article would describe it

If cosine sim systematically goes up from colloquial -> neutral -> encyclopedic
on most items, the hypothesis is confirmed. The fix is query rewriting.

If sim is flat across phrasings, the failure is something else (the right
chunk simply isn't the closest neighbor regardless of phrasing) and we should
look at chunking or reranking instead.

Usage:
    python test_phrasing.py corpus.db
"""

from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from test_corpus import EMBED_MODEL, open_db, retrieve, l2_to_cosine


# Each entry: (id, species_id, colloquial, neutral, encyclopedic, expected_keywords).
# Expected_keywords is what should appear in the top-1 chunk if retrieval is
# actually finding the right content (vs a topical-but-wrong chunk).
TESTS = [
    (
        "procyon_swim",
        "Procyon_lotor",
        "can they swim",                                    # original failing
        "do raccoons swim well",                            # neutral
        "raccoon swimming locomotion in water",             # encyclopedic
        ["swim", "water", "km/h"],
    ),
    (
        "procyon_smart",
        "Procyon_lotor",
        "are they smart",                                   # original failing
        "how smart are raccoons",                           # neutral
        "raccoon intelligence problem-solving cognition",   # encyclopedic
        ["intelligence", "lock", "memory", "Davis"],
    ),
    (
        "procyon_winter",
        "Procyon_lotor",
        "what do they do in winter",                        # original failing
        "raccoon behavior during winter",                   # neutral
        "raccoon winter rest fat storage torpor",           # encyclopedic
        ["winter", "fat", "rest"],
    ),
    (
        "procyon_dark",
        "Procyon_lotor",
        "how do they see in the dark",                      # original failing
        "do raccoons have good night vision",               # neutral
        "raccoon nocturnal vision tapetum lucidum",         # encyclopedic
        ["tapetum", "twilight", "vision", "nocturnal"],
    ),
    (
        "calypte_dive",
        "Calypte_anna",
        "how fast can they dive",                           # original failing
        "how fast is the dive",                             # neutral
        "Anna's hummingbird courtship dive speed",          # encyclopedic
        ["dive", "27", "m/s", "385", "courtship"],
    ),
    (
        "apis_waggle",
        "Apis_mellifera",
        "what is the waggle dance",                         # original failing (only just)
        "how do bees communicate food locations",           # neutral
        "honey bee waggle dance Karl von Frisch communication",  # encyclopedic
        ["waggle", "dance", "Frisch", "communicate"],
    ),
    # Control: a query that already worked. If encyclopedic phrasing doesn't
    # IMPROVE this one, that's a useful signal too - means we're at a ceiling
    # rather than systematically losing to phrasing.
    (
        "procyon_heavy_CONTROL",
        "Procyon_lotor",
        "how big do raccoons get",                          # colloquial version
        "how heavy can a raccoon get",                      # neutral (this passed)
        "raccoon body weight size dimensions",              # encyclopedic
        ["weight", "kg", "lb"],
    ),
]


def keyword_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def main(db_path: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    db = Path(db_path)
    if not db.exists():
        sys.exit(f"Corpus not found: {db}")

    print(f"Loading {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    conn = open_db(db)
    print()

    # Tracks improvement deltas across all tests
    deltas_neutral = []
    deltas_encyc = []
    keyword_flips = 0  # cases where encyclopedic phrasing recovered keyword match

    print(f"{'item':<28}{'phrasing':<14}{'cos':>6}  {'kw':<3}  top1 preview")
    print("-" * 110)

    for test_id, species_id, colloq, neutral, encyc, kws in TESTS:
        results = []
        for label, query in [("colloquial", colloq),
                             ("neutral", neutral),
                             ("encyclopedic", encyc)]:
            top = retrieve(conn, embedder, species_id, query, k=1)
            if top:
                cat, text, dist = top[0]
                cos = l2_to_cosine(dist)
                kw = keyword_hit(text, kws)
                preview = text[:60].replace("\n", " ")
            else:
                cos = float("nan")
                kw = False
                preview = "(no result)"
            results.append((label, query, cos, kw, preview))
            cos_str = f"{cos:.3f}" if cos == cos else "  -  "
            kw_str = "yes" if kw else "no"
            print(f"{test_id:<28}{label:<14}{cos_str:>6}  {kw_str:<3}  {preview}")

        # Compute deltas relative to colloquial
        coll_cos = results[0][2]
        neut_cos = results[1][2]
        encyc_cos = results[2][2]
        if coll_cos == coll_cos and neut_cos == neut_cos:
            deltas_neutral.append(neut_cos - coll_cos)
        if coll_cos == coll_cos and encyc_cos == encyc_cos:
            deltas_encyc.append(encyc_cos - coll_cos)

        # Did encyclopedic recover a keyword match the colloquial missed?
        if not results[0][3] and results[2][3]:
            keyword_flips += 1

        print()

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    if deltas_neutral:
        avg_n = sum(deltas_neutral) / len(deltas_neutral)
        print(f"  Avg cosine delta (neutral - colloquial):     {avg_n:+.3f}  "
              f"(n={len(deltas_neutral)})")
    if deltas_encyc:
        avg_e = sum(deltas_encyc) / len(deltas_encyc)
        print(f"  Avg cosine delta (encyclopedic - colloquial): {avg_e:+.3f}  "
              f"(n={len(deltas_encyc)})")
    print(f"  Keyword recoveries (colloq miss -> encyc hit): "
          f"{keyword_flips} / {len(TESTS)}")
    print()
    print("INTERPRETATION:")
    print("  If avg delta (encyclopedic - colloquial) is >= +0.05 AND keyword")
    print("  recoveries >= half of tests: hypothesis CONFIRMED. Query rewriting")
    print("  will fix it.")
    print()
    print("  If deltas are flat (< +0.02) or keyword flips are rare, the")
    print("  problem is NOT phrasing - look at chunking, reranking, or a")
    print("  different retrieval strategy.")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_phrasing.py <corpus.db>")
        sys.exit(1)
    main(sys.argv[1])