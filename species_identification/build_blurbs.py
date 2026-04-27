"""
Blurb drafter for RoboRanger species agent. Used on laptop, then compiled
into the corpus DB.

For each species, fetches the Wikipedia article (reusing build_corpus.py's
fetcher) and asks a local Ollama model to extract a structured blurb
optimized for a 360M-parameter on-device model. Output is a YAML file you
hand-edit before compiling into the corpus DB.

The schema is deliberately field-based rather than prose: SmolLM2-360M
follows labeled key-value context far more reliably than paragraphs.

Run on your laptop:
    pip install ollama pyyaml tqdm
    python build_blurbs.py species.json blurbs.yaml

The script is resumable: re-running skips species already in the output
file. Delete an entry from the YAML to force a re-draft.

Workflow:
    1. python build_blurbs.py species.json blurbs.yaml   (auto-draft)
    2. Open blurbs.yaml, review/edit every entry         (manual pass)
    3. Set `reviewed: true` on each entry when done      (gate)
    4. python compile_blurbs.py blurbs.yaml corpus.db    (separate step)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
import ollama
from tqdm import tqdm

from build_corpus import fetch_article, NON_SPECIES_CLASSES


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL = "llama3.1:8b"
MAX_TOKENS = 600
WIKI_INPUT_CHAR_LIMIT = 6000

SYSTEM_PROMPT = """You are extracting structured field-guide data for a small \
on-device language model (360M parameters) running on a robot. The model is \
not capable of synthesizing across paragraphs reliably — it needs short, \
labeled, factual fields it can read back almost verbatim.

For each species, output a JSON object with these exact fields:

  common_name: The most widely-used English common name. If multiple, pick one.
  size: A short measurement range with units. e.g. "5-8 cm body length",
        "wingspan 60-80 cm". Omit if not in source.
  appearance: 1-2 short sentences describing what it looks like. Focus on
              identification: color, markings, distinctive features. No prose
              flourishes. Max ~30 words.
  habitat: Where you'd encounter it. e.g. "Coastal scrub and chaparral in
           California; common on rocks and fences." Max ~25 words.
  diet: What it eats, briefly. Max ~15 words.
  behavior: 1-2 short identification-relevant behavior notes (e.g. nocturnal,
            solitary, builds web at dusk). Max ~25 words. Omit if nothing
            distinctive.
  dangerous_to_humans: One of: "no", "mild" (e.g. painful sting but not
                       medically serious), "yes" (medically significant), or
                       "unknown". Always include this field.
  dangerous_to_pets: Same scale. Always include this field.
  notable: One short interesting fact, if there's a genuinely notable one.
           Otherwise omit. Max ~25 words.

CRITICAL RULES:
- Use ONLY information present in the provided Wikipedia text. Do not add
  details from your own knowledge, even if you're sure they're correct.
- If a field's information is not in the source, set it to null
  (except dangerous_to_humans / dangerous_to_pets, which default to "unknown").
- Prefer short phrases over sentences. The model reads these back to users.
- No hedging language ("may be", "is thought to") — state facts plainly or
  omit. Hedging confuses small models.
- Output ONLY the JSON object. No preamble, no code fences, no commentary.
- For fields where the source has no information, use null (not an empty string)."""

USER_PROMPT_TEMPLATE = """Species (binomial): {binomial}
Wikipedia page title: {page_title}

--- Wikipedia text ---
{wiki_text}
--- end ---

Output the JSON object now."""

BLURB_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "common_name":         {"type": "string"},
        "size":                {"type": "string"},
        "appearance":          {"type": "string"},
        "habitat":             {"type": "string"},
        "diet":                {"type": "string"},
        "behavior":            {"type": "string"},        
        "dangerous_to_humans": {"type": "string", "enum": ["no", "mild", "yes", "unknown"]},
        "dangerous_to_pets":   {"type": "string", "enum": ["no", "mild", "yes", "unknown"]},
        "notable":             {"type": "string"},
    },
    "required": ["common_name", "dangerous_to_humans", "dangerous_to_pets"],
}


# -----------------------------------------------------------------------------
# Drafting
# -----------------------------------------------------------------------------

@dataclass
class DraftResult:
    fields: dict
    raw: str
    source_title: str


def _truncate_wiki(text: str) -> str:
    if len(text) <= WIKI_INPUT_CHAR_LIMIT:
        return text
    cut = text.rfind("\n\n", 0, WIKI_INPUT_CHAR_LIMIT)
    if cut < WIKI_INPUT_CHAR_LIMIT // 2:
        cut = WIKI_INPUT_CHAR_LIMIT
    return text[:cut].rstrip() + "\n\n[truncated]"


def draft_blurb(client: ollama.Client, species_id: str) -> DraftResult | None:
    """
    Fetch Wikipedia text for one species and ask the local model to extract
    fields. Returns None if the article can't be found or the response can't
    be parsed.
    """
    binomial = species_id.replace("_", " ")
    print(f"  [FETCH] {species_id} -> querying '{binomial}'", file=sys.stderr, flush=True)
    article = fetch_article(binomial)
    if not article:
        print(f"  [FETCH FAILED] {species_id}: no article returned",
              file=sys.stderr, flush=True)
        return None
    print(f"  [FETCH OK] {species_id} -> got '{article[0]}'",
          file=sys.stderr, flush=True)

    page_title, _common_name, wiki_text = article
    wiki_text = _truncate_wiki(wiki_text)

    user_msg = USER_PROMPT_TEMPLATE.format(
        binomial=binomial,
        page_title=page_title,
        wiki_text=wiki_text,
    )

    resp = None
    for attempt in range(3):
        try:
            resp = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format=BLURB_JSON_SCHEMA,
                options={
                    "temperature": 0.2,
                    "num_predict": MAX_TOKENS,
                    "num_ctx": 4096,
                },
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"\n[OLLAMA ERROR, retrying] {species_id} attempt "
                      f"{attempt + 1}/3: {e}", file=sys.stderr)
                time.sleep(5)
                continue
            print(f"\n[OLLAMA ERROR] {species_id}: {e}", file=sys.stderr)
            return None

    if resp is None:
        return None

    raw = resp["message"]["content"].strip()

    try:
        fields = json.loads(raw)
    except json.JSONDecodeError as e:
        # Fallback: extract first {...} block in case the model added stray text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                fields = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                print(f"\n[JSON PARSE ERROR] {species_id}: {e}", file=sys.stderr)
                print(f"  raw output:\n{raw}", file=sys.stderr)
                return None
        else:
            print(f"\n[JSON PARSE ERROR] {species_id}: {e}", file=sys.stderr)
            print(f"  raw output:\n{raw}", file=sys.stderr)
            return None

    if not isinstance(fields, dict):
        print(f"\n[BAD SHAPE] {species_id}: expected dict, got {type(fields).__name__}",
              file=sys.stderr)
        return None

    fields = {k: v for k, v in fields.items()
              if v is not None and v != "" and v != "null"}

    return DraftResult(fields=fields, raw=raw, source_title=page_title)


# -----------------------------------------------------------------------------
# YAML I/O
# -----------------------------------------------------------------------------

def load_existing(path: Path) -> dict:
    """Load an existing blurbs YAML, or return an empty skeleton."""
    if not path.exists():
        return {"blurbs": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "blurbs" not in data or not isinstance(data["blurbs"], dict):
        data["blurbs"] = {}
    return data


def save(path: Path, data: dict) -> None:
    """
    Write the YAML file atomically. Sorts entries alphabetically for stable
    diffs, but does so via a *copy* — the caller's `data["blurbs"]` reference
    must not be rebound, or callers holding a reference to it will silently
    write to an orphaned dict. (This is the bug that caused only the first
    species to ever land on disk.)
    """
    sorted_blurbs = dict(sorted(data["blurbs"].items()))
    payload = {**data, "blurbs": sorted_blurbs}

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=88,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # atomic on POSIX, near-atomic on Windows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(species_file: str, output_file: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    species_list = json.loads(Path(species_file).read_text(encoding="utf-8"))
    out_path = Path(output_file)
    data = load_existing(out_path)

    client = ollama.Client()  # talks to localhost:11434 by default

    to_process = [
        s for s in species_list
        if s not in NON_SPECIES_CLASSES and s not in data["blurbs"]
    ]
    skipped_existing = (
        len(species_list)
        - len(to_process)
        - sum(1 for s in species_list if s in NON_SPECIES_CLASSES)
    )

    if skipped_existing:
        print(f"Skipping {skipped_existing} species already in {out_path.name} "
              f"(delete entries to force re-draft).")

    if not to_process:
        print("Nothing to do.")
        return

    print(f"Drafting blurbs for {len(to_process)} species using {MODEL}...")
    print("Edit the output file by hand; set `reviewed: true` on each entry.\n")

    failures: list[str] = []

    for species_id in tqdm(to_process, desc="Drafting"):
        try:
            result = draft_blurb(client, species_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] {species_id}: {e}", file=sys.stderr)
            failures.append(species_id)
            continue

        if result is None:
            failures.append(species_id)
            continue

        # Mutate data["blurbs"] in place. Do NOT keep a separate `blurbs`
        # alias to it — save() used to rebind data["blurbs"] to a sorted
        # copy, which orphaned any outside reference and caused every
        # subsequent write to land on a dict that was never persisted.
        data["blurbs"][species_id] = {
            "reviewed": False,
            "source": f"wikipedia:{result.source_title}",
            **result.fields,
        }

        print(f"  [WRITE] {species_id} -> blurbs now has "
              f"{len(data['blurbs'])} entries", file=sys.stderr)

        save(out_path, data)

    print(f"\nDone. Drafted {len(to_process) - len(failures)} blurbs.")
    if failures:
        print(f"\n{len(failures)} species failed (no Wikipedia hit, parse error, "
              f"or API error). Add these by hand:")
        for s in failures:
            print(f"  - {s}")
    print(f"\nNext: open {out_path} and review every entry. Set "
          f"`reviewed: true` when each looks correct.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_blurbs.py <species.json> <blurbs.yaml>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])