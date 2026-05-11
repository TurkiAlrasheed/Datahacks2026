"""
Prose pass for RoboRanger blurbs.

For each reviewed species, generate natural-language clauses for the fields
that read awkwardly when templated directly (size, appearance, behavior,
notable). The runtime pipeline uses these for the direct-route formatters
when prose_reviewed=true, and falls back to the structured field otherwise.

Why a separate pass: the structured fields are field-guide data — terse,
labeled, machine-readable. The prose versions are what gets read aloud to
users by the on-device 360M model's text replacement. Keeping both lets the
corpus and prompt builder use the structured form while runtime answers
read naturally.

Workflow:
    1. python generate_prose.py blurbs.yaml          # generate for all
    2. Open blurbs.yaml, read each *_prose, edit by hand
    3. Set `prose_reviewed: true` on each entry when done
    4. Re-running skips species with prose_reviewed=true unless --force

Run on your laptop where Ollama is fast. Don't run this on the Uno Q.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
import ollama
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL = "llama3.1:8b"
MAX_TOKENS = 200          # clauses are always short; cap protects against rambling
NUM_CTX = 1024            # tiny inputs, tiny prompt — full 4096 is wasteful

# Fields we generate prose for. Order matches natural reading order so the
# YAML diff is readable.
PROSE_FIELDS = ("size", "appearance", "behavior", "diet", "notable")


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
#
# Each prompt is a system prompt + user message pair. The system prompt
# defines the rewriting task and the output shape; the user message carries
# the species-specific input.
#
# Few-shot exemplars do most of the work. Two or three exemplars per field,
# chosen to span the variety of real inputs (animal vs plant, simple vs
# multi-clause). Adding more exemplars plateaus quickly and burns context.
#
# All prompts produce a complete sentence starting with "The {common_name}".
# Originally I considered partial clauses for the runtime template to
# inject ("The X {prose}.") but full sentences let the prose carry its own
# punctuation and grammar — easier to review, no template/data coupling.

PROSE_SYSTEM_PROMPTS = {
    "size": """You rewrite size measurements as a complete sentence about a species. \
Use full unit words (millimeters, centimeters, meters, kilograms, grams). \
Convert hyphenated ranges to "X to Y". Keep all measurements from the input. \
Be factual and concise — no flourishes.

Output exactly one sentence starting with "The {common_name}". Output nothing \
else — no quotes, no preamble, no explanation.

Examples:

Input: "43-64 cm length, 91-153 cm wingspan, 1-2.5 kg weight"
Output: The great horned owl measures 43 to 64 centimeters in length, with a wingspan of 91 to 153 centimeters and a weight of 1 to 2.5 kilograms.

Input: "1-8 m long vine, 5-20 cm long and 5-6 cm diameter fruit"
Output: The wild cucumber grows as a vine 1 to 8 meters long, producing fruit 5 to 20 centimeters in length and 5 to 6 centimeters in diameter.

Input: "6-9 mm body length"
Output: The oblique streaktail hoverfly is 6 to 9 millimeters long.""",

    "appearance": """You rewrite appearance descriptions as a complete sentence about a species. \
Keep every visual detail from the input. Don't add details the input doesn't \
mention. Be factual and concise.

Output exactly one sentence starting with "The {common_name}". Output nothing \
else — no quotes, no preamble, no explanation.

Examples:

Input: "large, mottled gray-brown owl with prominent ear tufts. yellow eyes, and a white patch on the throat"
Output: The great horned owl is a large, mottled gray-brown bird with prominent ear tufts, yellow eyes, and a white patch on its throat.

Input: "black and yellow striped abdomen and thorax, clear wings with dark veins, large compound eyes"
Output: The oblique streaktail hoverfly has a black-and-yellow striped abdomen and thorax, clear wings with dark veins, and large compound eyes.

Input: "slender, flexible, gray-green fuzzy stems with a fragrant smell"
Output: The California sagebrush has slender, flexible, gray-green fuzzy stems with a fragrant smell.""",

    "diet": """You rewrite diet descriptions as a complete sentence about a species. \
Keep every dietary detail from the input. Don't add details the input doesn't \
mention. Be factual and concise.

Output exactly one sentence starting with "The {common_name}". Output nothing \
else — no quotes, no preamble, no explanation.

Examples:
Input: "carnivore, preying on small to medium-sized mammals, birds, reptiles, and insects"
Output: The great horned owl is a carnivore that preys on small to medium-sized mammals, birds, reptiles, and insects.

Input: "nectar and pollen, with a preference for flowers in the pea family"
Output: The oblique streaktail hoverfly feeds on nectar and pollen, with a preference for flowers in the pea family.

Input: "herbivore, consuming a variety of plants including grasses, shrubs, and trees"
Output: The California sagebrush is an herbivore that consumes a variety of plants, including grasses, shrubs, and trees.""",

    "behavior": """You rewrite behavior descriptions as a complete sentence about a species. \
Keep every behavioral detail from the input. Don't add details the input \
doesn't mention. Be factual and concise.

Output exactly one sentence starting with "The {common_name}". Output nothing \
else — no quotes, no preamble, no explanation.

Examples:

Input: "nocturnal, solitary"
Output: The great horned owl is nocturnal and solitary.

Input: "loud foragers, with a high-pitched buzzing sound"
Output: The yellow-faced bumblebee is a loud forager that makes a high-pitched buzzing sound.

Input: "provides good cover for smaller birds and other animals; allelopathic, secreting chemicals to inhibit other plants"
Output: The California sagebrush provides good cover for smaller birds and animals, and is allelopathic — it secretes chemicals that inhibit other plants.""",

    "notable": """You rewrite a notable-fact description as a complete, engaging sentence \
about a species. Keep the surprising or interesting detail front and center. \
Don't add details the input doesn't mention.

Output exactly one sentence starting with "The {common_name}". Output nothing \
else — no quotes, no preamble, no explanation.

Examples:

Input: "they have a poor sense of smell, but can hear sounds from 10 miles away and see in much dimmer light than humans"
Output: The great horned owl has a poor sense of smell, but it can hear sounds from 10 miles away and see in much dimmer light than humans can.

Input: "they pollinate tomatoes better than humans, as they can perform 'buzz pollination' by vibrating their flight muscles to release pollen from flowers"
Output: The yellow-faced bumblebee pollinates tomatoes better than humans by performing buzz pollination — vibrating its flight muscles to release pollen from flowers.""",
}


PROSE_USER_TEMPLATE = """Input: "{value}"
Output:"""


# Schema enforcement: ask Ollama to return a JSON object with one string
# field. Same approach as build_blurbs.py — the format param makes structured
# extraction reliable on Llama 3.1 8B.
PROSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string"},
    },
    "required": ["sentence"],
}


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

@dataclass
class ProseResult:
    field: str
    sentence: str
    raw: str


def _system_prompt(field: str, common_name: str) -> str:
    """Format the field-specific system prompt with the species' common name."""
    return PROSE_SYSTEM_PROMPTS[field].format(common_name=common_name)


def generate_prose(
    client: ollama.Client,
    species_id: str,
    common_name: str,
    field: str,
    value: str,
) -> ProseResult | None:
    """
    Ask the model to rewrite `value` (a structured field) as a complete
    sentence. Returns None on persistent failure.
    """
    # Wrap the schema instruction inline so the few-shot examples in the
    # system prompt don't fight with the JSON output requirement. Adding
    # "Return JSON {sentence: ...}" to the user message has worked better
    # than mixing it into the few-shot block in my testing — the few-shot
    # examples stay clean and the model still honors the format.
    user_msg = (
        PROSE_USER_TEMPLATE.format(value=value)
        + '\n\nReturn the result as JSON: {"sentence": "..."}'
    )

    resp = None
    for attempt in range(3):
        try:
            resp = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _system_prompt(field, common_name)},
                    {"role": "user", "content": user_msg},
                ],
                format=PROSE_JSON_SCHEMA,
                options={
                    # 0.2 matches build_blurbs.py — small randomness helps
                    # rewriting tasks vs greedy decoding's literal echo.
                    "temperature": 0.2,
                    "num_predict": MAX_TOKENS,
                    "num_ctx": NUM_CTX,
                },
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"\n[OLLAMA ERROR, retrying] {species_id}/{field} "
                      f"attempt {attempt + 1}/3: {e}", file=sys.stderr)
                time.sleep(5)
                continue
            print(f"\n[OLLAMA ERROR] {species_id}/{field}: {e}", file=sys.stderr)
            return None

    if resp is None:
        return None

    raw = resp["message"]["content"].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n[JSON PARSE ERROR] {species_id}/{field}: {e}", file=sys.stderr)
        print(f"  raw output:\n{raw}", file=sys.stderr)
        return None

    sentence = parsed.get("sentence", "").strip()
    if not sentence:
        print(f"\n[EMPTY OUTPUT] {species_id}/{field}", file=sys.stderr)
        return None

    # Strip wrapping quotes the model sometimes adds inside the JSON string.
    if (sentence.startswith('"') and sentence.endswith('"')) or \
       (sentence.startswith("'") and sentence.endswith("'")):
        sentence = sentence[1:-1].strip()

    return ProseResult(field=field, sentence=sentence, raw=raw)


# -----------------------------------------------------------------------------
# YAML I/O — same pattern as build_blurbs.py to avoid the ref-rebind footgun
# -----------------------------------------------------------------------------

def load_blurbs(path: Path) -> dict:
    if not path.exists():
        print(f"blurbs file not found: {path}", file=sys.stderr)
        sys.exit(1)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "blurbs" not in data or not isinstance(data["blurbs"], dict):
        print(f"malformed blurbs file: missing 'blurbs' key", file=sys.stderr)
        sys.exit(1)
    return data


def save_blurbs(path: Path, data: dict) -> None:
    """Atomic write. See build_blurbs.save() for the rationale on the copy."""
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
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def needs_field(blurb: dict, field: str, force: bool) -> bool:
    """Whether to (re)generate this field for this blurb."""
    prose_key = f"{field}_prose"
    if force:
        # --force still respects prose_reviewed: never clobber human-edited
        # prose. To force regeneration of reviewed prose, edit the YAML
        # directly to flip prose_reviewed back to false.
        if blurb.get("prose_reviewed"):
            return False
        return True
    if blurb.get("prose_reviewed"):
        return False
    if prose_key not in blurb or not blurb[prose_key]:
        return True
    return False


def main(blurbs_file: str, *, species_filter: str | None,
         field_filter: str | None, force: bool, dry_run: bool) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    out_path = Path(blurbs_file)
    data = load_blurbs(out_path)

    fields_to_run = (field_filter,) if field_filter else PROSE_FIELDS

    # Build the work list: which (species, field) pairs need generation.
    work: list[tuple[str, str]] = []
    for species_id, blurb in data["blurbs"].items():
        if species_filter and species_id != species_filter:
            continue
        if not blurb.get("reviewed"):
            # Never run prose against unreviewed blurbs — the source data
            # quality isn't guaranteed yet.
            continue
        for field in fields_to_run:
            if field not in blurb or not blurb[field]:
                continue
            if needs_field(blurb, field, force):
                work.append((species_id, field))

    if not work:
        print("Nothing to do.")
        return

    n_species = len({sid for sid, _ in work})
    print(f"Generating prose for {len(work)} fields across {n_species} species "
          f"using {MODEL}...")
    if dry_run:
        print("(dry run — will not write)")

    client = ollama.Client()
    failures: list[tuple[str, str]] = []

    # Group by species so the progress bar reads naturally and so we save
    # after each species finishes — partial progress survives a crash.
    by_species: dict[str, list[str]] = {}
    for sid, field in work:
        by_species.setdefault(sid, []).append(field)

    for species_id in tqdm(by_species, desc="Species"):
        blurb = data["blurbs"][species_id]
        common_name = blurb.get("common_name") or species_id.replace("_", " ")
        any_generated = False

        for field in by_species[species_id]:
            value = str(blurb[field]).strip()
            t = time.perf_counter()
            result = generate_prose(client, species_id, common_name, field, value)
            elapsed = time.perf_counter() - t

            if result is None:
                failures.append((species_id, field))
                continue

            print(f"  [{species_id}] {field}_prose ({elapsed:.1f}s)",
                  file=sys.stderr)
            print(f"    in:  {value}", file=sys.stderr)
            print(f"    out: {result.sentence}", file=sys.stderr)

            blurb[f"{field}_prose"] = result.sentence
            any_generated = True

        if any_generated:
            # Reset the review flag whenever any prose was regenerated.
            # A human flips it back to true after reading every prose
            # field for this species.
            blurb["prose_reviewed"] = False
            if not dry_run:
                save_blurbs(out_path, data)

    print(f"\nDone. Generated prose for {len(work) - len(failures)} fields.")
    if failures:
        print(f"\n{len(failures)} fields failed:")
        for sid, field in failures:
            print(f"  - {sid} / {field}")
    print(f"\nNext: open {out_path} and review every *_prose field. Set "
          f"`prose_reviewed: true` on each species when its prose looks correct.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blurbs_file", help="path to blurbs.yaml")
    p.add_argument("--species", help="generate for one species only")
    p.add_argument("--field", choices=PROSE_FIELDS,
                   help="only this field type across species")
    p.add_argument("--force", action="store_true",
                   help="regenerate prose even if it exists "
                        "(but never overwrites prose_reviewed: true)")
    p.add_argument("--dry-run", action="store_true",
                   help="show outputs without writing the YAML")
    args = p.parse_args()

    main(
        args.blurbs_file,
        species_filter=args.species,
        field_filter=args.field,
        force=args.force,
        dry_run=args.dry_run,
    )