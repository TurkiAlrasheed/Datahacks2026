"""
Prompt assembly for SmolLM2-360M on the Uno Q.

Builds the full prompt from:
  - the always-injected blurb (structured anchor for this species)
  - retrieved corpus chunks (may be empty if all below retriever threshold)
  - intent label (drives task instruction wording + which blurb fields to lean on)

Design choices for a 360M model:
  - Short, declarative system instruction. Long instructions get ignored.
  - Explicit "if the snippets don't answer the question, say so and use only
    the species facts above" — this is the relevance gate.
  - Intent-specific task line at the end so the model knows what shape of
    answer to produce.
  - Hard cap on chunk count and chunk length: the Uno Q has tight context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intent import Intent, IntentResult


# Tune to your context budget. SmolLM2 has 8k context but on the Uno Q you
# want generation to be fast, so keep prompts lean.
MAX_CHUNKS = 3
MAX_CHARS_PER_CHUNK = 400
MAX_BLURB_CHARS = 800


SYSTEM_PROMPT = (
    "You are RoboRanger, a field guide for Southern California wildlife. "
    "Answer the visitor's question about the identified species using only "
    "the SPECIES FACTS and SNIPPETS provided. Be concise: 1-3 sentences. "
    "Do not invent facts. If the snippets do not address the question, "
    "answer using only the SPECIES FACTS. If neither contains the answer, "
    "say you don't know."
)


# Intent-specific closing instructions. Kept short — every token costs latency.
TASK_INSTRUCTIONS: dict[Intent, str] = {
    Intent.DESCRIPTION: (
        "Give a brief description: appearance and one or two notable traits."
    ),
    Intent.DANGER: (
        "State plainly whether it poses a risk to humans and what to do. "
        "If unsure, err on the side of caution."
    ),
    Intent.DIET: "Describe what the species eats.",
    Intent.HABITAT: "Describe where the species lives.",
    Intent.BEHAVIOR: "Describe the relevant behavior.",
    Intent.IDENTIFICATION: (
        "Point out the key features that distinguish this species."
    ),
    Intent.CONSERVATION: "State the conservation status if known.",
    Intent.OTHER: "Answer the visitor's question directly and briefly.",
}


@dataclass
class Chunk:
    """A retrieved corpus chunk."""
    text: str
    score: float           # cosine similarity from retriever
    source: str = ""       # optional, for debugging only — not shown to LLM


# Allowed values for the danger enums. Mirror the schema in build_blurbs.py
# so a typo in either file fails loudly at construction time rather than
# silently confusing the LLM.
DANGER_LEVELS = ("no", "mild", "yes", "unknown")


@dataclass
class Blurb:
    """
    Structured species blurb. Mirrors the schema written by build_blurbs.py
    (and stored as columns on the `species` table by compile_blurbs.py).

    Field-by-field design notes:
      - appearance: 1-2 sentences for identification. Replaces the old
        free-form `description`. Loaded into DESCRIPTION/IDENTIFICATION
        prompts.
      - size: short measurement string. Useful in DESCRIPTION/IDENTIFICATION.
      - dangerous_to_humans / dangerous_to_pets: enum strings. The prompt
        renders the enum directly — small models read labeled enums more
        reliably than free-form risk prose, and the corpus chunks carry
        the specifics.
      - notable: one-sentence interesting fact. Folded into descriptions.
      - scientific_name: derived from species_id when constructing the
        Blurb; not stored as a column. Optional in the dataclass for
        cases where it's not derivable.
    """
    common_name: str
    scientific_name: str | None = None

    # Identification / description fields
    appearance: str | None = None
    size: str | None = None

    # Ecology fields
    habitat: str | None = None
    diet: str | None = None
    behavior: str | None = None

    # Risk fields — enums, not free text
    dangerous_to_humans: str | None = None   # "no" | "mild" | "yes" | "unknown"
    dangerous_to_pets: str | None = None     # same scale

    # Extras
    notable: str | None = None

    # Prose versions for direct-route formatters.
    size_prose: str | None = None
    appearance_prose: str | None = None
    behavior_prose: str | None = None
    notable_prose: str | None = None
    diet_prose: str | None = None
    prose_reviewed: bool = False

    def __post_init__(self) -> None:
        for field_name in ("dangerous_to_humans", "dangerous_to_pets"):
            v = getattr(self, field_name)
            if v is not None and v not in DANGER_LEVELS:
                raise ValueError(
                    f"{field_name}={v!r} not in {DANGER_LEVELS}. "
                    f"Check the BlurbStore column mapping or the source data."
                )


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    # Truncate on a sentence boundary if possible.
    cut = text[: limit].rsplit(". ", 1)
    if len(cut) == 2 and len(cut[0]) > limit // 2:
        return cut[0] + "."
    return text[: limit].rstrip() + "..."


def _format_danger_line(humans: str | None, pets: str | None) -> str | None:
    """
    Render the two danger enums as one prompt line. Used in NON-DANGER
    prompts so the field is available without dominating the answer.
    For DANGER intent, _format_danger_verdict() is used instead.
    """
    if not humans and not pets:
        return None
    if humans and pets and humans == pets:
        return f"- danger: {humans} (humans and pets)"
    parts = []
    if humans:
        parts.append(f"humans: {humans}")
    if pets:
        parts.append(f"pets: {pets}")
    return f"- danger: {', '.join(parts)}"


# Question polarity. Visitors phrase risk questions in two opposite ways:
#   "danger" framing: "is it dangerous?", "will it hurt me?", "venomous?"
#   "safety" framing: "is it safe to eat?", "is it ok to touch?"
# Word "yes" means opposite things in those two framings, and SmolLM2-360M
# does not reliably handle that flip on its own. So we detect polarity and
# pre-render both the verdict prose AND the lead word to match the question.

_SAFETY_KEYWORDS = (
    "safe to", "okay to", "ok to", "alright to", "fine to",
    "edible", "eat ", "eat?", "touch ", "touch?", "handle ", "handle?",
)
_DANGER_KEYWORDS = (
    "danger", "hurt", "harm", "bite", "sting", "venom", "poison",
    "toxic", "kill", "attack", "scared",
)


def _question_polarity(query: str) -> str:
    """Returns 'safety' or 'danger'. Defaults to 'danger' on ambiguity."""
    q = query.lower()
    has_safety = any(k in q for k in _SAFETY_KEYWORDS)
    has_danger = any(k in q for k in _DANGER_KEYWORDS)
    # If both appear ("is it safe or dangerous?") prefer 'danger' framing —
    # it's the more cautious default phrasing.
    if has_safety and not has_danger:
        return "safety"
    return "danger"


# Verdict prose templates. Indexed by (polarity, level) where polarity is
# the question framing and level is the enum value. The prose is written
# so the model can copy it almost verbatim into its answer.
_VERDICT_PHRASE: dict[tuple[str, str], str] = {
    # Danger-framing questions: "is it dangerous?"
    ("danger", "no"):   "is NOT dangerous",
    ("danger", "yes"):  "IS dangerous",
    ("danger", "mild"): "can cause MILD harm",
    # Safety-framing questions: "is it safe?"
    ("safety", "no"):   "IS safe",
    ("safety", "yes"):  "is NOT safe",
    ("safety", "mild"): "is mostly safe but can cause mild harm",
}

# Lead word for the directive, indexed by (polarity, level). This is the
# word the model is told to begin its answer with. It must agree with how
# the question was phrased.
_LEAD_WORD: dict[tuple[str, str], str] = {
    ("danger", "no"):   '"No"',
    ("danger", "yes"):  '"Yes"',
    ("danger", "mild"): '"It can cause mild harm"',
    ("safety", "no"):   '"Yes"',
    ("safety", "yes"):  '"No"',
    ("safety", "mild"): '"Mostly, but"',
}


def _format_danger_verdict(
    humans: str | None,
    pets: str | None,
    polarity: str = "danger",
) -> str | None:
    """
    Render the danger enums as a plain-English VERDICT line. Polarity is
    'safety' or 'danger' — the prose is rephrased to match how the visitor
    asked the question, so the model can paraphrase rather than reason
    about polarity flips.

    Returns None when there's no useful signal to convey.
    """
    h = humans if humans in _VERDICT_PHRASE_LEVELS else None
    p = pets if pets in _VERDICT_PHRASE_LEVELS else None

    if h is None and p is None:
        return None

    # Both known
    if h is not None and p is not None:
        if h == p:
            verb = _VERDICT_PHRASE[(polarity, h)]
            return f"This species {verb} to humans or pets."
        h_clause = _verdict_clause(h, "humans", polarity)
        p_clause = _verdict_clause(p, "pets", polarity)
        return f"This species {h_clause}, but {p_clause}."

    # Only one known
    if h is not None:
        verb = _VERDICT_PHRASE[(polarity, h)]
        return (
            f"This species {verb} to humans. "
            f"Risk to pets is not known."
        )
    verb = _VERDICT_PHRASE[(polarity, p)]
    return (
        f"Risk to humans is not known. "
        f"This species {verb} to pets."
    )


_VERDICT_PHRASE_LEVELS = {"no", "yes", "mild"}  # not "unknown" — no signal


def _verdict_clause(level: str, who: str, polarity: str) -> str:
    """E.g. 'IS dangerous to humans' or 'is NOT safe to humans'."""
    verb = _VERDICT_PHRASE[(polarity, level)]
    return f"{verb} to {who}"


def _danger_directive(
    humans: str | None,
    pets: str | None,
    polarity: str = "danger",
) -> str:
    """
    Build the per-query directive. Polarity-aware: lead word matches the
    question framing so the model doesn't have to flip "yes"/"no" itself.
    """
    h = humans if humans in _VERDICT_PHRASE_LEVELS else None
    p = pets if pets in _VERDICT_PHRASE_LEVELS else None

    if h is None and p is None:
        return (
            'The species facts do not state whether this species is '
            'dangerous. Say plainly that you do not know, and suggest '
            'caution. Answer in ONE short sentence.'
        )

    # Disagreement case: humans and pets have different verdicts. The
    # visitor's question may be specific to one or the other, so we don't
    # pre-declare a lead word. The verdict prose is the model's anchor.
    if h is not None and p is not None and h != p:
        return (
            'The VERDICT above states the risk to humans and pets '
            "separately. Restate it in one sentence using the visitor's "
            'wording. Do not add details not in the VERDICT. '
            'Answer in ONE short sentence.'
        )

    # Agreement (or only one known): pre-declare the lead word.
    level = h if h is not None else p
    lead = _LEAD_WORD[(polarity, level)]

    return (
        f'The VERDICT above is the correct answer. Begin your response '
        f'with {lead} and restate the VERDICT in one sentence. '
        f'Do not add details not in the VERDICT. '
        f'Answer in ONE short sentence.'
    )


# Default rendering order for the non-danger blurb fields. Each entry is
# (label_for_prompt, attribute_on_blurb). Intent routing reorders this so
# the most relevant field appears first.
_FIELD_ORDER = [
    ("appearance",     "appearance"),
    ("size",           "size"),
    ("habitat",        "habitat"),
    ("diet",           "diet"),
    ("behavior",       "behavior"),
    ("notable",        "notable"),
]

# Map from intent to which field should lead. DANGER is handled specially
# below since the danger line spans two attributes. CONSERVATION isn't in
# the schema, so it falls through to default ordering.
_INTENT_LEAD: dict[Intent, str] = {
    Intent.DESCRIPTION:    "appearance",
    Intent.IDENTIFICATION: "appearance",
    Intent.HABITAT:        "habitat",
    Intent.DIET:           "diet",
    Intent.BEHAVIOR:       "behavior",
}


def _format_blurb(
    blurb: Blurb,
    intent: Intent,
    *,
    danger_polarity: str = "danger",
) -> str:
    """
    Format the blurb as a compact key: value block. Lead with the field
    most relevant to the intent so the model attends to it first.

    danger_polarity: only used when intent is DANGER. Determines whether
    the VERDICT block uses "is dangerous" or "is safe" framing.
    """
    # Collect non-empty (label, value) pairs in default order.
    body_pairs: list[tuple[str, str]] = []
    for label, attr in _FIELD_ORDER:
        value = getattr(blurb, attr)
        if value:
            body_pairs.append((label, value.strip()))

    danger_line = _format_danger_line(
        blurb.dangerous_to_humans, blurb.dangerous_to_pets
    )

    # For DANGER intent, the per-line danger field is replaced with a
    # plain-English VERDICT block at the very top. This reads as a
    # directive sentence rather than a labeled value, which 360M-class
    # models follow far more reliably. The other body fields still appear
    # below in their normal order.
    if intent == Intent.DANGER:
        verdict = _format_danger_verdict(
            blurb.dangerous_to_humans,
            blurb.dangerous_to_pets,
            polarity=danger_polarity,
        )
        ordered_body = []
        if verdict:
            # Blank line on either side so the verdict stands out visually
            # in the prompt — small models track section breaks in token
            # space better than they track inline modifiers.
            ordered_body.append(f"VERDICT: {verdict}")
            ordered_body.append("")  # blank separator
        elif danger_line:
            # No clean verdict (everything unknown). Fall back to the line
            # form — at least it's something.
            ordered_body.append(danger_line)
        ordered_body.extend(f"- {label}: {value}" for label, value in body_pairs)
    else:
        lead = _INTENT_LEAD.get(intent)
        if lead is not None:
            body_pairs.sort(key=lambda kv: 0 if kv[0] == lead else 1)
        ordered_body = [f"- {label}: {value}" for label, value in body_pairs]
        if danger_line:
            ordered_body.append(danger_line)

    # Header
    header = f"Species: {blurb.common_name}"
    if blurb.scientific_name:
        header += f" ({blurb.scientific_name})"

    # Apply the per-blurb char budget. Truncate the line that overflows;
    # drop everything after.
    out = [header]
    used = len(header)
    for line in ordered_body:
        if used + len(line) + 1 <= MAX_BLURB_CHARS:
            out.append(line)
            used += len(line) + 1
            continue
        # Try to fit a truncated version of this line.
        label_part, _, value_part = line.partition(":")
        remaining = MAX_BLURB_CHARS - used - len(label_part) - 3
        if remaining < 20:
            break
        truncated = _truncate(value_part.strip(), remaining)
        if truncated:
            out.append(f"{label_part}: {truncated}")
        break
    return "\n".join(out)


def _format_chunks(chunks: Sequence[Chunk]) -> str:
    if not chunks:
        return "(no snippets retrieved)"
    parts = []
    for i, c in enumerate(chunks[:MAX_CHUNKS], 1):
        parts.append(f"[{i}] {_truncate(c.text, MAX_CHARS_PER_CHUNK)}")
    return "\n".join(parts)


def build_prompt(
    *,
    query: str,
    blurb: Blurb,
    chunks: Sequence[Chunk],
    intent_result: IntentResult,
    chat_template: str = "smollm2",
) -> str:
    """
    Assemble the full prompt for the LLM.

    chat_template: which model's chat format to emit. SmolLM2 uses ChatML.
    If you're calling llama.cpp's `--chat-template` or letting it auto-detect
    from the GGUF, you can instead build a `messages` list and let llama.cpp
    apply the template; in that case use `build_messages` below.
    """
    messages = build_messages(
        query=query,
        blurb=blurb,
        chunks=chunks,
        intent_result=intent_result,
    )
    if chat_template == "smollm2":
        return _apply_chatml(messages)
    raise ValueError(f"unknown chat_template: {chat_template}")


def build_messages(
    *,
    query: str,
    blurb: Blurb,
    chunks: Sequence[Chunk],
    intent_result: IntentResult,
) -> list[dict]:
    """
    Return a messages list suitable for llama.cpp's chat completion endpoint
    or for manual templating. Decoupled from the chat template so you can
    swap models without rewriting prompt logic.
    """
    intent = intent_result.intent

    # DANGER intent gets a dynamic directive that depends on the species'
    # actual risk profile AND the polarity of the visitor's question
    # ("is it dangerous?" vs "is it safe?"). Both the verdict prose and
    # the directive's lead word are rephrased to match the question, so
    # the model can paraphrase rather than reason about polarity flips.
    if intent == Intent.DANGER:
        polarity = _question_polarity(query)
        task_line = _danger_directive(
            blurb.dangerous_to_humans,
            blurb.dangerous_to_pets,
            polarity=polarity,
        )
        blurb_block = _format_blurb(blurb, intent, danger_polarity=polarity)
    else:
        task_line = TASK_INSTRUCTIONS.get(
            intent, TASK_INSTRUCTIONS[Intent.OTHER]
        )
        blurb_block = _format_blurb(blurb, intent)

    chunks_block = _format_chunks(chunks)

    user_content = (
        f"SPECIES FACTS:\n{blurb_block}\n\n"
        f"SNIPPETS:\n{chunks_block}\n\n"
        f"VISITOR QUESTION: {query}\n\n"
        f"TASK: {task_line}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _apply_chatml(messages: list[dict]) -> str:
    """ChatML format used by SmolLM2."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---- smoke test ----
if __name__ == "__main__":
    from intent import IntentResult

    blurb = Blurb(
        common_name="Southern Pacific Rattlesnake",
        scientific_name="Crotalus oreganus helleri",
        appearance="A medium-sized pit viper with a triangular head and segmented rattle. Diamond pattern fading toward the tail.",
        size="80-130 cm long",
        habitat="Coastal sage scrub, chaparral, and rocky hillsides across Southern California.",
        diet="Small mammals, lizards, and birds, ambushed and subdued with venom.",
        behavior="Mostly crepuscular; coils and rattles when threatened.",
        dangerous_to_humans="yes",
        dangerous_to_pets="yes",
        notable="Responsible for most envenomations in San Diego County.",
    )
    chunks = [
        Chunk(text="The Southern Pacific rattlesnake is responsible for most "
                   "envenomations in San Diego County.", score=0.71),
        Chunk(text="Rattlesnakes typically give an audible warning before "
                   "striking, but not always.", score=0.62),
    ]
    ir = IntentResult(
        intent=Intent.DANGER,
        confidence="high",
        score=0.72,
        margin=0.15,
        all_scores={},
    )
    print(build_prompt(query="is this snake dangerous?",
                       blurb=blurb, chunks=chunks, intent_result=ir))