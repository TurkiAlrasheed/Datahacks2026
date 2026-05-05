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


@dataclass
class Blurb:
    """
    Structured species blurb. Mirrors the YAML schema from build_blurbs.py.
    Only the fields actually used in the prompt need values; the rest can be
    None or empty.
    """
    common_name: str
    scientific_name: str | None = None
    description: str | None = None
    habitat: str | None = None
    diet: str | None = None
    behavior: str | None = None
    danger: str | None = None       # venom, defensive behavior, allergens
    identification: str | None = None
    conservation: str | None = None


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


def _format_blurb(blurb: Blurb, intent: Intent) -> str:
    """
    Format the blurb as a compact key: value block. Lead with the field most
    relevant to the intent so the model attends to it first.
    """
    field_order = [
        ("description", blurb.description),
        ("habitat", blurb.habitat),
        ("diet", blurb.diet),
        ("behavior", blurb.behavior),
        ("danger", blurb.danger),
        ("identification", blurb.identification),
        ("conservation", blurb.conservation),
    ]

    intent_to_field = {
        Intent.DESCRIPTION: "description",
        Intent.HABITAT: "habitat",
        Intent.DIET: "diet",
        Intent.BEHAVIOR: "behavior",
        Intent.DANGER: "danger",
        Intent.IDENTIFICATION: "identification",
        Intent.CONSERVATION: "conservation",
    }
    lead = intent_to_field.get(intent)
    if lead is not None:
        field_order.sort(key=lambda kv: 0 if kv[0] == lead else 1)

    header = f"Species: {blurb.common_name}"
    if blurb.scientific_name:
        header += f" ({blurb.scientific_name})"

    lines = [header]
    used = 0
    for name, value in field_order:
        if not value:
            continue
        line = f"- {name}: {value.strip()}"
        if used + len(line) > MAX_BLURB_CHARS:
            line = f"- {name}: {_truncate(value, MAX_BLURB_CHARS - used - len(name) - 4)}"
            if line.strip().endswith(":"):
                break
            lines.append(line)
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


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
    task_line = TASK_INSTRUCTIONS.get(intent, TASK_INSTRUCTIONS[Intent.OTHER])

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
        description="A medium-sized pit viper with a triangular head and segmented rattle.",
        habitat="Coastal sage scrub, chaparral, and rocky hillsides across Southern California.",
        diet="Small mammals, lizards, and birds, ambushed and subdued with venom.",
        behavior="Mostly crepuscular; coils and rattles when threatened.",
        danger="Venomous. Bites are medical emergencies. Back away slowly; do not attempt to handle.",
        identification="Diamond pattern fading toward the tail, blunt tail with rattle.",
        conservation="Common; not listed.",
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