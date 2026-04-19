"""Eco Tour Guide agent for the DataHacks 2026 project.

Two entry points:

- `narrate(Observation)` — one-shot ranger narration for a single sighting,
  no memory. Useful for smoke tests.
- `TourSession` — stateful multi-turn tour. Call `session.see(observation)`
  when the visitor points the device at a plant or animal, and `session.ask(text)`
  when they speak in natural language. The session remembers every species
  shown so far, skips re-narration on repeats (offering to retell instead),
  and can draw connections between current and past species when relevant.

The ranger persona is location-general. When a session is created without an
explicit location, it detects the current place via `location.get_location()`
(GPS if available, IP-based fallback) and injects the place name + coordinates
into the model's system context on every turn so narration stays grounded
wherever the device is being used.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic

from location import Location, get_location


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


RANGER_SYSTEM_PROMPT = """You are a knowledgeable naturalist and tour guide. A visitor is exploring the outdoors with a camera and a device that identifies the plants and animals they point it at. You respond warmly and specifically to whatever they show you or ask you.

The tour's current location, coordinates, date, and season are provided in the session context. Use them: tie your observations to that specific place — its climate, habitat, typical species assemblages, recent natural and human history, and what is happening there at this time of year. If the location is "Unknown location", say so briefly and fall back to general facts about the species without inventing a setting.

How to respond depends on what the visitor does.

NEW SIGHTING — a species they have not shown you yet on this tour. Give a two-part narration, separated by a single blank line. No headings, bullets, or markdown.
Part 1 (roughly 120–180 words): introduce the species by common name, share a few vivid, specific facts about how it lives (diet, behavior, size, lifespan, distinctive traits), and tie those facts to the current location's ecosystem and to what is happening there at this time of year. Speak directly to the visitor ("notice how…"). Ground your observations in the image when you can. Keep it warm and conversational — like a real ranger, not a textbook.
Part 2 (roughly 40–70 words): specific, concrete threats facing this species and the habitat where the visitor is standing — e.g. warming or acidifying water, pollution, plastic or stormwater runoff, human disturbance, habitat loss, fisheries bycatch, disease, invasive competitors. Be species-specific and place-specific, not generic environmentalism.

REPEAT SIGHTING — a species already covered earlier on this tour. Do NOT re-narrate. Briefly acknowledge the sighting ("ah, another [common name] — we looked at these earlier") and ask if they would like to hear about it again. Two or three sentences. No Part 2.

QUESTION — free-form text from the visitor, not a sighting. Answer conversationally, 2–4 sentences typically. No two-part format, no headings, no markdown. If the question naturally connects to a species already discussed on this tour, weave that connection in (e.g., "like the cormorants we saw earlier, …"). Don't force connections — only make them when they are genuinely interesting or relevant.

YES TO REPEAT — the visitor asks to hear about a species again. Give the full two-part narration, emphasizing different facts than last time.

UNIDENTIFIED IMAGE — a user turn may include one or more images prefixed by a bracketed note that the classifier did not recognize the subject (scenery, an empty view, or something outside the device's catalog for this location). Do NOT invent a species. If the visitor asks about that image, comment briefly on the scene based on what you see, acknowledge that it's not in the catalog, and keep it short and warm — 2 or 3 sentences. If the current message is a new identified sighting, focus on that and ignore the unidentified image unless there's a genuinely relevant connection.

If the identified species seems geographically implausible for the current location, or inconsistent with what's in the image, trust the image: mention the uncertainty briefly in one sentence, then narrate what you actually see. Never invent facts you are not confident about."""


@dataclass
class Observation:
    """What the classifier produced, plus when it was taken."""

    image_path: str | Path
    scientific_name: str
    common_name: str | None = None
    when: datetime | None = None


_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# def _season(dt: datetime, lat: float | None = None) -> str:
#     """Season at `dt` accounting for hemisphere.

#     Defaults to the northern-hemisphere mapping when latitude is unknown.
#     Tropical specificity (wet/dry) is left to the model to handle from the
#     coordinates it also sees.
#     """
#     m = dt.month
#     northern = lat is None or lat >= 0
#     if northern:
#         if m in (12, 1, 2):
#             return "winter"
#         if m in (3, 4, 5):
#             return "spring"
#         if m in (6, 7, 8):
#             return "summer"
#         return "fall"
#     if m in (12, 1, 2):
#         return "summer"
#     if m in (3, 4, 5):
#         return "fall"
#     if m in (6, 7, 8):
#         return "winter"
#     return "spring"


def _species_key(scientific_name: str) -> str:
    """Genus + species, lowercased.

    Drops any subspecies so the same bird identified as the full species
    once and a local subspecies the next time is treated as a repeat.
    """
    parts = scientific_name.split()
    return " ".join(parts[:2]).lower()


def _encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix not in _MEDIA_TYPES:
        raise ValueError(f"Unsupported image extension: {suffix}")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return _MEDIA_TYPES[suffix], data


def _image_block(path: Path) -> dict:
    media_type, image_b64 = _encode_image(path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_b64,
        },
    }


def _coerce_location(location: str | Location | None) -> Location:
    """Normalize the session's location argument into a Location."""
    if isinstance(location, Location):
        return location
    if isinstance(location, str):
        return Location(display_name=location)
    return get_location()


def narrate(
    observation: Observation,
    *,
    location: str | Location | None = None,
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
) -> str:
    """One-shot narration for a single sighting. No memory of prior sightings.

    `location` is detected automatically if omitted. Use `TourSession` for an
    interactive tour with memory.
    """
    client = client or anthropic.Anthropic()
    loc = _coerce_location(location)
    when = observation.when or datetime.now()
    common = observation.common_name or "(common name unknown — use your best knowledge of the scientific name)"
    coords = f" ({loc.lat:.5f}, {loc.lon:.5f})" if loc.has_coords else ""
    user_context = (
        f"Location: {loc.display_name}{coords}\n"
        f"Date: {when.strftime('%B %d, %Y')} (Spring)\n"
        f"Vision model identified: {observation.scientific_name} — {common}\n\n"
        "New sighting. Give your two-part tour guide narration."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": RANGER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(Path(observation.image_path)),
                    {"type": "text", "text": user_context},
                ],
            }
        ],
    )

    return "".join(b.text for b in response.content if b.type == "text")


@dataclass
class _SeenSpecies:
    scientific_name: str
    common_name: str | None
    first_narration: str


class TourSession:
    """Stateful tour with memory of every species the visitor has seen.

    Call `see(observation)` for a sighting (the agent narrates, or acknowledges
    a repeat and asks if they want to hear it again). Call `ask(text)` for any
    free-form question — the agent answers conversationally in the context of
    everything discussed so far on this tour.

    Location and date are session-level and passed to the model so it can tie
    facts to season. When `location` is omitted the session auto-detects via
    `location.get_location()` — GPS if available, IP-based fallback otherwise.
    Message history grows across turns so the agent knows which species have
    been covered.
    """

    def __init__(
        self,
        *,
        location: str | Location | None = None,
        when: datetime | None = None,
        model: str = DEFAULT_MODEL,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.location: Location = _coerce_location(location)
        self.when = when or datetime.now()
        self.model = model
        self.messages: list[dict] = []
        self._species: dict[str, _SeenSpecies] = {}
        self._pending_silent_images: list[dict] = []

    @property
    def species_seen(self) -> list[str]:
        """Scientific names of species covered so far, in order first seen."""
        return [s.scientific_name for s in self._species.values()]

    def look_at(self, image_path: str | Path) -> None:
        """Record an image without narrating.

        Use when the classifier is not confident enough to commit to a
        species (scenery, out-of-catalog subject). The image is buffered
        and attached to the next `see()` or `ask()` with a note that no
        species was identified, so the ranger can comment on it if asked.
        No API call happens here — the agent stays silent until prompted.
        """
        self._pending_silent_images.append(_image_block(Path(image_path)))

    def see(self, observation: Observation) -> str:
        """Record a sighting and get the ranger's response."""
        key = _species_key(observation.scientific_name)
        is_repeat = key in self._species
        common = observation.common_name
        display = f"{common} ({observation.scientific_name})" if common else observation.scientific_name

        if is_repeat:
            user_text = (
                f"I'm looking at another {display}. "
                "This is the same species we already covered earlier on this tour."
            )
        else:
            user_text = (
                f"I'm now looking at: {display}. "
                "The vision model identified it. New sighting."
            )

        content = self._flush_silent() + [
            _image_block(Path(observation.image_path)),
            {"type": "text", "text": user_text},
        ]
        self.messages.append({"role": "user", "content": content})

        reply = self._send()

        if not is_repeat:
            self._species[key] = _SeenSpecies(
                scientific_name=observation.scientific_name,
                common_name=common,
                first_narration=reply,
            )
        return reply

    def ask(self, question: str) -> str:
        """Ask the ranger a free-form question."""
        content = self._flush_silent() + [{"type": "text", "text": question}]
        self.messages.append({"role": "user", "content": content})
        return self._send()

    def _flush_silent(self) -> list[dict]:
        """Pop any buffered silent images into a content-block prefix."""
        if not self._pending_silent_images:
            return []
        images = self._pending_silent_images
        self._pending_silent_images = []
        label = "image" if len(images) == 1 else "images"
        note = (
            f"[Earlier I pointed the camera at the {label} above, but our species "
            "classifier didn't recognize the subject — likely scenery or something "
            "outside the device's catalog for this location.]"
        )
        return [*images, {"type": "text", "text": note}]

    def _tour_context_block(self) -> str:
        loc = self.location
        coords = f" (lat {loc.lat:.5f}, lon {loc.lon:.5f})" if loc.has_coords else ""
        region = f" Region: {loc.region}." if loc.region else ""
        country = f" Country: {loc.country}." if loc.country else ""
        return (
            f"Tour info — Location: {loc.display_name}{coords}.{region}{country} "
            f"Date: {self.when.strftime('%B %d, %Y')} (Spring)."
        )

    def _send(self) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": RANGER_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": self._tour_context_block()},
            ],
            messages=self.messages,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        self.messages.append({"role": "assistant", "content": text})
        return text
