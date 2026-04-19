"""Interactive voice-first demo for the Eco Tour Guide.

Start the tour:
    python demo.py                           # empty tour, voice + text REPL
    python demo.py <image_path>              # open with a first sighting
    python demo.py --no-voice                # disable voice I/O
    python demo.py --no-voice <image_path>   # both

At the >> prompt:
    <Enter>                                  — speak your question (press Enter again to stop)
    see <image>                              — run classifier, show to ranger
    see <image> "<scientific>" ["<common>"]  — skip the classifier (quote multi-word names)
    quit                                     — exit
    <anything else>                          — type a question (speech synthesized in reply)

Voice I/O uses ElevenLabs (STT + TTS). Requires ANTHROPIC_API_KEY and
ELEVEN_LABS_API_KEY in TourGuide_Agent/.env (or the environment). Voice is
disabled automatically when stdin is not a terminal (piped / heredoc input).
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from tour_guide import Observation, TourSession


# Top-1 softmax probability below this is treated as "no animal detected".
# Correctly-classified animals in our validation set score 90%+, so 0.7 leaves
# plenty of headroom. OOD inputs (scenery, off-catalog subjects) either fall
# well below this — or if they sneak above, the ranger's system prompt tells
# it to trust the image and refuse to narrate a species it can't actually see.
CONFIDENCE_THRESHOLD = 0.7

# `seashore` is a negative class — scenery with no catalogued species.
# Treat a top prediction of `seashore` identically to a low-confidence result.
NEGATIVE_CLASSES = {"seashore"}


HELP = """Commands:
  <Enter>                           — speak your question (press Enter again to stop)
  see <image>                       — show the ranger an image (runs the classifier)
  see <image> "<scientific>" ["<common>"] — skip the classifier (quote multi-word names)
  quit                              — exit
  anything else                     — type a question to the ranger

While the ranger is speaking, press any key to cut them off and ask a new
question or show a new image. (On non-Windows terminals, press Enter.)
"""


def _init_voice(enabled: bool):
    """Return a VoiceIO instance, or None if voice is disabled or unavailable."""
    if not enabled:
        return None
    try:
        from voice import VoiceIO
        return VoiceIO()
    except Exception as e:
        print(f"[voice disabled: {e}]")
        return None


def main(argv: list[str]) -> int:
    # Strip --no-voice flag out of argv so the rest parses cleanly.
    voice_requested = sys.stdin.isatty()
    cleaned: list[str] = []
    for arg in argv[1:]:
        if arg == "--no-voice":
            voice_requested = False
        elif arg == "--voice":
            voice_requested = True
        else:
            cleaned.append(arg)

    voice = _init_voice(voice_requested)
    print("Detecting location (GPS → IP)...")
    session = TourSession()
    classifier = None

    def ensure_classifier():
        nonlocal classifier
        if classifier is None:
            from classifier import load_classifier
            remote = os.environ.get("CLASSIFIER_URL")
            print(
                "Connecting to remote classifier..." if remote
                else "Loading classifier (first time only)..."
            )
            classifier = load_classifier()
            print(f"Classifier ready on {classifier.device}.\n")
        return classifier

    def respond(text: str) -> None:
        """Print the ranger's reply and speak it if voice is on."""
        print(text)
        print()
        if voice is not None:
            if voice.speak(text):
                print("[interrupted — go ahead]")

    def do_see(args: list[str]) -> None:
        if not args:
            print("Usage: see <image_path> [scientific_name [common_name]]")
            return
        image_path = Path(args[0])
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return

        if len(args) >= 2:
            scientific = args[1]
            common = args[2] if len(args) > 2 else None
        else:
            clf = ensure_classifier()
            try:
                predictions = clf.predict(image_path, top_k=3)
            except Exception as err:
                # Remote API down / timeout / bad image / anything else. Treat the
                # frame as unidentified so the tour continues: the image is buffered
                # silently and the ranger stays quiet until asked about it.
                print(f"\nClassifier error ({type(err).__name__}: {err}).")
                print("Treating this as an unidentified image. The ranger will wait — ask a question if you'd like them to comment on the scene.\n")
                session.look_at(image_path)
                return
            print("Top-3:")
            for p in predictions:
                tag = f" — {p.common_name}" if p.common_name else ""
                print(f"  {p.probability:6.1%}  {p.scientific_name}{tag}")
            top = predictions[0]
            is_negative = top.scientific_name.lower() in NEGATIVE_CLASSES
            if is_negative or top.probability < CONFIDENCE_THRESHOLD:
                if is_negative:
                    print(f"\nNo animal detected (top prediction is '{top.scientific_name}', a scenery class).")
                else:
                    print(f"\nNo animal detected (top prediction {top.probability:.1%} < threshold {CONFIDENCE_THRESHOLD:.0%}).")
                print("The ranger will wait. Ask a question if you'd like them to comment on the scene.\n")
                session.look_at(image_path)
                return
            scientific = top.scientific_name
            common = top.common_name

        obs = Observation(image_path=image_path, scientific_name=scientific, common_name=common)
        print()
        respond(session.see(obs))

    loc = session.location
    coords = f" ({loc.lat:.4f}, {loc.lon:.4f})" if loc.has_coords else ""
    print(f"Ranger at {loc.display_name}{coords}, ready. Voice {'ON' if voice else 'off'}.\n")
    print(HELP)

    # Optional bootstrap sighting from argv.
    if cleaned:
        do_see(cleaned)

    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        # Empty Enter → voice turn (if enabled).
        if not line:
            if voice is None:
                continue
            question = voice.listen()
            if not question:
                print("[nothing heard]")
                continue
            print(f"You: {question}\n")
            respond(session.ask(question))
            continue

        if line.lower() in {"quit", "exit", "/quit", "/exit"}:
            break
        if line.lower() == "help":
            print(HELP)
            continue

        lower = line.lower()
        if lower == "see" or lower.startswith("see "):
            do_see(shlex.split(line[3:].strip()))
        else:
            print()
            respond(session.ask(line))

    if session.species_seen:
        print(f"Tour covered: {', '.join(session.species_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
