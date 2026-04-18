"""Interactive demo for the Eco Tour Guide.

Start the tour:
    python demo.py                           # empty tour, interactive REPL
    python demo.py <image_path>              # open with a first sighting

At the >> prompt:
    see <image_path>                         — run classifier, show to ranger
    see <image_path> <scientific> [common]   — skip classifier
    quit                                     — exit
    <anything else>                          — ask the ranger a question

Requires ANTHROPIC_API_KEY in TourGuide_Agent/.env (or the environment).
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from tour_guide import Observation, TourSession


HELP = """Commands:
  see <image>                       — show the ranger an image (runs the classifier)
  see <image> "<scientific>" ["<common>"] — skip the classifier (quote multi-word names)
  quit                              — exit
  anything else                     — ask the ranger a question
"""


def main(argv: list[str]) -> int:
    session = TourSession()
    classifier = None

    def ensure_classifier():
        nonlocal classifier
        if classifier is None:
            print("Loading classifier (first time only)...")
            from classifier import SpeciesClassifier
            classifier = SpeciesClassifier.load()
            print(f"Classifier ready on {classifier.device}.\n")
        return classifier

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
            predictions = clf.predict(image_path, top_k=3)
            print("Top-3:")
            for p in predictions:
                tag = f" — {p.common_name}" if p.common_name else ""
                print(f"  {p.probability:6.1%}  {p.scientific_name}{tag}")
            scientific = predictions[0].scientific_name
            common = predictions[0].common_name

        obs = Observation(image_path=image_path, scientific_name=scientific, common_name=common)
        print()
        print(session.see(obs))
        print()

    print(f"Ranger at {session.location}, ready.\n")
    print(HELP)

    # Optional bootstrap sighting from argv.
    if len(argv) > 1:
        do_see(argv[1:])

    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
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
            print(session.ask(line))
            print()

    if session.species_seen:
        print(f"Tour covered: {', '.join(session.species_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
