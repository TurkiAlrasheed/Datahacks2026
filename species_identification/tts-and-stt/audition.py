"""
audition.py — listen to RoboRanger direct-route answers through piper TTS.

Two source modes, since the iteration loop and the deployment check have
different needs:

  --source yaml (default): read blurbs.yaml directly. Edit YAML, re-run,
      listen. No recompile in the middle. Use this while building the
      prose-editing backlog.

  --source db: read corpus.db through BlurbStore — exactly what the runtime
      reads. Use this before deploying, to confirm what the device will
      actually speak. Requires sqlite-vec because BlurbStore opens the
      same connection the retriever uses.

Usage:
    # Fast iteration loop — YAML, all six direct-route intents
    python audition.py Buteo_lineatus

    # Multiple species, paged with prompts between
    python audition.py Buteo_lineatus Salvia_mellifera Marah_macrocarpa

    # Random sample
    python audition.py --random 12

    # Spot-check a single intent across many species
    python audition.py --intent size --random 15

    # Text only, no piper — fast scan for prose bugs
    python audition.py --no-audio --random 41 > audition_scan.txt

    # Save WAVs instead of playing live
    python audition.py --save audio_out/ --random 10

    # Verify what the deployed runtime will actually speak
    python audition.py --source db --db corpus.db Buteo_lineatus

Requirements:
    pip install pyyaml
    piper binary on PATH (https://github.com/rhasspy/piper)
    A piper voice .onnx (+ .onnx.json sibling), via --voice or PIPER_VOICE.
    For --source db: sqlite-vec must be importable.

The script imports the real RoboRangerPipeline formatter methods, so what
you hear is byte-identical to what the runtime speaks. Refactor the
formatters and audition tracks automatically — no parallel templating to
keep in sync.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(1, "../pipeline")
sys.path.insert(2, "../llm-tuning")
from pipeline import (
    RoboRangerPipeline,
    DIRECT_ROUTE_INTENTS,  
)
from prompt_builder import Blurb
from intent import Intent


# Friendly names for CLI --intent and printed headers.
INTENT_LABELS: dict[Intent, str] = {
    Intent.DANGER:      "danger",
    Intent.DIET:        "diet",
    Intent.SIZE:        "size",
    Intent.HABITAT:     "habitat",
    Intent.BEHAVIOR:    "behavior",
    Intent.DESCRIPTION: "description",
}
LABEL_TO_INTENT = {v: k for k, v in INTENT_LABELS.items()}

# Print intents in a stable, predictable order regardless of frozenset
# iteration order. Order chosen to match how a visitor might naturally
# probe: what is it, how big, where does it live, etc.
ORDERED_INTENTS: list[Intent] = [
    Intent.DESCRIPTION,
    Intent.SIZE,
    Intent.HABITAT,
    Intent.DIET,
    Intent.BEHAVIOR,
    Intent.DANGER,
]

# Sanity-check: every direct-route intent must have a label and an order
# slot. If pipeline.py adds a new direct-route intent, this assert is the
# tripwire that tells us to update both maps here.
assert set(ORDERED_INTENTS) == set(DIRECT_ROUTE_INTENTS), (
    f"audition is out of sync with pipeline.DIRECT_ROUTE_INTENTS — "
    f"missing: {DIRECT_ROUTE_INTENTS - set(ORDERED_INTENTS)}, "
    f"extra: {set(ORDERED_INTENTS) - DIRECT_ROUTE_INTENTS}"
)


# ---------------------------------------------------------------------------
# Source adapters: YAML and DB both return {species_id: Blurb}
# ---------------------------------------------------------------------------

def load_blurbs_from_yaml(yaml_path: Path) -> dict[str, Blurb]:
    """Read blurbs.yaml. Top-level envelope is `blurbs:` per the file."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "blurbs" not in data:
        raise ValueError(
            f"{yaml_path}: expected top-level 'blurbs:' key, got "
            f"{list(data)[:5] if isinstance(data, dict) else type(data).__name__}"
        )
    raw_by_id: dict[str, dict] = data["blurbs"]
    return {sid: _blurb_from_yaml(sid, raw) for sid, raw in raw_by_id.items()}


def _blurb_from_yaml(species_id: str, raw: dict) -> Blurb:
    """
    Construct a Blurb dataclass from a YAML dict. The formatters call
    self._name(blurb) which reads blurb.species_id, so we must set that.
    """
    scientific_name = raw.get("scientific_name") or species_id.replace("_", " ")
    blurb_field_names = {f.name for f in fields(Blurb)}

    kwargs = {
        k: v for k, v in raw.items()
        if k in blurb_field_names and k != "scientific_name"
    }

    # Known 'apprearance' typo in some entries — BlurbStore handles it,
    # mirror that behavior here so YAML and DB paths produce the same Blurb.
    if "apprearance" in raw and "appearance" not in kwargs:
        kwargs["appearance"] = raw["apprearance"]

    # species_id is a Blurb field (the formatters read it). Inject it
    # explicitly because the YAML keys it as the outer dict key, not a
    # field inside the entry.
    if "species_id" in blurb_field_names:
        kwargs["species_id"] = species_id

    return Blurb(scientific_name=scientific_name, **kwargs)


def load_blurbs_from_db(db_path: Path) -> dict[str, Blurb]:
    """
    Read every species through BlurbStore — the runtime path. This is the
    pre-deploy verification mode: if it sounds right here, it'll sound
    right on the device.
    """
    import sqlite3
    import sqlite_vec  # required by BlurbStore's shared connection
    from blurb_store import BlurbStore

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    store = BlurbStore(conn)

    # BlurbStore exposes per-species lookup but not a bulk list. The DB
    # has a `species` table (see compile_blurbs.py); query it for ids,
    # then go through the store so the Blurb shape matches what the
    # runtime sees.
    species_ids = [
        r[0] for r in conn.execute("SELECT species_id FROM species").fetchall()
    ]
    return {sid: store.get(sid) for sid in species_ids}


# ---------------------------------------------------------------------------
# Formatter access — call the real pipeline methods so audition can't drift
# ---------------------------------------------------------------------------

def make_formatter() -> RoboRangerPipeline:
    """
    Build a RoboRangerPipeline instance without running __init__. The
    formatter methods only touch _name() and _decap(), neither of which
    reads any state set up in __init__ (embedder, db_conn, llm, etc.),
    so __new__ is safe. If a formatter ever starts reading self.X, this
    is the line that'll start failing.
    """
    return RoboRangerPipeline.__new__(RoboRangerPipeline)


def format_one(pipeline: RoboRangerPipeline, blurb: Blurb,
               intent: Intent) -> str | None:
    return pipeline._format_from_blurb(intent, blurb)


# ---------------------------------------------------------------------------
# Piper invocation
# ---------------------------------------------------------------------------

def speak(text: str, voice: Path, save_to: Path | None) -> None:
    """
    Synthesize `text` with piper. If save_to is given, write a WAV there;
    otherwise play through the system default device.

    piper CLI reads text from stdin and writes WAV to --output_file.
    """
    out_path: Path
    cleanup = False
    if save_to is not None:
        out_path = save_to
    else:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="audition_")
        os.close(fd)
        out_path = Path(tmp)
        cleanup = True

    proc = subprocess.run(
        ["piper", "--model", str(voice), "--output_file", str(out_path)],
        input=text,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"  [piper error] {proc.stderr.strip()}", file=sys.stderr)
        if cleanup:
            out_path.unlink(missing_ok=True)
        return

    if save_to is not None:
        print(f"  saved: {out_path}")
        return

    if sys.platform == "win32":
        _play_windows(out_path)
    else:
        player = _find_player()
        if player is None:
            print(f"  [no player found — wav at {out_path}, play it yourself]")
            return
        subprocess.run(player + [str(out_path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cleanup:
        out_path.unlink(missing_ok=True)


def _find_player() -> list[str] | None:
    """Find a CLI audio player that exists on this system.
    
    Returns argv *prefix* — the WAV path is appended by the caller.
    On Windows, returns None and the caller uses _play_windows() instead,
    because PowerShell's argument-passing is too fiddly for a generic
    'argv + [path]' pattern.
    """
    if sys.platform == "win32":
        return None  # handled separately
    candidates = [
        ["aplay", "-q"],   # Linux ALSA
        ["paplay"],        # Linux PulseAudio
        ["afplay"],        # macOS
    ]
    for cmd in candidates:
        if subprocess.run(["which", cmd[0]],
                          capture_output=True).returncode == 0:
            return cmd
    return None


def _play_windows(wav_path: Path) -> None:
    """Play a WAV synchronously on Windows via PowerShell SoundPlayer."""
    # Build the command as a single string so the path is interpolated
    # at script-construction time, not argument-binding time. Escape
    # single quotes in the path by doubling them (PowerShell convention).
    path_str = str(wav_path.resolve()).replace("'", "''")
    ps_script = (
        f"$p = New-Object Media.SoundPlayer '{path_str}'; "
        f"$p.PlaySync()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# ---------------------------------------------------------------------------
# Per-species audition driver
# ---------------------------------------------------------------------------

def audition_species(
    pipeline: RoboRangerPipeline,
    species_id: str,
    blurb: Blurb,
    intents: list[Intent],
    voice: Path | None,
    save_dir: Path | None,
    no_audio: bool,
) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {species_id}  ({blurb.common_name or blurb.scientific_name})")
    print(f"  prose_reviewed: {getattr(blurb, 'prose_reviewed', False)}")
    print(f"{'=' * 64}")

    for intent in intents:
        label = INTENT_LABELS[intent]
        text = format_one(pipeline, blurb, intent)

        if text is None:
            print(f"\n[{label}] <falls through to LLM — no direct answer>")
            continue

        print(f"\n[{label}]")
        print(f"  {text}")

        if no_audio or voice is None:
            continue

        save_to = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_to = save_dir / f"{species_id}__{label}.wav"

        speak(text, voice, save_to)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Listen to RoboRanger direct-route answers via piper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("species_ids", nargs="*",
                   help="Species IDs to audition. Omit if using --random.")
    p.add_argument("--source", choices=["yaml", "db"], default="yaml",
                   help="Where to read blurbs from. yaml = fast iteration "
                        "(default); db = pre-deploy runtime parity check.")
    p.add_argument("--yaml", default=Path("../blurbs.yaml"), type=Path,
                   help="Path to blurbs.yaml (used when --source yaml).")
    p.add_argument("--db", default=Path("../offline-info/corpus.db"), type=Path,
                   help="Path to corpus.db (used when --source db).")
    p.add_argument("--random", type=int, metavar="N",
                   help="Sample N random species instead of named ones.")
    p.add_argument("--intent", choices=list(INTENT_LABELS.values()),
                   help="Only audition this one intent (default: all six).")
    p.add_argument("--voice", type=Path,
                   default=Path.cwd() / "voices" / "en_US-lessac-medium.onnx",
                   help="Path to piper voice .onnx (or set PIPER_VOICE).")
    p.add_argument("--save", type=Path, metavar="DIR",
                   help="Save WAVs to DIR instead of playing.")
    p.add_argument("--no-audio", action="store_true",
                   help="Print formatter output but don't run piper.")
    p.add_argument("--no-pause", action="store_true",
                   help="Don't prompt between species (good with --save).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Validate piper args up front; loading blurbs takes time and we want
    # the fast failure path.
    if not args.no_audio and not args.voice:
        print("error: --voice or PIPER_VOICE required (or use --no-audio)",
              file=sys.stderr)
        return 1
    if args.voice and not args.voice.exists():
        print(f"error: voice file {args.voice} not found", file=sys.stderr)
        return 1

    # Load blurbs from the requested source.
    if args.source == "yaml":
        if not args.yaml.exists():
            print(f"error: {args.yaml} not found", file=sys.stderr)
            return 1
        blurbs = load_blurbs_from_yaml(args.yaml)
        source_desc = f"YAML ({args.yaml})"
    else:
        if not args.db.exists():
            print(f"error: {args.db} not found", file=sys.stderr)
            return 1
        try:
            blurbs = load_blurbs_from_db(args.db)
        except ImportError as e:
            print(f"error: --source db needs sqlite_vec / blurb_store: {e}",
                  file=sys.stderr)
            return 1
        source_desc = f"DB ({args.db})"

    print(f"loaded {len(blurbs)} species from {source_desc}")

    # Resolve species selection.
    if args.random:
        if args.species_ids:
            print("warning: --random ignores positional species_ids",
                  file=sys.stderr)
        species_ids = random.sample(list(blurbs), min(args.random, len(blurbs)))
    elif args.species_ids:
        species_ids = args.species_ids
    else:
        print("error: provide species_ids or --random N", file=sys.stderr)
        return 1

    missing = [s for s in species_ids if s not in blurbs]
    if missing:
        print(f"error: not in source: {', '.join(missing)}", file=sys.stderr)
        return 1

    # Resolve intents.
    if args.intent:
        intents = [LABEL_TO_INTENT[args.intent]]
    else:
        intents = ORDERED_INTENTS

    pipeline = make_formatter()

    for i, species_id in enumerate(species_ids):
        audition_species(
            pipeline=pipeline,
            species_id=species_id,
            blurb=blurbs[species_id],
            intents=intents,
            voice=args.voice,
            save_dir=args.save,
            no_audio=args.no_audio,
        )
        if (not args.no_pause
                and args.save is None
                and i < len(species_ids) - 1):
            try:
                input("\n  [enter for next species, ctrl-c to quit] ")
            except (KeyboardInterrupt, EOFError):
                print()
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())