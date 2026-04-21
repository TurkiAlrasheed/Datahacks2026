"""Generate a downloadable audio file from text using ElevenLabs TTS.

Usage:
    python generate_audio.py "Hello, welcome to La Jolla Cove!"
    python generate_audio.py "Hello!" --output greeting.mp3
    python generate_audio.py "Hello!" --voice nPczCjzI2devNBz1zQrb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from elevenlabs.client import ElevenLabs

DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # "Brian"
DEFAULT_MODEL = "eleven_turbo_v2_5"


def main():
    parser = argparse.ArgumentParser(description="Generate audio from text via ElevenLabs")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("--output", "-o", default="output.mp3", help="Output filename (default: output.mp3)")
    parser.add_argument("--voice", default=DEFAULT_VOICE_ID, help=f"ElevenLabs voice ID (default: {DEFAULT_VOICE_ID})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"TTS model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("ELEVEN_LABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVEN_LABS_API_KEY not set in environment or .env file")
        return 1

    client = ElevenLabs(api_key=api_key)

    fmt = "mp3_44100_128"
    if args.output.endswith(".wav"):
        fmt = "pcm_22050"

    print(f"Generating audio for: \"{args.text[:80]}{'...' if len(args.text) > 80 else ''}\"")
    stream = client.text_to_speech.convert(
        voice_id=args.voice,
        text=args.text,
        model_id=args.model,
        output_format=fmt,
    )
    audio_bytes = b"".join(stream)

    if args.output.endswith(".wav"):
        import struct
        sr, ch, bps = 22050, 1, 16
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + len(audio_bytes), b"WAVE",
            b"fmt ", 16, 1, ch, sr, sr * ch * bps // 8,
            ch * bps // 8, bps, b"data", len(audio_bytes),
        )
        audio_bytes = header + audio_bytes

    Path(args.output).write_bytes(audio_bytes)
    print(f"Saved to {args.output} ({len(audio_bytes) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
