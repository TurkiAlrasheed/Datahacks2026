#!/usr/bin/env python3
"""
Benchmark whisper.cpp speech-to-text on the device it will run on (the UNO Q).

Times the exact call the voice loop makes: pywhispercpp's Model.transcribe()
on a float32 array (transcribe() in full_roboranger_run.py). For each
(--threads, --audio-ctx) configuration it loads a fresh model and reports:
    load     ggml file -> whisper context
    first    the first transcribe after load. The runtime never warms
             Whisper, so a visitor's first question pays this.
    steady   p50 / min / max over --runs further calls
    RTF      steady p50 / clip length (below 1.0 = faster than real time)
    heard    the transcript, so a faster setting that changes it shows up

whisper.cpp pads every clip to a 30 s window, so the encoder costs about the
same for a 2 s question as for a 10 s one. --audio-ctx shrinks that window
(about 50 per second of audio; 1500 = the full 30 s, 0 = whisper's default),
which is the main speed lever for short push-to-talk questions. Only adopt a
value whose transcript still matches.

Usage on the device (from the repo root):
    python3 species_identification/tests/bench_stt.py
    python3 species_identification/tests/bench_stt.py \\
        --whisper-model models/ggml-tiny.en-q5_1.bin \\
        --wav species_identification/tts-and-stt/test.wav \\
        --threads 1,2,4 --runs 5 --audio-ctx 0,512,256

    # record a question from the default mic instead (and keep it):
    python3 species_identification/tests/bench_stt.py --record 4 --save q.wav
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    from pywhispercpp.model import Model as WhisperModel
except ImportError:
    print("error: pywhispercpp not installed — pip install pywhispercpp",
          file=sys.stderr)
    raise

_ROOT = Path(__file__).resolve().parents[2]
_STT_DIR = _ROOT / "species_identification" / "tts-and-stt"
# First existing wins: repo-root models/ is the device layout
# (onDeviceAgentREADME.md), tts-and-stt/models/ the laptop one.
DEFAULT_MODELS = [
    _ROOT / "models" / "ggml-tiny.en-q5_1.bin",
    _STT_DIR / "models" / "ggml-tiny.en-q5_1.bin",
]
DEFAULT_WAV = _STT_DIR / "test.wav"

SAMPLE_RATE = 16000   # whisper.cpp wants 16kHz mono
CTX_PER_SEC = 50      # whisper encoder frames per second of audio


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1e6


def load_wav(path: Path) -> np.ndarray:
    """16-bit PCM WAV -> float32 mono in [-1, 1] at 16 kHz."""
    with wave.open(str(path), "rb") as w:
        channels, width = w.getnchannels(), w.getsampwidth()
        rate, raw = w.getframerate(), w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: need 16-bit PCM, got {8 * width}-bit")
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        print(f"note: resampling {rate} Hz -> {SAMPLE_RATE} Hz (linear)")
        n_out = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio)
    return (audio / 32768.0).astype(np.float32)


def record(seconds: float, save: Path | None) -> np.ndarray:
    """Record from the default input device, the same one the runtime uses."""
    import sounddevice as sd

    input(f"[enter, then speak for {seconds:g}s] ")
    print("  recording...", end=" ", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("done")
    audio = audio.flatten()
    if save:
        with wave.open(str(save), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        print(f"  saved {save}")
    return audio.astype(np.float32) / 32768.0


def transcribe(model: WhisperModel, audio: np.ndarray) -> tuple[float, str]:
    t = time.perf_counter()
    segments = model.transcribe(audio)
    return time.perf_counter() - t, " ".join(s.text for s in segments).strip()


def bench(model_path: Path, audio: np.ndarray, threads: int, audio_ctx: int,
          runs: int) -> dict:
    # audio_ctx=0 is whisper's default; leave it out so that configuration
    # is the runtime's WhisperModel(path, n_threads=...). print_progress
    # only silences whisper.cpp's "Progress: 0%" lines.
    params = {"n_threads": threads, "print_progress": False}
    if audio_ctx:
        params["audio_ctx"] = audio_ctx
    t = time.perf_counter()
    model = WhisperModel(str(model_path), redirect_whispercpp_logs_to=None,
                         **params)
    load_s = time.perf_counter() - t
    rss = _rss_mb()
    try:
        first_s, text = transcribe(model, audio)
        steady = [transcribe(model, audio)[0] for _ in range(runs)]
    finally:
        del model
        gc.collect()
    return {"threads": threads, "audio_ctx": audio_ctx, "load_s": load_s,
            "first_s": first_s, "p50_s": statistics.median(steady),
            "min_s": min(steady), "max_s": max(steady), "text": text,
            "rss_mb": rss}


def _ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper-model", type=Path, default=None,
                    help="whisper.cpp ggml model (default: "
                         "ggml-tiny.en-q5_1.bin under models/ or "
                         "tts-and-stt/models/)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--wav", type=Path, default=None,
                     help=f"16-bit PCM WAV to transcribe (default: "
                          f"{DEFAULT_WAV.relative_to(_ROOT).as_posix()})")
    src.add_argument("--record", type=float, metavar="SECONDS", default=None,
                     help="record this many seconds from the mic instead")
    ap.add_argument("--save", type=Path, default=None,
                    help="with --record: also write the clip to this WAV")
    ap.add_argument("--threads", default="1,2,4",
                    help="comma list of whisper.cpp thread counts "
                         "(the runtime default is 4)")
    ap.add_argument("--audio-ctx", default="0",
                    help="comma list of encoder context sizes; 0 = whisper's "
                         "default (full 30 s window), which the runtime uses")
    ap.add_argument("--runs", type=int, default=5,
                    help="timed calls per configuration after the first")
    args = ap.parse_args()

    model_path = args.whisper_model or next(
        (p for p in DEFAULT_MODELS if p.is_file()), DEFAULT_MODELS[0])
    if not model_path.is_file():
        print(f"error: whisper model {model_path} not found; pass "
              f"--whisper-model", file=sys.stderr)
        return 1
    if args.record is not None:
        audio = record(args.record, args.save)
        source = f"mic, {args.record:g}s"
    else:
        wav = args.wav or DEFAULT_WAV
        if not wav.is_file():
            print(f"error: {wav} not found; pass --wav or --record",
                  file=sys.stderr)
            return 1
        audio = load_wav(wav)
        source = wav.name
    clip_s = len(audio) / SAMPLE_RATE
    if clip_s < 0.3:
        print("error: clip is shorter than 0.3 s", file=sys.stderr)
        return 1

    print(f"host     {platform.node()}  {platform.machine()}  "
          f"{platform.system()} {platform.release()}  "
          f"({os.cpu_count()} cpus)")
    print(f"python   {platform.python_version()}")
    print(f"whisper  {WhisperModel.system_info().strip()}")
    print(f"model    {model_path}  "
          f"({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"audio    {source}: {clip_s:.2f}s "
          f"(audio_ctx >= {math.ceil(clip_s * CTX_PER_SEC)} covers it)")

    results = []
    for ctx in _ints(args.audio_ctx):
        for threads in _ints(args.threads):
            print(f"\n>>> threads={threads}  "
                  f"audio_ctx={ctx or 'default'} ...", flush=True)
            r = bench(model_path, audio, threads, ctx, args.runs)
            results.append(r)
            print(f"    load {r['load_s']:.2f}s   first {r['first_s']:.2f}s"
                  f"   steady {r['p50_s']:.2f}s (min {r['min_s']:.2f}, "
                  f"max {r['max_s']:.2f})   RTF {r['p50_s'] / clip_s:.2f}"
                  + (f"   rss {r['rss_mb']:.0f}MB" if r["rss_mb"] else ""))
            print(f"    heard: {r['text']!r}")

    # Judge every transcript against the runtime's setting (default ctx).
    baseline = next((r["text"] for r in results if r["audio_ctx"] == 0),
                    results[0]["text"])
    print(f"\n=== summary ({source}, {clip_s:.2f}s; p50 of {args.runs}) ===")
    print(f"  {'threads':>7s} {'audio_ctx':>9s} {'load':>7s} {'first':>7s} "
          f"{'steady':>7s} {'RTF':>5s}  transcript")
    for r in sorted(results, key=lambda r: r["p50_s"]):
        print(f"  {r['threads']:>7d} {r['audio_ctx'] or 'default':>9} "
              f"{r['load_s']:>6.2f}s {r['first_s']:>6.2f}s "
              f"{r['p50_s']:>6.2f}s {r['p50_s'] / clip_s:>5.2f}  "
              + ("same" if r["text"] == baseline else "DIFFERS"))
    best = min(results, key=lambda r: r["p50_s"])
    print(f"\n  fastest: threads={best['threads']}, "
          f"audio_ctx={best['audio_ctx'] or 'default'} at "
          f"{best['p50_s']:.2f}s for {clip_s:.2f}s of speech. STT runs once "
          f"per question,\n  on the critical path before the pipeline, so "
          f"this adds directly to every answer's latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
