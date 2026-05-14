"""
RoboRanger voice loop — push-to-talk STT + TTS wrapped around the pipeline.

This is the audio front-end / back-end. It does NOT touch pipeline.answer():
that stays a pure (species_id, query) -> Response function with no I/O, so
it stays testable and run_pipeline.py keeps working unchanged. All the
microphone / speaker / model-loading I/O lives here, at the edges:

    button down/up  ->  record audio        (STT in)
    whisper          ->  query string
    pipeline.answer  ->  Response            (UNCHANGED core)
    resp.text        ->  piper               (TTS out)

Species selection is still a CLI arg (--species), exactly like
run_pipeline.py. When the MobileNetV3 classifier is wired in later, it
replaces that one arg with a camera read — nothing else in this loop
changes.

Usage:
    # Laptop, push-to-talk REPL against one species
    python voice_loop.py --species Marah_macrocarpa \\
        --voice voices/en_US-lessac-low.onnx \\
        --whisper-model models/ggml-tiny.en-q5_1.bin

    # Uno Q deployment (llama.cpp backend)
    python voice_loop.py --backend llama-cpp \\
        --db corpus.db --species Marah_macrocarpa \\
        --voice voices/en_US-lessac-low.onnx \\
        --whisper-model models/ggml-tiny.en-q5_1.bin

Requirements:
    pip install sounddevice numpy pywhispercpp piper-tts
    A piper voice .onnx (+ .onnx.json sibling).
    A whisper.cpp ggml model (ggml-tiny.en-q5_1.bin recommended).

Press-to-talk prototype: this uses <enter> as the button (press to start
recording, press again to stop). On the device, swap record_utterance()
for a GPIO button-edge implementation — that is the ONLY function that
needs to change for the hardware port.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

# Heavy, optional deps — imported here so a missing dep gives a clear
# message instead of a traceback three frames deep.
try:
    import sounddevice as sd
except ImportError:
    print("error: sounddevice not installed — pip install sounddevice",
          file=sys.stderr)
    raise
try:
    from pywhispercpp.model import Model as WhisperModel
except ImportError:
    print("error: pywhispercpp not installed — pip install pywhispercpp",
          file=sys.stderr)
    raise
try:
    from piper import PiperVoice
except ImportError:
    print("error: piper not installed — pip install piper-tts",
          file=sys.stderr)
    raise

sys.path.insert(1, "species_identification/pipeline")
sys.path.insert(2, "species_identification/llm-tuning")
sys.path.insert(3, "species_identification/tests")
from pipeline_factory import build_pipeline
from pipeline import RoboRangerPipeline, Response


# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000   # whisper.cpp wants 16kHz mono; don't change without
CHANNELS = 1          # also resampling before transcription.
DTYPE = "int16"

# Utterances shorter than this are almost always an accidental
# double-tap of the button. Skip them rather than transcribing 80ms
# of silence (which whisper will happily hallucinate words onto).
MIN_UTTERANCE_SEC = 0.3


# ---------------------------------------------------------------------------
# STT — speech in
# ---------------------------------------------------------------------------

def record_utterance() -> np.ndarray:
    """
    Capture one utterance from the default mic. Returns int16 mono samples
    at SAMPLE_RATE.

    PROTOTYPE: <enter> is the button — press to start, press again to stop.
    THIS is the one function to replace for the hardware port: swap the two
    input() calls for GPIO button-down / button-up edges. Everything
    downstream takes a numpy array and doesn't care where it came from.
    """
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status: {status}]", file=sys.stderr)
        q.put(indata.copy())

    input("  [enter to start speaking] ")
    print("  recording — [enter to stop] ", end="", flush=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback):
        input()  # blocks until the user presses enter again

    chunks: list[np.ndarray] = []
    while not q.empty():
        chunks.append(q.get_nowait())

    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks, axis=0).flatten()


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    """
    Run whisper.cpp on an int16 mono array. pywhispercpp wants float32 in
    [-1, 1], so convert. Returns the joined transcript, stripped.
    """
    audio_f32 = audio.astype(np.float32) / 32768.0
    segments = model.transcribe(audio_f32)
    return " ".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# TTS — speech out
# ---------------------------------------------------------------------------

def speak(voice: PiperVoice, text: str) -> None:
    """
    Synthesize `text` with a pre-loaded PiperVoice and play it through the
    system default output device.

    Synthesizes to a temp WAV then plays it. This is the simple, blocking
    version. Two known future improvements, both deliberately deferred:
      - sentence-level streaming: split text on '. ', synth+play sentence
        one while sentence two synthesizes. Cuts time-to-first-audio.
      - interruptibility: if the button is pressed during playback, kill
        the player subprocess and start recording. The player call below
        is already a subprocess specifically so this is a clean add later.
    """
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="roboranger_")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with wave.open(str(tmp_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        _play_wav(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _play_wav(path: Path) -> None:
    """Play a WAV synchronously. Tries common players per platform."""
    if sys.platform == "win32":
        path_str = str(path.resolve()).replace("'", "''")
        ps = (f"$p = New-Object Media.SoundPlayer '{path_str}'; "
              f"$p.PlaySync()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    candidates = [["aplay", "-q"], ["paplay"], ["afplay"]]
    for cmd in candidates:
        if subprocess.run(["which", cmd[0]],
                          capture_output=True).returncode == 0:
            subprocess.run(cmd + [str(path)], check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return
    print(f"  [no audio player found — wav at {path}]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Latency table — same shape as run_pipeline.py, plus stt/tts rows
# ---------------------------------------------------------------------------

# pipeline stages, in run order, bracketed by the two audio stages this
# loop adds. "total_voice" is button-up to end-of-speech: the number a
# visitor actually feels.
VOICE_STAGE_ORDER = (
    "stt",
    "gate", "blurb", "intent", "format", "retrieval", "prompt", "llm",
    "tts",
    "total_voice",
)


def fmt_ms(seconds: float | None) -> str:
    return "      —" if seconds is None else f"{seconds * 1000:7.1f}ms"


def print_turn(resp: Response, stt_s: float, tts_s: float,
               total_voice_s: float, transcript: str) -> None:
    """Print transcript, answer, and the full voice-inclusive latency table."""
    merged: dict[str, float] = dict(resp.latency)
    merged["stt"] = stt_s
    merged["tts"] = tts_s
    merged["total_voice"] = total_voice_s

    print()
    print("─" * 60)
    print(f"  heard:  {transcript!r}")
    print(f"  answer: {resp.text}")
    print("─" * 60)
    print(f"  path:   {resp.path}")
    if resp.intent is not None:
        print(f"  intent: {resp.intent.intent.name} ({resp.intent.confidence})")
    if resp.error:
        print(f"  error:  {resp.error}")
    print("  latency:")
    for stage in VOICE_STAGE_ORDER:
        marker = "  └─" if stage == "total_voice" else "  │ "
        print(f"  {marker} {stage:12s} {fmt_ms(merged.get(stage))}")
    print()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def voice_repl(pipeline: RoboRangerPipeline, species_id: str,
               whisper_model: WhisperModel, voice: PiperVoice) -> int:
    """
    Push-to-talk loop. One iteration = one utterance = one spoken answer.
    Ctrl-C to exit.

    Species is fixed for the session (CLI arg). When the classifier is
    wired in, this is where a camera read would update species_id —
    either once at the top, or per-turn if the robot is re-pointed.
    """
    print(f"\nvoice loop ready — species: {species_id}")
    print("Ctrl-C to exit.\n")

    while True:
        try:
            # --- STT: speech in -------------------------------------------
            t_stt = time.perf_counter()
            audio = record_utterance()
            transcript = ""
            if len(audio) >= SAMPLE_RATE * MIN_UTTERANCE_SEC:
                transcript = transcribe(whisper_model, audio)
            stt_s = time.perf_counter() - t_stt

            # Empty / too-short / whisper-returned-nothing: don't run the
            # pipeline on garbage. Speak a short reprompt and loop. This is
            # the "I didn't catch that" unhappy path — keep it spoken so
            # the device never just goes silent.
            if not transcript:
                print("  (nothing heard)")
                speak(voice, "I didn't catch that. Press the button and "
                             "try again.")
                continue

            # --- pipeline core: UNCHANGED ---------------------------------
            # answer() does its own per-stage timing into resp.latency.
            resp = pipeline.answer(species_id, transcript)

            # --- TTS: speech out ------------------------------------------
            t_tts = time.perf_counter()
            speak(voice, resp.text)
            tts_s = time.perf_counter() - t_tts

            total_voice_s = stt_s + resp.latency.get("total", 0.0) + tts_s
            print_turn(resp, stt_s, tts_s, total_voice_s, transcript)

        except KeyboardInterrupt:
            print("\nbye")
            return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Pipeline args — kept identical to run_pipeline.py so muscle memory
    # transfers and the two scripts can share command lines.
    p.add_argument("--db",
                   default="species_identification/offline-info/corpus.db",
                   help="path to corpus.db")
    p.add_argument("--backend", choices=("ollama", "llama-cpp"),
                   default="ollama",
                   help="LLM backend (default: ollama for laptop)")
    p.add_argument("--model", default="smollm2:360m",
                   help="model name as the backend understands it")
    p.add_argument("--backend-host", default=None,
                   help="override backend host URL")
    p.add_argument("--threshold", type=float, default=None,
                   help="cosine retrieval threshold (default: from pipeline.py)")
    p.add_argument("--species", required=True,
                   help="species_id (e.g. Crotalus_oreganus_helleri)")
    # Audio args — new.
    p.add_argument("--voice", type=Path, required=True,
                   help="path to piper voice .onnx (+ .onnx.json sibling)")
    p.add_argument("--whisper-model", type=Path, required=True,
                   help="path to whisper.cpp ggml model "
                        "(ggml-tiny.en-q5_1.bin recommended)")
    p.add_argument("--whisper-threads", type=int, default=4,
                   help="threads for whisper.cpp (default: 4)")
    args = p.parse_args()

    # Fail fast on missing files before loading anything heavy.
    if not args.voice.exists():
        print(f"error: voice file {args.voice} not found", file=sys.stderr)
        return 1
    if not args.whisper_model.exists():
        print(f"error: whisper model {args.whisper_model} not found",
              file=sys.stderr)
        return 1

    # Load the two audio models once. Both have meaningful load time;
    # loading per-utterance is exactly the cold-start trap we already
    # fixed for piper. Time them so a slow load is visible.
    print("loading whisper...", end=" ", flush=True)
    t = time.perf_counter()
    whisper_model = WhisperModel(str(args.whisper_model),
                                 n_threads=args.whisper_threads)
    print(f"({time.perf_counter() - t:.1f}s)")

    print("loading piper voice...", end=" ", flush=True)
    t = time.perf_counter()
    try:
        voice = PiperVoice.load(str(args.voice))
    except Exception as e:
        print()
        print(f"error: failed to load voice {args.voice}: {e}",
              file=sys.stderr)
        return 1
    print(f"({time.perf_counter() - t:.1f}s)")

    # Build the pipeline — same call run_pipeline.py makes.
    print(f"building pipeline (backend={args.backend}, model={args.model})...",
          flush=True)
    t = time.perf_counter()
    kwargs = dict(
        db_path=args.db,
        backend=args.backend,
        model=args.model,
        backend_host=args.backend_host,
    )
    if args.threshold is not None:
        kwargs["retrieval_threshold"] = args.threshold
    pipeline = build_pipeline(**kwargs)
    print(f"  built in {time.perf_counter() - t:.1f}s")

    # Warm-up: first .encode() / first vector query / llama prefix cache
    # are all slower than steady state. Run one throwaway query so the
    # first *real* utterance isn't misleadingly slow. 
    print("warming up...", end=" ", flush=True)
    t = time.perf_counter()
    pipeline.answer(args.species, "warmup query, ignore")
    print(f"({(time.perf_counter() - t) * 1000:.0f}ms)")

    return voice_repl(pipeline, args.species, whisper_model, voice)


if __name__ == "__main__":
    sys.exit(main())