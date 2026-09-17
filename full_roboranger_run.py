"""
RoboRanger loop — image classification + push-to-talk STT + TTS wrapped around the pipeline,
with one-shot species identification at session start.

This is the audio front-end / back-end. It does NOT touch pipeline.answer():
that stays a pure (species_id, query) -> Response function with no I/O, so
it stays testable and run_pipeline.py keeps working unchanged. All the
microphone / speaker / camera / model-loading I/O lives here, at the edges:

    camera frame   ->  TFLite classifier (TTA) -> species_id (CV in, once)
    button down/up ->  record audio            -> STT in
    whisper        ->  query string
    pipeline.answer->  Response                              (UNCHANGED core)
    resp.text      ->  piper                                 (TTS out)

Species is identified ONCE at startup. Point the robot, classify, then
take questions. A failed identification (low confidence, seashore, camera
hiccup) asks you to try again instead of exiting, so a retry doesn't reload
every model. To re-identify after that, restart the loop — keeps the
per-turn latency budget intact (no ~1s camera+TTA on every button press).

The classifier is whatever model lives in --model-dir (default
species_identification/outputs/deploy): model_int8.tflite plus
model_manifest.json, which declares the input preprocessing, class order,
temperature and identification threshold. See
species_identification/vision/tflite_classifier.py.

Usage:
    # Identify from camera, then chat:
    python full_roboranger_run.py \\
        --voice voices/en_US-lessac-low.onnx \\
        --whisper-model models/ggml-tiny.en-q5_1.bin

    # Override classifier (e.g. for testing without a camera):
    python full_roboranger_run.py --species Marah_macrocarpa \\
        --voice voices/en_US-lessac-low.onnx \\
        --whisper-model models/ggml-tiny.en-q5_1.bin

Requirements:
    pip install sounddevice numpy pywhispercpp piper-tts opencv-python \\
                ai-edge-litert
    A piper voice .onnx (+ .onnx.json sibling).
    Ollama installed and running (when on ollama backend).
    A whisper.cpp ggml model (ggml-tiny.en-q5_1.bin recommended).
    A model dir with model_int8.tflite + model_manifest.json.
    A schema v2 corpus.db (species-partitioned vectors).

Press-to-talk prototype: this uses <enter> as the button (press to start
recording, press again to stop). On the device, swap record_utterance()
for a GPIO button-edge implementation — that is the ONLY function that
needs to change for the hardware port.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
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
try:
    import cv2
except ImportError:
    print("error: opencv-python not installed — pip install opencv-python",
          file=sys.stderr)
    raise

# Resolve project modules from this file's location, not the caller's CWD.
_ROOT = Path(__file__).resolve().parent
for _sub in ("species_identification/vision", "species_identification/tests",
             "species_identification/llm-tuning",
             "species_identification/pipeline", "species_identification"):
    sys.path.insert(1, str(_ROOT / _sub))
from corpus_schema import (CorpusSchemaError, check_corpus_schema,  # noqa: E402
                           read_meta)
from mem_check import print_total_rss, growth_check  # noqa: E402
from pipeline import RoboRangerPipeline, Response  # noqa: E402
from pipeline_factory import build_pipeline  # noqa: E402
from test_corpus import open_db  # noqa: E402
from tflite_classifier import (ModelContractError,  # noqa: E402
                               TFLiteClassifierTTA)


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
# Vision constants
# ---------------------------------------------------------------------------

# The identification threshold, temperature and negative classes (e.g.
# "seashore" -> "no animal") come from the model's manifest: the threshold
# is fit by tests/eval_tflite.py on the int8 model's TTA-averaged
# probabilities, which is the distribution it is applied to here.
DEFAULT_MODEL_DIR = _ROOT / "species_identification" / "outputs" / "deploy"

# Number of throwaway frames before the "real" capture. Cheap cameras
# (and V4L2 on Linux) buffer 1-2 stale frames; reading them flushes the
# pipe so we classify what's actually in front of the robot.
CAMERA_WARMUP_FRAMES = 5

REFUSAL_MESSAGES = {
    "negative_class": ("I do not see any animals right now. "
                       "Point me at something and try again."),
    "low_confidence": ("I am not sure what I see. Try moving closer or "
                       "pointing me more directly at the animal."),
}


# ---------------------------------------------------------------------------
# Vision — identify species once
# ---------------------------------------------------------------------------

def _annotate_id_frame(frame: np.ndarray,
                       preds: list[tuple[str, float]]) -> np.ndarray:
    """
    Burn top-N predictions onto a single frame. Mirrors the style in
    live_inference.py's _annotate() so debug images look consistent
    between the two scripts.
    """
    h, w = frame.shape[:2]
    show = preds[:5]
    strip_h = 30 + 26 * len(show)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame,
                time.strftime("%Y-%m-%d %H:%M:%S") + "  [identify_species]",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)

    y = 50
    for name, p in show:
        if p > 0.7:
            color = (0, 255, 0)
        elif p > 0.4:
            color = (0, 200, 255)
        else:
            color = (140, 140, 140)
        cv2.putText(frame, f"{name:<28s} {p * 100:5.1f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 26
    return frame


def identify_species(classifier: TFLiteClassifierTTA,
                     camera_index: int,
                     conf_threshold: float | None = None,
                     save_dir: Path | None = None,
                     ) -> tuple[str | None, float, str]:
    """
    Grab one frame and classify it. Returns (species_id, confidence, message).

    species_id is None on any failure (camera, low confidence, non-species
    top class); message is a human-readable explanation suitable for piper
    to speak. On success species_id is the top-1 label and message names
    what we saw + confidence.

    If save_dir is given, writes two files there:
      - id_raw_<timestamp>.jpg     : exactly what the camera saw
      - id_annotated_<timestamp>.jpg: with top-3 predictions burned in
    Look at these after a run to confirm the camera was actually used and
    pointed at the right thing.

    Camera is opened and released here — the voice loop never holds the
    handle, so there's no contention with anything else on the device.
    """
    # Backend selection: MSMF (the Windows default) is slow and flaky for
    # USB cameras — it can take 3-5s just to open the Brio. DirectShow is
    # both faster and more reliable. On Linux/macOS, CAP_ANY is correct.
    t0 = time.perf_counter()
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    t_open = time.perf_counter() - t0

    if not cap.isOpened():
        return None, 0.0, (f"I could not open the camera. "
                           f"Check that it is plugged in and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Smaller buffer means cap.read() returns the freshest frame, not the
    # oldest queued one. Some drivers ignore this; harmless when they do.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        # Flush stale buffer frames. Without this the first read() is often
        # a frame from before the camera was even pointed at anything.
        t0 = time.perf_counter()
        for _ in range(CAMERA_WARMUP_FRAMES):
            cap.read()
        t_warmup = time.perf_counter() - t0

        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_grab = time.perf_counter() - t0

        if not ok or frame is None:
            return None, 0.0, ("I could not get a picture from the camera. "
                               "Please try again.")

        # Save the raw frame BEFORE classification, so if the classifier
        # crashes we still have evidence of what the camera saw.
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        raw_path = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            raw_path = save_dir / f"id_raw_{ts_tag}.jpg"
            cv2.imwrite(str(raw_path), frame)
            print(f"  saved raw frame: {raw_path}")

        # TTA-averaged, temperature-scaled predictions with the manifest's
        # acceptance policy (negative classes, id_threshold) applied.
        t0 = time.perf_counter()
        ident = classifier.identify(frame, threshold=conf_threshold)
        t_infer = time.perf_counter() - t0
        preds = ident.top

        # Per-stage timing — tells you exactly where the time went on
        # the FIRST identification (where TFLite warmup also lands if
        # the classifier wasn't pre-warmed).
        print(f"  timing: open={t_open*1000:6.0f}ms  "
              f"warmup={t_warmup*1000:6.0f}ms  "
              f"grab={t_grab*1000:6.0f}ms  "
              f"infer={t_infer*1000:6.0f}ms")

        # Show the top-3 so a low-conf failure is debuggable from stdout.
        print("  classifier top-3:")
        for name, p in preds[:3]:
            print(f"    {name:<32s} {p * 100:5.1f}%")

        # Save annotated frame with predictions burned in. Reuses the same
        # overlay style as live_inference.py so the files look familiar.
        if save_dir is not None:
            annotated = _annotate_id_frame(frame.copy(), preds)
            ann_path = save_dir / f"id_annotated_{ts_tag}.jpg"
            cv2.imwrite(str(ann_path), annotated)
            print(f"  saved annotated frame: {ann_path}")

        if not ident.accepted:
            return None, ident.confidence, REFUSAL_MESSAGES[ident.reason]

        # Replace underscores so the spoken confirmation reads naturally.
        # The species_id passed to the pipeline keeps the underscored form.
        spoken = ident.label.replace("_", " ")
        msg = f"I see a {spoken}. Press the button to ask me questions."
        return ident.label, ident.confidence, msg

    finally:
        cap.release()


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

    Species is fixed for the session (identified at startup OR passed
    via --species). To re-identify, restart the script.
    """
    print(f"\nvoice loop ready — species: {species_id}")
    print("Ctrl-C to exit.\n")

    print_total_rss(label="after warmup")

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
            speak(voice, "-" + resp.text)
            tts_s = time.perf_counter() - t_tts

            total_voice_s = stt_s + resp.latency.get("total", 0.0) + tts_s
            print_turn(resp, stt_s, tts_s, total_voice_s, transcript)

        except KeyboardInterrupt:
            print("\nbye")
            return 0


# ---------------------------------------------------------------------------
# Startup checks + device status
# ---------------------------------------------------------------------------

def corpus_preflight(db_path: Path) -> tuple[dict, set[str]]:
    """
    Check corpus.db before any camera/voice work: it must exist, be schema
    v2 (species-partitioned vectors), and match the runtime embedder.
    Returns (corpus_meta, species_ids). Raises CorpusSchemaError or
    FileNotFoundError with an actionable message.
    """
    if not db_path.is_file():
        # sqlite3.connect() would silently create an empty file here.
        raise FileNotFoundError(f"corpus not found: {db_path}")
    conn = open_db(db_path)
    try:
        meta = check_corpus_schema(conn)
        species = {r[0] for r in conn.execute(
            "SELECT species_id FROM species WHERE blurb_json IS NOT NULL")}
    finally:
        conn.close()
    return meta, species


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_device_status(path: Path, *, db_path: Path, corpus_meta: dict,
                        classifier: TFLiteClassifierTTA | None,
                        species_id: str, warmup: dict) -> None:
    """
    Record exactly which corpus and model this unit is running — the thing a
    device would report upstream for OTA bookkeeping (see "OTA corpus
    updates" in onDeviceAgentREADME.md).
    """
    model = None
    if classifier is not None:
        m = classifier.manifest
        model = {
            "dir": str(m.path.parent) if m.path else None,
            "arch": m.arch,
            "tflite_sha256": _sha256(m.model_path),
            "input_range": m.input_range,
            "temperature": m.temperature,
            "id_threshold": m.id_threshold,
            "evaluated_at": m.eval.get("evaluated_at"),
        }
    status = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "corpus": {
            "path": str(db_path),
            **{k: corpus_meta.get(k) for k in (
                "schema_version", "corpus_version", "content_sha256",
                "species_count", "chunk_count", "sqlite_vec_version",
                "embed_model", "built_at")},
        },
        "model": model,
        "session": {"species_id": species_id, "warmup": warmup},
    }
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


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
                   default=str(_ROOT / "species_identification/offline-info/corpus.db"),
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
    # --species used to be required. Now it's an override: provide it to
    # skip the classifier entirely (laptop testing without a camera);
    # leave it out to identify from the camera at startup.
    p.add_argument("--species", default=None,
                   help="override species_id; if omitted, identify from camera")
    # Audio args.
    p.add_argument("--voice", type=Path, required=True,
                   help="path to piper voice .onnx (+ .onnx.json sibling)")
    p.add_argument("--whisper-model", type=Path, required=True,
                   help="path to whisper.cpp ggml model "
                        "(ggml-tiny.en-q5_1.bin recommended)")
    p.add_argument("--whisper-threads", type=int, default=4,
                   help="threads for whisper.cpp (default: 4)")
    # Vision args.
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="directory with model_int8.tflite + "
                        "model_manifest.json (default: outputs/deploy)")
    p.add_argument("--camera-index", type=int, default=0,
                   help="V4L2 camera index (default: 0)")
    p.add_argument("--id-threshold", type=float, default=None,
                   help="override the manifest's min confidence to accept "
                        "an identification")
    p.add_argument("--status-file", type=Path,
                   default=Path("device_status.json"),
                   help="where to record the running corpus/model versions")
    # Debug / diagnostic flags.
    p.add_argument("--save-id-frames", type=Path, default=None,
                   help="if set, save raw + annotated capture frame here "
                        "(useful for verifying the camera worked)")
    p.add_argument("--camera-test", action="store_true",
                   help="open the camera, save one frame to "
                        "camera_test_<timestamp>.jpg, exit. Skips all "
                        "models — fast 'is the camera alive' check.")
    args = p.parse_args()

    # --- Camera test mode: skip every model, just prove the camera works.
    # Saves one frame and exits. Useful when something is wrong and you
    # don't want to wait 15s for whisper + pipeline + classifier to load
    # only to find out /dev/video0 isn't there.
    if args.camera_test:
        out_path = Path(f"camera_test_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        t0 = time.perf_counter()
        if sys.platform == "win32":
            cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(args.camera_index)
        t_open = time.perf_counter() - t0
        if not cap.isOpened():
            print(f"error: could not open /dev/video{args.camera_index}",
                  file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            t0 = time.perf_counter()
            for _ in range(CAMERA_WARMUP_FRAMES):
                cap.read()
            t_warmup = time.perf_counter() - t0
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t_grab = time.perf_counter() - t0
            if not ok or frame is None:
                print("error: camera opened but frame grab failed",
                      file=sys.stderr)
                return 1
            cv2.imwrite(str(out_path), frame)
            print(f"camera test OK — saved {out_path} "
                  f"({frame.shape[1]}x{frame.shape[0]})")
            print(f"  timing: open={t_open*1000:.0f}ms  "
                  f"warmup={t_warmup*1000:.0f}ms  "
                  f"grab={t_grab*1000:.0f}ms")
            return 0
        finally:
            cap.release()

    # Fail fast on missing files before loading anything heavy.
    if not args.voice.exists():
        print(f"error: voice file {args.voice} not found", file=sys.stderr)
        return 1
    if not args.whisper_model.exists():
        print(f"error: whisper model {args.whisper_model} not found",
              file=sys.stderr)
        return 1

    # Load piper FIRST — we need to speak failure messages from the
    # classifier or pipeline build. Whisper comes after, since it's
    # only needed once the loop starts.
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

    # --- Corpus preflight --------------------------------------------------
    # Seconds-cheap, and a stale (v1) or missing corpus would otherwise only
    # surface after the camera flow, when the pipeline is built.
    db_path = Path(args.db)
    try:
        corpus_meta, corpus_species = corpus_preflight(db_path)
    except (CorpusSchemaError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        speak(voice, "My knowledge base is missing or out of date. "
                     "Please check the setup.")
        return 1
    print(f"corpus {db_path}: version {corpus_meta.get('corpus_version')} "
          f"({corpus_meta.get('species_count')} species, "
          f"sha {str(corpus_meta.get('content_sha256'))[:12]})")

    # --- Species ID stage --------------------------------------------------
    # Two paths: --species override (laptop dev, no camera) OR identify
    # from a single camera frame. A failed identification speaks why and
    # waits for another try — the loop won't start with an unknown species.
    #
    # Performance note: while the user is pointing the camera and we're
    # running the classifier (~1-2s), the pipeline + whisper builds run on
    # background threads. They almost always finish before identification
    # does, hiding several seconds of latency behind work the user is
    # already waiting for. The threads write into a shared dict and we
    # join them right before entering the voice loop.
    import threading

    background: dict[str, object] = {}
    background_errors: dict[str, BaseException] = {}

    def _load_whisper_bg():
        try:
            t = time.perf_counter()
            background["whisper"] = WhisperModel(
                str(args.whisper_model), n_threads=args.whisper_threads)
            background["whisper_load_s"] = time.perf_counter() - t
        except BaseException as e:
            background_errors["whisper"] = e

    # Kick off whisper load only — pipeline build has thread-bound sqlite
    # connections, so we have to build it on the main thread.
    print("loading whisper in background...", flush=True)
    whisper_thread = threading.Thread(
        target=_load_whisper_bg, name="whisper-load", daemon=True)
    whisper_thread.start()

    species_id: str
    classifier: TFLiteClassifierTTA | None = None
    if args.species:
        print(f"using --species override: {args.species}")
        species_id = args.species
        if species_id not in corpus_species:
            print(f"warn: {species_id} has no blurb in {db_path}; every "
                  f"question will get the 'no information' reply",
                  file=sys.stderr)
    else:
        # The manifest carries the preprocessing contract, class order,
        # temperature and threshold; a missing or mismatched one is fatal
        # rather than silently mis-preprocessing every frame.
        print("loading classifier...", end=" ", flush=True)
        t = time.perf_counter()
        try:
            classifier = TFLiteClassifierTTA.from_dir(args.model_dir)
        except ModelContractError as e:
            print()
            print(f"error: {e}", file=sys.stderr)
            speak(voice, "I cannot find my vision model. Please check "
                         "the setup.")
            return 1
        print(f"({time.perf_counter() - t:.1f}s)")
        print(f"  {classifier.describe()}")

        # Every species the classifier can announce needs a blurb, or the
        # visitor gets "I don't have information about this species".
        uncovered = sorted(
            c for c in classifier.class_names
            if c not in classifier.negative_classes
            and c not in corpus_species)
        if uncovered:
            print(f"warn: {len(uncovered)} classifier classes have no blurb "
                  f"in {db_path}: {uncovered}", file=sys.stderr)

        # Warm the TFLite interpreter with one throwaway prediction.
        print("warming classifier...", end=" ", flush=True)
        t = time.perf_counter()
        dummy = np.full((480, 640, 3), 128, dtype=np.uint8)
        classifier.predict(dummy)
        print(f"({(time.perf_counter() - t) * 1000:.0f}ms)")

        # Ready beat: prompt the user to point the camera before we capture.
        # A failed identification speaks why and loops back here instead of
        # exiting, so a retry doesn't cost a full model reload on stage.
        speak(voice, "Point me at an animal, then press enter.")
        prompt = "\n  [enter when ready to identify] "
        while True:
            try:
                input(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\naborted before identification")
                return 0

            print("identifying species from camera...", flush=True)
            species_id, conf, msg = identify_species(
                classifier, args.camera_index, args.id_threshold,
                save_dir=args.save_id_frames)
            speak(voice, msg)
            if species_id is not None:
                break
            print(f"identification failed (top conf {conf*100:.1f}%): "
                  f"{msg}", file=sys.stderr)
            prompt = "\n  [enter to try again, Ctrl-C to quit] "
        print(f"identified: {species_id} ({conf*100:.1f}%)")

    # --- Build pipeline on main thread (sqlite is thread-bound) -----------
    print(f"building pipeline (backend={args.backend}, "
          f"model={args.model})...", flush=True)
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

    # --- Join whisper -----------------------------------------------------
    print("waiting for whisper...", flush=True)
    t = time.perf_counter()
    whisper_thread.join()
    if "whisper" in background_errors:
        raise background_errors["whisper"]
    whisper_model = background["whisper"]
    print(f"  whisper loaded in {background['whisper_load_s']:.1f}s "
          f"(foreground wait {time.perf_counter() - t:.1f}s)")

    # --- Warm up ----------------------------------------------------------
    # Real retrieval + one LLM generation on this species' prompt, so the
    # first visitor question doesn't pay cold embedder / llama.cpp costs.
    print("warming up...", end=" ", flush=True)
    t = time.perf_counter()
    warm = pipeline.warmup(species_id)
    print(f"({(time.perf_counter() - t) * 1000:.0f}ms: "
          + ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in warm.items()
                      if isinstance(v, float)) + ")")
    if "error" in warm:
        print(f"warn: warmup: {warm['error']}", file=sys.stderr)
    # growth_check(pipeline, species_id, n=12)

    try:
        write_device_status(args.status_file, db_path=db_path,
                            corpus_meta=corpus_meta, classifier=classifier,
                            species_id=species_id, warmup=warm)
        print(f"status written to {args.status_file}")
    except OSError as e:
        print(f"warn: could not write {args.status_file}: {e}",
              file=sys.stderr)

    return voice_repl(pipeline, species_id, whisper_model, voice)


if __name__ == "__main__":
    sys.exit(main())