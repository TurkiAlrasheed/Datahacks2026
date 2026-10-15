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
    pipeline.answer->  Response, sentence by sentence while the LLM streams
    sentences      ->  Speaker (piper synth thread + player thread)  (TTS out)

Species is identified ONCE at startup. Point the robot, classify, then
take questions. A failed identification (low confidence, seashore, camera
hiccup) asks you to try again instead of exiting, so a retry doesn't reload
every model. To re-identify after that, restart the loop — keeps the
per-turn latency budget intact (no ~1s camera+TTA on every button press).

Startup overlaps work with the visitor's own time: Whisper, the embedder,
the gate/intent centroids and the LLM weights all load on background
threads while the camera is being aimed. The identification is announced
the moment it is made, and the push-to-talk prompt follows the
announcement: nothing is joined before it. The LLM prefill for the species
runs in the background while the visitor presses the button and talks;
Whisper is joined at the first transcription and the pipeline is built at
the first question if its components were still loading. Only the corpus
connection is built on the main thread, because sqlite3 connections are
bound to their creating thread.

Each answer is spoken while it is generated: the pipeline hands over every
complete sentence as the LLM streams it, and a pre-rendered filler line
("Gathering all the information...") covers the prompt processing when a
question goes to the LLM.

The classifier is whatever model lives in --model-dir (default
species_identification/outputs/deploy): a model_*.tflite plus
model_manifest.json, which names the .tflite file and declares the input
preprocessing, class order, temperature and identification threshold. See
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
    A model dir with model_*.tflite + model_manifest.json.
    A schema v2 corpus.db (species-partitioned vectors).

Press-to-talk prototype: this uses <enter> as the button (press to start
recording, press again to stop). On the device, swap record_utterance()
for a GPIO button-edge implementation — that is the ONLY function that
needs to change for the hardware port.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

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
from pipeline_factory import build_pipeline, prewarm_components  # noqa: E402
from test_corpus import _get_common_name, open_db  # noqa: E402
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
                     common_names: dict[str, str] | None = None,
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

    common_names maps species_id -> the name to announce ("song sparrow"
    rather than "Melospiza melodia"); labels missing from it are spoken as
    the scientific name.

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

        # Announce the common name when the corpus has one; otherwise drop
        # the underscores so the scientific name reads naturally. The
        # species_id passed to the pipeline keeps the underscored form.
        spoken = ((common_names or {}).get(ident.label)
                  or ident.label.replace("_", " "))
        article = "an" if spoken[:1].lower() in "aeiou" else "a"
        msg = (f"I see {article} {spoken}. "
               f"Press the button to ask me questions.")
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

# Silence in front of every clip. Each clip is played by a fresh player
# call, and an output device that has just been opened can swallow the first
# syllable; this replaces the old `"-" + text` workaround.
LEAD_SILENCE_SEC = 0.12

# Spoken as soon as a question goes to the LLM (the slow path), so the
# visitor knows the robot is working. Rendered while the synth thread is
# idle at startup, so playing one costs no CPU during the LLM's prompt
# processing; rotated so repeat questions don't hear the same line.
FILLER_PHRASES = (
    "Gathering all the information...",
    "Let me check my field notes.",
    "Good question. One moment.",
)


class Speaker:
    """
    Non-blocking text-to-speech. say() queues text and returns at once. A
    synth thread turns each item into a WAV and a player thread plays them
    in order, so sentence N+1 is synthesized while sentence N plays; the
    voice loop hands over sentences while the LLM is still generating.

    Only the synth thread touches the PiperVoice. Per-turn timings are
    measured from the reset_turn() timestamp (button-up).
    """

    def __init__(self, voice: PiperVoice) -> None:
        self.voice = voice
        self._synth_q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._play_q: "queue.Queue[tuple[Path, str, bool]]" = queue.Queue()
        self._dir = Path(tempfile.mkdtemp(prefix="roboranger_tts_"))
        atexit.register(shutil.rmtree, self._dir, ignore_errors=True)
        self._clip_n = 0
        # Filler state belongs to the synth thread, except _next_filler
        # (main thread only).
        self._filler_wavs: dict[str, list[Path]] = {}
        self._unrendered = list(FILLER_PHRASES)
        self._next_filler = 0
        self.reset_turn(time.perf_counter())
        threading.Thread(target=self._synth_loop, name="tts-synth",
                         daemon=True).start()
        threading.Thread(target=self._play_loop, name="tts-play",
                         daemon=True).start()

    # -- public, main thread ---------------------------------------------

    def say(self, text: str) -> None:
        """Queue text to be spoken after everything already queued."""
        text = text.strip()
        if text:
            self._synth_q.put(("text", text))

    def say_and_wait(self, text: str) -> None:
        self.say(text)
        self.wait()

    def play_filler(self) -> None:
        """Queue the next filler line (see FILLER_PHRASES)."""
        phrase = FILLER_PHRASES[self._next_filler % len(FILLER_PHRASES)]
        self._next_filler += 1
        self._synth_q.put(("filler", phrase))

    def wait(self) -> None:
        """Block until everything queued has been synthesized and played."""
        self._synth_q.join()   # items reach _play_q before task_done()
        self._play_q.join()

    def reset_turn(self, t0: float) -> None:
        """Start a new turn's timings; call while idle, at button-up."""
        self._t0 = t0
        self._first_audio: float | None = None
        self._first_answer: float | None = None
        self._synth_s = 0.0

    def turn_timings(self) -> dict[str, float]:
        """first_audio (filler included), first_answer, tts_synth — seconds,
        the first two relative to reset_turn()'s timestamp."""
        out = {"tts_synth": self._synth_s}
        if self._first_audio is not None:
            out["first_audio"] = self._first_audio - self._t0
        if self._first_answer is not None:
            out["first_answer"] = self._first_answer - self._t0
        return out

    # -- worker threads --------------------------------------------------

    def _synthesize(self, text: str) -> Iterator[Path]:
        """
        One WAV per sentence, each yielded as soon as it is synthesized:
        piper splits the text into sentences itself, so a long blurb answer
        starts playing after its first sentence instead of after all of it.
        """
        for chunk in self.voice.synthesize(text):
            self._clip_n += 1
            path = self._dir / f"clip_{self._clip_n}.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setframerate(chunk.sample_rate)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setnchannels(chunk.sample_channels)
                lead = int(chunk.sample_rate * LEAD_SILENCE_SEC)
                wav_file.writeframes(bytes(lead * chunk.sample_width
                                           * chunk.sample_channels))
                wav_file.writeframes(chunk.audio_int16_bytes)
            yield path

    def _synth_loop(self) -> None:
        while True:
            try:
                # While idle, render the filler lines one at a time; real
                # speech never waits behind more than one of them.
                kind, text = self._synth_q.get(
                    timeout=0.25 if self._unrendered else None)
            except queue.Empty:
                phrase = self._unrendered.pop(0)
                try:
                    self._filler_wavs[phrase] = list(self._synthesize(phrase))
                except Exception as e:
                    print(f"  [tts: filler failed: {e!r}]", file=sys.stderr)
                continue
            try:
                if kind == "filler":
                    if text not in self._filler_wavs:
                        self._filler_wavs[text] = list(self._synthesize(text))
                        if text in self._unrendered:
                            self._unrendered.remove(text)
                    for path in self._filler_wavs[text]:
                        self._play_q.put((path, kind, False))
                else:
                    t = time.perf_counter()
                    for path in self._synthesize(text):
                        self._synth_s += time.perf_counter() - t
                        self._play_q.put((path, kind, True))
                        t = time.perf_counter()
            except Exception as e:
                print(f"  [tts: synthesis failed: {e!r}]", file=sys.stderr)
            finally:
                self._synth_q.task_done()

    def _play_loop(self) -> None:
        while True:
            path, kind, delete = self._play_q.get()
            try:
                now = time.perf_counter()
                if self._first_audio is None:
                    self._first_audio = now
                if kind == "text" and self._first_answer is None:
                    self._first_answer = now
                _play_wav(path)
            except Exception as e:
                print(f"  [tts: playback failed: {e!r}]", file=sys.stderr)
            finally:
                if delete:
                    path.unlink(missing_ok=True)
                self._play_q.task_done()


# Resolved once: the first playback finds the player, later ones reuse it.
_PLAYER_CMD: list[str] | None = None


def _play_wav(path: Path) -> None:
    """Play a WAV synchronously with the platform's player."""
    global _PLAYER_CMD
    if sys.platform == "win32":
        import winsound  # stdlib, Windows only; no subprocess per clip
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    if _PLAYER_CMD is None:
        _PLAYER_CMD = next((cmd for cmd in (["aplay", "-q"], ["paplay"],
                                            ["afplay"])
                            if shutil.which(cmd[0])), [])
    if not _PLAYER_CMD:
        print(f"  [no audio player found — skipped {path.name}]",
              file=sys.stderr)
        return
    subprocess.run(_PLAYER_CMD + [str(path)], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Latency table — same shape as run_pipeline.py, plus stt/tts rows
# ---------------------------------------------------------------------------

# Everything is measured from button-up (end of recording), so the time the
# visitor spends waiting to press the button or talking isn't counted:
#   stt                 button-up -> transcript (incl. a first-turn wait
#                       for Whisper to finish loading)
#   gate .. llm         pipeline stages; llm_first_sentence is when the
#                       streamed answer's first full sentence arrived
#   first_audio         button-up -> first sound (often the filler line)
#   first_answer        button-up -> first sentence of the answer playing
#   tts_synth           total piper synthesis time (overlaps playback)
#   total_voice         button-up -> end of speech
VOICE_STAGE_ORDER = (
    "stt",
    "gate", "blurb", "intent", "format", "retrieval", "prompt",
    "llm_first_sentence", "llm",
    "first_audio", "first_answer", "tts_synth",
    "total_voice",
)


def fmt_ms(seconds: float | None) -> str:
    return "      —" if seconds is None else f"{seconds * 1000:7.1f}ms"


def print_turn(resp: Response, transcript: str, *, stt_s: float,
               audio: dict[str, float], total_voice_s: float) -> None:
    """Print transcript, answer, and the full voice-inclusive latency table."""
    merged: dict[str, float] = dict(resp.latency)
    merged.update(audio)
    merged["stt"] = stt_s
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
        print(f"  {marker} {stage:18s} {fmt_ms(merged.get(stage))}")
    print()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class Session:
    """
    What the voice loop needs that may still be loading when it starts:
    the Whisper model and the pipeline. Each is waited for only when a turn
    actually needs it, so the visitor can press the button and talk while
    they finish loading.

    Main thread only: the pipeline's sqlite connection is bound to the
    thread that builds it. Background threads report through note(), and
    the loop prints those lines between turns so they don't land in the
    middle of the push-to-talk prompt.
    """

    def __init__(self, *, speaker: Speaker,
                 whisper_thread: threading.Thread,
                 prewarm_thread: threading.Thread,
                 background: dict, background_errors: dict,
                 build_kwargs: dict) -> None:
        self._speaker = speaker
        self._whisper_thread = whisper_thread
        self._prewarm_thread = prewarm_thread
        self._background = background
        self._errors = background_errors
        self._build_kwargs = build_kwargs
        self._pipeline: RoboRangerPipeline | None = None
        self._notes: "queue.Queue[str]" = queue.Queue()

    def note(self, line: str) -> None:
        """Queue a log line from any thread; printed at the next turn."""
        self._notes.put(line)

    def print_notes(self) -> None:
        while not self._notes.empty():
            print(self._notes.get_nowait(), file=sys.stderr)

    def components_ready(self) -> bool:
        return not self._prewarm_thread.is_alive()

    def whisper(self) -> WhisperModel:
        if self._whisper_thread.is_alive():
            print("  waiting for whisper to finish loading...", flush=True)
        self._whisper_thread.join()
        if "whisper" in self._errors:
            raise self._errors["whisper"]
        if "whisper_load_s" in self._background:  # first call only
            print(f"  whisper loaded in "
                  f"{self._background.pop('whisper_load_s'):.1f}s")
        return self._background["whisper"]

    def pipeline(self) -> RoboRangerPipeline:
        """The pipeline, built on first use from the prewarmed components
        (joining the prewarm thread first if it is still running)."""
        if self._pipeline is not None:
            return self._pipeline
        if self._prewarm_thread.is_alive():
            print("waiting for pipeline components...", flush=True)
            self._speaker.say("One moment while I get ready.")
        t = time.perf_counter()
        self._prewarm_thread.join()
        components = self._background.get("components")
        if "components" in self._errors:
            print(f"warn: background prewarm failed "
                  f"({self._errors['components']!r}); loading inline",
                  file=sys.stderr)
            components = None
        else:
            timings = getattr(components, "timings", None) or {}
            print(f"  components ready in "
                  f"{self._background['components_s']:.1f}s "
                  f"(foreground wait {time.perf_counter() - t:.1f}s): "
                  + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()
                              if isinstance(v, float)))
            if timings.get("llm_error"):
                print(f"warn: LLM prewarm: {timings['llm_error']}",
                      file=sys.stderr)

        print("building pipeline...", end=" ", flush=True)
        t = time.perf_counter()
        self._pipeline = build_pipeline(**self._build_kwargs,
                                        components=components)
        print(f"({(time.perf_counter() - t) * 1000:.0f}ms)")
        return self._pipeline


def voice_repl(session: Session, species_id: str, speaker: Speaker) -> int:
    """
    Push-to-talk loop. One iteration = one utterance = one spoken answer.
    Ctrl-C to exit.

    Species is fixed for the session (identified at startup OR passed
    via --species). To re-identify, restart the script.
    """
    print(f"\nvoice loop ready — species: {species_id}")
    print("Ctrl-C to exit.\n")

    print_total_rss(label="voice loop start")

    while True:
        try:
            # Don't open the mic while the robot is still talking, or the
            # recording starts with its own voice.
            speaker.wait()
            session.print_notes()

            # --- STT: speech in -------------------------------------------
            audio = record_utterance()
            t_up = time.perf_counter()  # button-up: every turn timing's zero
            speaker.reset_turn(t_up)
            transcript = ""
            if len(audio) >= SAMPLE_RATE * MIN_UTTERANCE_SEC:
                transcript = transcribe(session.whisper(), audio)
            stt_s = time.perf_counter() - t_up

            # Empty / too-short / whisper-returned-nothing: don't run the
            # pipeline on garbage. Speak a short reprompt and loop. This is
            # the "I didn't catch that" unhappy path — keep it spoken so
            # the device never just goes silent.
            if not transcript:
                print("  (nothing heard)")
                speaker.say("I didn't catch that. Press the button and "
                            "try again.")
                continue

            # --- pipeline core --------------------------------------------
            # answer() does its own per-stage timing into resp.latency. On
            # the LLM path it cues the filler line, then hands over each
            # sentence as it is generated so speech starts before the
            # answer is finished.
            resp = session.pipeline().answer(
                species_id, transcript,
                on_llm_start=speaker.play_filler,
                on_sentence=speaker.say)

            # --- TTS: speech out ------------------------------------------
            if not resp.streamed:
                speaker.say(resp.text)
            speaker.wait()
            print_turn(resp, transcript, stt_s=stt_s,
                       audio=speaker.turn_timings(),
                       total_voice_s=time.perf_counter() - t_up)

        except (KeyboardInterrupt, EOFError):
            # EOFError: stdin closed (piped input, systemd, ssh without a
            # tty). Exit as cleanly as Ctrl-C rather than tracebacking out
            # of the push-to-talk prompt.
            print("\nbye")
            return 0


# ---------------------------------------------------------------------------
# Startup checks + device status
# ---------------------------------------------------------------------------

def corpus_preflight(db_path: Path) -> tuple[dict, dict[str, str]]:
    """
    Check corpus.db before any camera/voice work: it must exist, be schema
    v2 (species-partitioned vectors), and match the runtime embedder.
    Returns (corpus_meta, {species_id: common_name}) for every species with
    a blurb. Raises CorpusSchemaError or FileNotFoundError with an
    actionable message.
    """
    if not db_path.is_file():
        # sqlite3.connect() would silently create an empty file here.
        raise FileNotFoundError(f"corpus not found: {db_path}")
    conn = open_db(db_path)
    try:
        meta = check_corpus_schema(conn)
        species = [r[0] for r in conn.execute(
            "SELECT species_id FROM species WHERE blurb_json IS NOT NULL")]
        names = {sid: _get_common_name(conn, sid) for sid in species}
    finally:
        conn.close()
    return meta, names


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_device_status(path: Path, *, db_path: Path, corpus_meta: dict,
                        classifier: TFLiteClassifierTTA | None,
                        species_id: str, warmup: dict,
                        kernels: str | None = None) -> None:
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
            "kernels": kernels,
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
    # Required unless --camera-test (checked after parsing, so the camera
    # check can run without any model files).
    p.add_argument("--voice", type=Path, default=None,
                   help="path to piper voice .onnx (+ .onnx.json sibling); "
                        "required unless --camera-test")
    p.add_argument("--whisper-model", type=Path, default=None,
                   help="path to whisper.cpp ggml model "
                        "(ggml-tiny.en-q5_1.bin recommended); "
                        "required unless --camera-test")
    p.add_argument("--whisper-threads", type=int, default=4,
                   help="threads for whisper.cpp (default: 4)")
    # Vision args.
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="directory with model_*.tflite + "
                        "model_manifest.json (default: outputs/deploy)")
    p.add_argument("--camera-index", type=int, default=0,
                   help="V4L2 camera index (default: 0)")
    p.add_argument("--id-threshold", type=float, default=None,
                   help="override the manifest's min confidence to accept "
                        "an identification")
    p.add_argument("--kernels", choices=("ref", "default"), default="default",
                   help="LiteRT kernels: default uses XNNPACK (14-43x faster "
                        "on the UNO Q); ref disables it, for debugging a "
                        "suspected delegate problem. Compare with "
                        "species_identification/tests/bench_device.py")
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
    missing = [flag for flag, value in (("--voice", args.voice),
                                        ("--whisper-model", args.whisper_model))
               if value is None]
    if missing:
        p.error(f"the following arguments are required: {', '.join(missing)} "
                f"(only --camera-test runs without them)")
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
    # All speech goes through the Speaker's threads from here on. While it
    # is idle it pre-renders the filler lines.
    speaker = Speaker(voice)

    # --- Corpus preflight --------------------------------------------------
    # Seconds-cheap, and a stale (v1) or missing corpus would otherwise only
    # surface after the camera flow, when the pipeline is built.
    db_path = Path(args.db)
    try:
        corpus_meta, common_names = corpus_preflight(db_path)
    except (CorpusSchemaError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        speaker.say_and_wait("My knowledge base is missing or out of date. "
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
    # Performance note: aiming the camera and classifying one frame takes
    # several seconds of the visitor's time, so everything that doesn't need
    # the species loads on background threads meanwhile:
    #
    #   whisper-load      the STT model
    #   pipeline-prewarm  sentence-transformer, gate + intent centroids, and
    #                     a 1-token LLM request that makes the model server
    #                     load its weights and cache the system prompt
    #
    # Only the corpus connection is left for the main thread (sqlite3
    # connections are bound to the thread that creates them), and that part
    # is milliseconds. The threads write into a shared dict; neither is
    # joined until a question needs it (see Session).
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

    def _prewarm_pipeline_bg():
        try:
            t = time.perf_counter()
            background["components"] = prewarm_components(
                backend=args.backend, model=args.model,
                backend_host=args.backend_host)
            background["components_s"] = time.perf_counter() - t
        except BaseException as e:
            background_errors["components"] = e

    print(f"loading whisper + pipeline components in background "
          f"(backend={args.backend}, model={args.model})...", flush=True)
    whisper_thread = threading.Thread(
        target=_load_whisper_bg, name="whisper-load", daemon=True)
    prewarm_thread = threading.Thread(
        target=_prewarm_pipeline_bg, name="pipeline-prewarm", daemon=True)
    whisper_thread.start()
    prewarm_thread.start()

    species_id: str
    id_message: str | None = None
    classifier: TFLiteClassifierTTA | None = None
    if args.species:
        print(f"using --species override: {args.species}")
        species_id = args.species
        if species_id not in common_names:
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
            classifier = TFLiteClassifierTTA.from_dir(
                args.model_dir, reference_kernels=args.kernels == "ref")
        except ModelContractError as e:
            print()
            print(f"error: {e}", file=sys.stderr)
            speaker.say_and_wait("I cannot find my vision model. Please "
                                 "check the setup.")
            return 1
        print(f"({time.perf_counter() - t:.1f}s)")
        print(f"  {classifier.describe()}")

        # Every species the classifier can announce needs a blurb, or the
        # visitor gets "I don't have information about this species".
        uncovered = sorted(
            c for c in classifier.class_names
            if c not in classifier.negative_classes
            and c not in common_names)
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
        speaker.say("Point me at an animal, then press enter.")
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
                save_dir=args.save_id_frames, common_names=common_names)
            if species_id is not None:
                id_message = msg
                break
            speaker.say(msg)
            print(f"identification failed (top conf {conf*100:.1f}%): "
                  f"{msg}", file=sys.stderr)
            prompt = "\n  [enter to try again, Ctrl-C to quit] "
        print(f"identified: {species_id} ({conf*100:.1f}%)")

    # --- Announce, then hand over to the visitor --------------------------
    # The announcement plays on the Speaker's threads while the rest of this
    # runs, and the push-to-talk prompt follows it. Nothing here waits for a
    # background load: Whisper is joined at the first transcription, and the
    # pipeline is built at the first question if its components are still
    # loading (see Session).
    if id_message:
        speaker.say(id_message)

    build_kwargs = dict(
        db_path=args.db,
        backend=args.backend,
        model=args.model,
        backend_host=args.backend_host,
    )
    if args.threshold is not None:
        build_kwargs["retrieval_threshold"] = args.threshold
    session = Session(speaker=speaker, whisper_thread=whisper_thread,
                      prewarm_thread=prewarm_thread, background=background,
                      background_errors=background_errors,
                      build_kwargs=build_kwargs)

    def _write_status(warm: dict) -> None:
        """Safe from any thread; reports through the session's notes."""
        try:
            write_device_status(args.status_file, db_path=db_path,
                                corpus_meta=corpus_meta, classifier=classifier,
                                species_id=species_id, warmup=warm,
                                kernels=args.kernels)
            session.note(f"status written to {args.status_file}")
        except OSError as e:
            session.note(f"warn: could not write {args.status_file}: {e}")

    # --- Warm up ----------------------------------------------------------
    # With the components already loaded, the pipeline builds in
    # milliseconds, and the species' retrieval and prompt assembly run here
    # because they use the corpus connection this thread owns. The LLM
    # prefill for that prompt is pure HTTP, so it runs on a thread while the
    # announcement plays and the visitor asks a first question. A question
    # that arrives before it finishes queues behind it on the model server
    # and reuses the prefix it cached. With the components still loading,
    # all of this is skipped: the first real question does the same prefill.
    if session.components_ready():
        pipeline = session.pipeline()
        messages, warm = pipeline.warmup_prompt(species_id)
        print("warmup prompt ready ("
              + ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in warm.items()
                          if isinstance(v, float))
              + "); LLM prefill continues in the background")
        # growth_check(pipeline, species_id, n=12)
        if messages is None:
            print(f"warn: warmup: {warm.get('error')}", file=sys.stderr)
            _write_status(warm)
        else:
            def _warm_llm_bg():
                t0 = time.perf_counter()
                error = pipeline.warm_llm(messages)
                warm["llm"] = time.perf_counter() - t0
                if error:
                    warm["error"] = error
                    session.note(f"warn: warmup: {error}")
                else:
                    session.note(f"  LLM prefill for {species_id} done in "
                                 f"{warm['llm']:.1f}s")
                _write_status(warm)

            threading.Thread(target=_warm_llm_bg, name="llm-warm",
                             daemon=True).start()
    else:
        print("pipeline components still loading; the first question will "
              "wait for them", flush=True)
        _write_status({"deferred": "pipeline built at the first question"})

    return voice_repl(session, species_id, speaker)


if __name__ == "__main__":
    sys.exit(main())