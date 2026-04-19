"""ElevenLabs-backed voice I/O for the tour guide.

- `VoiceIO.listen()` records the default microphone until the user presses
  Enter, then transcribes via ElevenLabs Scribe. Returns the transcript.
- `VoiceIO.speak(text)` synthesizes `text` via ElevenLabs TTS and plays it
  through the default audio output, blocking until playback finishes.

Audio is mono PCM — 16 kHz for capture, 22.05 kHz for playback. Requires
`ELEVEN_LABS_API_KEY` in the environment (loaded from `.env` by `demo.py`).
"""

from __future__ import annotations

import io
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs


# ElevenLabs preset voice — "Brian", deep American male — fits a ranger persona.
# Swap for any other voice_id the account has access to.
DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"
DEFAULT_TTS_MODEL = "eleven_turbo_v2_5"   # low-latency, good quality
DEFAULT_STT_MODEL = "scribe_v1"

_RECORD_SR = 16000
_PLAYBACK_SR = 22050
_MIN_RECORDING_SAMPLES = _RECORD_SR // 2  # drop anything shorter than 0.5 s


class VoiceIO:
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        tts_model: str = DEFAULT_TTS_MODEL,
        stt_model: str = DEFAULT_STT_MODEL,
    ) -> None:
        api_key = os.environ.get("ELEVEN_LABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVEN_LABS_API_KEY is not set")
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.tts_model = tts_model
        self.stt_model = stt_model

    def listen(self) -> str:
        """Record mic until Enter, transcribe, return the text (may be empty)."""
        audio = self._record_until_enter()
        if audio.size < _MIN_RECORDING_SAMPLES:
            return ""
        return self._transcribe(audio)

    def speak(self, text: str) -> None:
        """Synthesize `text` and play it through the default audio output."""
        if not text.strip():
            return
        stream = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.tts_model,
            output_format=f"pcm_{_PLAYBACK_SR}",
        )
        audio_bytes = b"".join(stream)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(audio, _PLAYBACK_SR)
        sd.wait()

    def _record_until_enter(self) -> np.ndarray:
        frames: list[np.ndarray] = []

        def callback(indata, _frames, _time, _status):
            frames.append(indata.copy())

        print("[listening — press Enter to stop]", flush=True)
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=_RECORD_SR,
            dtype="int16",
        ):
            try:
                input()
            except EOFError:
                pass

        if not frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(frames, axis=0).flatten()

    def _transcribe(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, _RECORD_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        result = self.client.speech_to_text.convert(
            file=buf,
            model_id=self.stt_model,
        )
        return (result.text or "").strip()
