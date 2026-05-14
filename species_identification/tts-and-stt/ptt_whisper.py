import io
import time
import wave

import numpy as np
import sounddevice as sd
from pywhispercpp.model import Model

# Reuse from Phase 2.
from ptt_test import record_until_keypress, SAMPLE_RATE, CHANNELS

MODEL_PATH = "models/ggml-tiny.en-q5_1.bin"


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """
    pywhispercpp's transcribe() wants a path or a numpy float32 array
    in [-1, 1]. We have int16 from sounddevice. Convert.
    """
    return (audio.astype(np.float32) / 32768.0)


def main() -> None:
    print("loading whisper...")
    t0 = time.perf_counter()
    model = Model(MODEL_PATH, n_threads=4)
    print(f"loaded in {time.perf_counter() - t0:.2f}s")

    print("\nready. press enter twice per utterance, ctrl-c to quit.\n")
    while True:
        try:
            audio = record_until_keypress()
        except KeyboardInterrupt:
            print("\nbye")
            return

        if len(audio) < SAMPLE_RATE * 0.3:
            print("  too short, skipping")
            continue

        audio_f32 = audio_to_wav_bytes(audio)

        t0 = time.perf_counter()
        segments = model.transcribe(audio_f32)
        elapsed = time.perf_counter() - t0

        text = " ".join(s.text for s in segments).strip()
        print(f"  [{elapsed:.2f}s] {text!r}\n")


if __name__ == "__main__":
    main()