import queue
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # whisper wants 16kHz mono
CHANNELS = 1
DTYPE = "int16"

def record_until_keypress() -> np.ndarray:
    """
    Capture audio from the default mic until the user presses Enter again.
    Returns int16 mono samples at SAMPLE_RATE.

    Prototype substitute for a hardware button: press Enter to start,
    press Enter again to stop. On the device this becomes
    button_down/button_up GPIO edges.
    """
    chunks: list[np.ndarray] = []
    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status: {status}]", file=sys.stderr)
        q.put(indata.copy())

    input("press enter to start recording...")
    print("recording — press enter to stop...")
    t0 = time.perf_counter()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, callback=callback):
        input()  # blocks until the user presses enter again
        # Drain anything still in the queue.
        while not q.empty():
            chunks.append(q.get_nowait())

    # The stream callback runs on a separate thread; everything in q now
    # is what was captured between the two enter presses.
    while not q.empty():
        chunks.append(q.get_nowait())

    elapsed = time.perf_counter() - t0
    audio = np.concatenate(chunks, axis=0).flatten()
    print(f"  captured {len(audio) / SAMPLE_RATE:.2f}s of audio "
          f"(wall: {elapsed:.2f}s)")
    return audio


def save_wav(audio: np.ndarray, path: str) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


if __name__ == "__main__":
    audio = record_until_keypress()
    save_wav(audio, "test.wav")
    print("saved test.wav")