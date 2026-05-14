import time
from pywhispercpp.model import Model

print("loading model...")
t0 = time.perf_counter()
model = Model("models/ggml-tiny.en-q5_1.bin", n_threads=4)
print(f"loaded in {time.perf_counter() - t0:.2f}s")

# Use any 16kHz mono WAV you have. 
print("transcribing...")
t0 = time.perf_counter()
segments = model.transcribe("test.wav")
elapsed = time.perf_counter() - t0
text = " ".join(s.text for s in segments).strip()
print(f"transcribed in {elapsed:.2f}s: {text!r}")