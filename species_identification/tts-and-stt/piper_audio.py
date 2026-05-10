import sounddevice as sd
import numpy as np
import re
import onnxruntime as ort
import time

t0 = time.time()
import piper
print(f"import: {time.time()-t0:.2f}s")

t0 = time.time()
voice = piper.PiperVoice.load("en_US-danny-low.opt.onnx")
print(f"load: {time.time()-t0:.2f}s")
print([attr for attr in dir(voice) if not attr.startswith('_')])

t0 = time.time()
list(voice.synthesize("first"))
print(f"first synth: {time.time()-t0:.2f}s")

# Check if voice exposes the session - varies by piper version
if hasattr(voice, 'session'):
    # Already loaded, but you can recreate with custom options:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

def synthesize_to_array(text):
    """Run Piper synthesis and return a flat int16 numpy array + sample rate."""
    chunks = []
    sample_rate = None
    for chunk in voice.synthesize(text):
        # AudioChunk exposes raw int16 bytes; sample_rate is on the chunk too
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
    if not chunks:
        return np.array([], dtype=np.int16), 22050
    return np.concatenate(chunks), sample_rate

def speak_streaming(token_iterator):
    buffer = ""
    sentence_end = re.compile(r'[.!?]\s')
    for token in token_iterator:
        buffer += token
        while match := sentence_end.search(buffer):
            sentence = buffer[:match.end()]
            buffer = buffer[match.end():]
            audio, sr = synthesize_to_array(sentence)
            sd.play(audio, sr, blocking=True)
    if buffer.strip():
        audio, sr = synthesize_to_array(buffer)
        sd.play(audio, sr, blocking=True)

if __name__ == "__main__":
    for chunk in voice.synthesize("test"):
        print(type(chunk), dir(chunk))
        break

    def example_token_generator():
        for word in "Hello world! This is a test of streaming synthesis.".split():
            yield word + " "
    speak_streaming(example_token_generator())