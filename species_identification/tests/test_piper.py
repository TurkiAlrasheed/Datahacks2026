import sys
import time 

sys.path.insert(1, "../species_identification/tts-and-stt")
from piper_audio import speak_streaming
from piper_loader import PiperLoader

def test_piper_loader():
    piper_loader = PiperLoader("en_US-danny-low.opt.onnx")
    piper_loader._load()  # Force load in test to catch errors
    for i in range(10):
        print(i)
        time.sleep(1)

    def example_token_generator():
        for word in "Hello world! This is a test of streaming synthesis.".split():
            yield word + " "
    speak_streaming(example_token_generator())
        
if __name__ == "__main__":
    test_piper_loader()

    