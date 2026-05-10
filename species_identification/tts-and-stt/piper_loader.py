import threading
from concurrent.futures import Future

class PiperLoader:
    """Loads Piper in the background. Block on .voice only when needed."""
    
    def __init__(self, model_path: str):
        self._future: Future = Future()
        self._model_path = model_path
        thread = threading.Thread(target=self._load, daemon=True)
        thread.start()
    
    def _load(self):
        try:
            import piper
            voice = piper.PiperVoice.load(self._model_path)
            # Warmup synthesis so first user query doesn't pay the
            # first-inference cost either
            list(voice.synthesize("ready"))
            self._future.set_result(voice)
        except Exception as e:
            self._future.set_exception(e)
    
    @property
    def voice(self):
        """Block until loaded. Re-raises load errors."""
        return self._future.result()
    
    @property
    def ready(self) -> bool:
        return self._future.done()