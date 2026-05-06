"""
LLM backend adapters.

Two backends with the same interface:
  - OllamaBackend: laptop iteration via Ollama's /api/chat endpoint.
  - LlamaCppBackend: Uno Q deployment via llama.cpp's OpenAI-compatible
    /v1/chat/completions endpoint.

Both expose `name`, `model`, and `generate(messages) -> str`. Sampling
params are pinned identically on both so cross-environment differences
come from the runtime, not the sampler.

This module deliberately has no dependencies on any other RoboRanger
module — anything that needs an LLM client can import from here without
dragging in the corpus, prompt builder, or eval fixtures.
"""

from __future__ import annotations

import json
import urllib.request


# Shared sampling params. Pin these on both backends so cross-environment
# differences come from the model runtime, not from sampler defaults.
SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 42,
    "max_tokens": 200,
}


class OllamaBackend:
    """Talks to Ollama's /api/chat. Default host: localhost:11434."""

    name = "ollama"

    def __init__(self, model: str = "smollm2:360m",
                 host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def generate(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": SAMPLING["temperature"],
                "top_p": SAMPLING["top_p"],
                "top_k": SAMPLING["top_k"],
                "seed": SAMPLING["seed"],
                "num_predict": SAMPLING["max_tokens"],
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]


class LlamaCppBackend:
    """
    Talks to llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.
    Start the server on the Uno Q with:
        ./llama-server -m smollm2-360m-q8_0.gguf --host 0.0.0.0 --port 8080 \
                       --ctx-size 2048 --threads 4 --chat-template chatml
    """

    name = "llama-cpp"

    def __init__(self, host: str = "http://localhost:8080",
                 model: str = "smollm2") -> None:
        self.host = host
        self.model = model

    def generate(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
            "seed": SAMPLING["seed"],
            "max_tokens": SAMPLING["max_tokens"],
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]