"""
LLM backend adapters.

Two backends with the same interface:
  - OllamaBackend: laptop iteration via Ollama's /api/chat endpoint.
  - LlamaCppBackend: Uno Q deployment via llama.cpp's OpenAI-compatible
    /v1/chat/completions endpoint.

Both expose `name`, `model`, `generate(messages, *, max_tokens=None) -> str`
and `stream(messages, *, max_tokens=None) -> Iterator[str]`, which yields
text deltas as the server produces them (the pipeline speaks each sentence
as soon as it is complete). Sampling params are pinned identically on both
so cross-environment differences come from the runtime, not the sampler.
`max_tokens` overrides the shared cap for one call — used by
pipeline_factory.prewarm_components() to send a 1-token request that makes
the server load its weights and cache the system prompt. `stop` (constructor)
lists strings that end generation as soon as the model writes one;
pipeline_factory passes prompt_builder.PROMPT_HEADINGS, so a model that
starts copying its prompt is cut off by the server.

This module deliberately has no dependencies on any other RoboRanger
module — anything that needs an LLM client can import from here without
dragging in the corpus, prompt builder, or eval fixtures.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Iterator, Sequence


# Shared sampling params. Pin these on both backends so cross-environment
# differences come from the model runtime, not from sampler defaults.
# max_tokens=100 is the UNO Q budget (200 was too slow there); Ollama used to
# get 200 while llama.cpp hard-coded 100, so laptop evals didn't match the
# device's answer length.
SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 42,
    "max_tokens": 100,
}


class OllamaBackend:
    """Talks to Ollama's /api/chat. Default host: localhost:11434."""

    name = "ollama"

    def __init__(self, model: str = "smollm2:360m",
                 host: str = "http://localhost:11434",
                 stop: Sequence[str] | None = None) -> None:
        self.model = model
        self.host = host
        self.stop = list(stop) if stop else None

    def _request(self, messages: list[dict], max_tokens: int | None,
                 stream: bool) -> urllib.request.Request:
        options = {
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
            "seed": SAMPLING["seed"],
            "num_predict": max_tokens or SAMPLING["max_tokens"],
        }
        if self.stop:
            options["stop"] = self.stop
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }).encode()
        return urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

    def generate(self, messages: list[dict], *,
                 max_tokens: int | None = None) -> str:
        req = self._request(messages, max_tokens, stream=False)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]

    def stream(self, messages: list[dict], *,
               max_tokens: int | None = None) -> Iterator[str]:
        """Yield content deltas from Ollama's NDJSON stream."""
        req = self._request(messages, max_tokens, stream=True)
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"ollama: {data['error']}")
                content = (data.get("message") or {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    return


class LlamaCppBackend:
    """
    Talks to llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.
    Start the server on the Uno Q with (see onDeviceAgentREADME.md):
        ./llama-server -m smollm2-360m-q8_0.gguf --host 0.0.0.0 --port 8080 \
                       --ctx-size 4096 -t 3 -tb 4 -np 1 --chat-template chatml

    Every request is pinned to one server slot (`slot`, default 0).
    llama-server runs several slots by default, and a slot is also its
    prompt cache: unpinned, a question that arrives during the startup
    warmup lands in a different slot, prefills its whole prompt from scratch
    and splits the CPU with the warmup. Pinned, the server defers it until
    the slot is free, then reuses the prefix the warmup cached. None leaves
    slot choice to the server.
    """

    name = "llama-cpp"

    def __init__(self, host: str = "http://localhost:8080",
                 model: str = "smollm2", slot: int | None = 0,
                 stop: Sequence[str] | None = None) -> None:
        self.host = host
        self.model = model
        self.slot = slot
        self.stop = list(stop) if stop else None

    def _request(self, messages: list[dict], max_tokens: int | None,
                 stream: bool) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
            "seed": SAMPLING["seed"],
            "max_tokens": max_tokens or SAMPLING["max_tokens"],
            # Reuse the KV cache for a shared prompt prefix (the system
            # prompt + SPECIES FACTS the startup warmup already processed).
            # Default on in current llama-server; explicit for older builds.
            "cache_prompt": True,
        }
        if self.slot is not None:
            body["id_slot"] = self.slot
        if self.stop:
            body["stop"] = self.stop
        return urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )

    def generate(self, messages: list[dict], *,
                 max_tokens: int | None = None) -> str:
        req = self._request(messages, max_tokens, stream=False)
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], *,
               max_tokens: int | None = None) -> Iterator[str]:
        """Yield content deltas from the server-sent-events stream."""
        req = self._request(messages, max_tokens, stream=True)
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    return
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError(f"llama.cpp: {data['error']}")
                choices = data.get("choices") or [{}]
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content