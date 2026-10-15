"""
Unit tests for llm_backends.py: request bodies and stream parsing.

An in-process HTTP server plays Ollama (/api/chat, NDJSON when streaming)
and llama.cpp (/v1/chat/completions, server-sent events when streaming)
with canned replies, so no model server is needed.

Run from anywhere:
    python -m pytest species_identification/tests/test_llm_backends.py
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "llm-tuning"))
from llm_backends import SAMPLING, LlamaCppBackend, OllamaBackend  # noqa: E402

MESSAGES = [{"role": "user", "content": "what does it eat?"}]


def _ndjson(objs: list[dict]) -> str:
    return "".join(json.dumps(o) + "\n" for o in objs)


def _sse(objs: list[dict | str]) -> str:
    return "".join(
        f"data: {o if isinstance(o, str) else json.dumps(o)}\n\n" for o in objs)


# Canned replies keyed by (path, streaming). A test can override one by
# assigning server.replies[key].
DEFAULT_REPLIES = {
    ("/api/chat", False): json.dumps(
        {"message": {"role": "assistant", "content": "Seeds."}, "done": True}),
    ("/api/chat", True): _ndjson([
        {"message": {"role": "assistant", "content": "It eats"}, "done": False},
        {"message": {"role": "assistant", "content": " seeds."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "done_reason": "length"},
        # Anything after done must be ignored.
        {"message": {"role": "assistant", "content": " ignored"},
         "done": False},
    ]),
    ("/v1/chat/completions", False): json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "Seeds."}}]}),
    ("/v1/chat/completions", True): _sse([
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "It eats"}}]},
        {"choices": [{"delta": {"content": " seeds."}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        "[DONE]",
        {"choices": [{"delta": {"content": " ignored"}}]},
    ]),
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        payload = self.server.replies[(self.path, bool(body.get("stream")))]
        data = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests = []
    srv.replies = dict(DEFAULT_REPLIES)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()
    srv.server_close()


class TestOllama:

    def test_generate(self, server):
        backend = OllamaBackend(model="m", host=server.url)
        assert backend.generate(MESSAGES) == "Seeds."
        path, body = server.requests[0]
        assert path == "/api/chat" and body["stream"] is False
        assert body["options"]["num_predict"] == SAMPLING["max_tokens"]

    def test_stream_yields_deltas_and_stops_at_done(self, server):
        backend = OllamaBackend(model="m", host=server.url)
        assert list(backend.stream(MESSAGES)) == ["It eats", " seeds."]
        _, body = server.requests[0]
        assert body["stream"] is True
        assert body["messages"] == MESSAGES

    def test_stop_strings_sent(self, server):
        OllamaBackend(model="m", host=server.url,
                      stop=("SPECIES FACTS:",)).generate(MESSAGES)
        assert server.requests[0][1]["options"]["stop"] == ["SPECIES FACTS:"]

    def test_stream_max_tokens_override(self, server):
        backend = OllamaBackend(model="m", host=server.url)
        list(backend.stream(MESSAGES, max_tokens=7))
        assert server.requests[0][1]["options"]["num_predict"] == 7

    def test_stream_error_line_raises(self, server):
        server.replies[("/api/chat", True)] = _ndjson(
            [{"error": "model not found"}])
        backend = OllamaBackend(model="m", host=server.url)
        with pytest.raises(RuntimeError, match="model not found"):
            list(backend.stream(MESSAGES))


class TestLlamaCpp:

    def test_generate_sends_cache_prompt(self, server):
        backend = LlamaCppBackend(host=server.url, model="m")
        assert backend.generate(MESSAGES) == "Seeds."
        path, body = server.requests[0]
        assert path == "/v1/chat/completions" and body["stream"] is False
        assert body["cache_prompt"] is True
        assert body["max_tokens"] == SAMPLING["max_tokens"]

    def test_requests_pinned_to_slot_zero_by_default(self, server):
        backend = LlamaCppBackend(host=server.url, model="m")
        backend.generate(MESSAGES)
        list(backend.stream(MESSAGES))
        assert [body["id_slot"] for _, body in server.requests] == [0, 0]

    def test_stop_strings_sent(self, server):
        LlamaCppBackend(host=server.url, model="m",
                        stop=("SPECIES FACTS:",)).generate(MESSAGES)
        LlamaCppBackend(host=server.url, model="m").generate(MESSAGES)
        assert server.requests[0][1]["stop"] == ["SPECIES FACTS:"]
        assert "stop" not in server.requests[1][1]

    def test_slot_none_leaves_choice_to_server(self, server):
        backend = LlamaCppBackend(host=server.url, model="m", slot=None)
        backend.generate(MESSAGES)
        assert "id_slot" not in server.requests[0][1]

    def test_stream_yields_deltas_and_stops_at_done(self, server):
        backend = LlamaCppBackend(host=server.url, model="m")
        assert list(backend.stream(MESSAGES)) == ["It eats", " seeds."]
        _, body = server.requests[0]
        assert body["stream"] is True and body["cache_prompt"] is True

    def test_closing_the_stream_closes_the_connection(self, server):
        # The pipeline stops reading early (a long enough answer) and calls
        # close(); the HTTP response must be released so the server sees the
        # disconnect and stops generating.
        backend = LlamaCppBackend(host=server.url, model="m")
        deltas = backend.stream(MESSAGES)
        assert next(deltas) == "It eats"
        resp = deltas.gi_frame.f_locals["resp"]
        deltas.close()
        assert resp.closed

    def test_stream_error_event_raises(self, server):
        server.replies[("/v1/chat/completions", True)] = _sse(
            [{"error": {"code": 500, "message": "context overflow"}}])
        backend = LlamaCppBackend(host=server.url, model="m")
        with pytest.raises(RuntimeError, match="context overflow"):
            list(backend.stream(MESSAGES))
