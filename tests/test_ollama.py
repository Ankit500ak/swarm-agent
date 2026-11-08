import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from swarm_agent.ollama import OllamaClient, SwarmError


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = b"Ollama is running"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/tags":
            self._json({"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:8b"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/chat":
            self._json({"message": {"role": "assistant",
                                    "content": f"echo:{payload['messages'][-1]['content']}"},
                        "done": True})
        elif self.path == "/api/generate":
            self._json({"response": f"gen:{payload['prompt']}", "done": True})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_tags(server):
    client = OllamaClient(server, default_model="qwen2.5-coder:7b")
    assert client.tags() == ["qwen2.5-coder:7b", "llama3:8b"]


def test_healthy(server):
    assert OllamaClient(server).healthy() is True


def test_chat(server):
    client = OllamaClient(server, default_model="qwen2.5-coder:7b")
    out = client.chat([{"role": "user", "content": "hello"}])
    assert out == "echo:hello"


def test_generate(server):
    client = OllamaClient(server, default_model="qwen2.5-coder:7b")
    assert client.generate("ping") == "gen:ping"


def test_unreachable_raises_swarm_error():
    client = OllamaClient("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(SwarmError):
        client.tags(timeout=1.0)
    assert client.healthy(timeout=1.0) is False
