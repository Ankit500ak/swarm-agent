"""Tests for Level 1: the OpenAI-compatible serve gateway.

The gateway is exercised over real HTTP on an ephemeral localhost
port, with a fake chat backend injected - no Ollama peer needed.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Optional

import pytest

from swarm_agent import server as server_mod
from swarm_agent.config import Config
from swarm_agent.ollama import SwarmError


def _reply(messages: list, model: Optional[str]) -> str:
    last = messages[-1]["content"] if messages else ""
    return f"echo: {last}"


def _down(messages: list, model: Optional[str]) -> str:
    raise SwarmError("no Ollama peer found - start one with: ollama serve")


@pytest.fixture()
def live():
    """Start a gateway on an ephemeral port; yield (base_url, factory)."""
    servers = []

    def start(backend=_reply, api_key=None):
        httpd = server_mod.make_server(
            Config(model="qwen2.5-coder:7b"),
            host="127.0.0.1",
            port=0,
            backend=backend,
            api_key=api_key,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append(httpd)
        host, port = httpd.server_address[:2]
        return f"http://{host}:{port}"

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def _request(url: str, payload=None, headers=None, method=None):
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get_content_type(), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read().decode("utf-8")


# -- routes ---------------------------------------------------------------------


def test_health_and_models(live) -> None:
    base = live()
    status, _, body = _request(base + "/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"

    status, _, body = _request(base + "/v1/models")
    assert status == 200
    data = json.loads(body)
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "qwen2.5-coder:7b"
    assert data["data"][0]["owned_by"] == "swarm-agent"


def test_chat_completion_shape(live) -> None:
    base = live()
    status, ctype, body = _request(
        base + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello swarm"}]},
    )
    assert status == 200 and ctype == "application/json"
    data = json.loads(body)
    assert data["object"] == "chat.completion"
    assert data["model"] == "qwen2.5-coder:7b"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "echo: hello swarm"
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] > 0


def test_chat_streaming_sse(live) -> None:
    base = live()
    status, ctype, body = _request(
        base + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "stream me"}], "stream": True},
    )
    assert status == 200 and ctype == "text/event-stream"
    assert "echo: stream me" in body
    assert '"finish_reason": "stop"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_unknown_route_404(live) -> None:
    base = live()
    status, _, body = _request(base + "/v1/nope")
    assert status == 404
    assert "error" in json.loads(body)


# -- validation & backend failures -------------------------------------------------


def test_bad_requests_400(live) -> None:
    base = live()
    status, _, _ = _request(base + "/v1/chat/completions", {"messages": []})
    assert status == 400
    status, _, _ = _request(
        base + "/v1/chat/completions", {"messages": [{"role": "user"}]}
    )
    assert status == 400
    status, _, _ = _request(
        base + "/v1/chat/completions", payload=b"{not json"
    )
    assert status == 400


def test_backend_down_is_503(live) -> None:
    base = live(backend=_down)
    status, _, body = _request(
        base + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 503
    assert "ollama serve" in json.loads(body)["error"]["message"]


# -- auth ---------------------------------------------------------------------------


def test_api_key_required_on_v1_but_health_open(live) -> None:
    base = live(api_key="sekrit")
    status, _, _ = _request(base + "/health")
    assert status == 200  # probes stay open

    status, _, _ = _request(base + "/v1/models")
    assert status == 401
    status, _, _ = _request(
        base + "/v1/models", headers={"Authorization": "Bearer wrong"}
    )
    assert status == 401
    status, _, body = _request(
        base + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert status == 200
    assert "echo: hi" in body


# -- pure helpers ---------------------------------------------------------------------


def test_opencode_snippet_points_at_gateway() -> None:
    snippet = json.loads(server_mod.opencode_snippet("192.168.1.20", 8765, "m1"))
    provider = snippet["provider"]["swarm"]
    assert provider["options"]["baseURL"] == "http://192.168.1.20:8765/v1"
    assert "m1" in provider["models"]
    # A wildcard bind is shown as localhost in the snippet.
    snippet = json.loads(server_mod.opencode_snippet("0.0.0.0", 9000, "m1"))
    assert "127.0.0.1:9000" in snippet["provider"]["swarm"]["options"]["baseURL"]


def test_validate_messages_rejects_garbage() -> None:
    with pytest.raises(SwarmError):
        server_mod.validate_messages({})
    with pytest.raises(SwarmError):
        server_mod.validate_messages({"messages": "nope"})
    ok = server_mod.validate_messages(
        {"messages": [{"role": "user", "content": "x", "extra": 1}]}
    )
    assert ok == [{"role": "user", "content": "x"}]
