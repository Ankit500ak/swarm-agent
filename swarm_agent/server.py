"""Level 1 - the serve gateway: the swarm as an OpenAI-compatible API.

``swarm-agent serve`` puts one HTTP endpoint in front of the whole
LAN swarm, speaking the OpenAI chat-completions dialect:

- ``GET  /health``                 - liveness + the model on offer
- ``GET  /v1/models``              - the model list (one entry)
- ``POST /v1/chat/completions``    - chat, non-streaming or SSE streaming

Any OpenAI-compatible client can use the swarm through it - most
usefully **OpenCode**, which accepts custom providers: point it at
``http://<this-host>:8765/v1`` and the IDE's agent loop runs on your
own machines, code never leaving the network.

Standard library only (``http.server``), like the rest of the
project. A request is served by the distributor: fastest live peer,
failover handled there, exactly as the CLI flows do it. The backend
(distributor) is built lazily on the first chat request, so
``/health`` and ``/v1/models`` answer even with no peer up - chat
then returns 503 with the swarm's own error message until a peer
appears.

Optional shared-secret auth: with an API key set (``--api-key`` or
``SWARM_SERVE_API_KEY``), ``/v1/*`` requires
``Authorization: Bearer <key>``. ``/health`` stays open so monitors
can probe liveness without the secret. Default bind is localhost;
binding a LAN address is an explicit choice (``--host 0.0.0.0``) -
pair it with an API key.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .config import Config
from .ollama import SwarmError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# A chat backend answers (messages, model) with reply text. Tests and
# embeddings inject fakes; production uses the distributor, lazily.
ChatBackend = Callable[[list, Optional[str]], str]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for usage accounting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def validate_messages(payload: object) -> list[dict]:
    """Extract and validate the ``messages`` array, or raise SwarmError."""
    if not isinstance(payload, dict):
        raise SwarmError("request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SwarmError("'messages' must be a non-empty array")
    cleaned: list[dict] = []
    for item in messages:
        if not isinstance(item, dict):
            raise SwarmError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise SwarmError("each message needs a non-empty 'role'")
        if not isinstance(content, str):
            raise SwarmError("each message needs a string 'content'")
        cleaned.append({"role": role, "content": content})
    return cleaned


def completion_object(model: str, reply: str, messages: list[dict]) -> dict:
    """Build an OpenAI ``chat.completion`` response object."""
    prompt_text = "\n".join(m.get("content", "") for m in messages)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": estimate_tokens(prompt_text),
            "completion_tokens": estimate_tokens(reply),
            "total_tokens": estimate_tokens(prompt_text) + estimate_tokens(reply),
        },
    }


def stream_chunks(model: str, reply: str) -> list[dict]:
    """Build the SSE chunk sequence for a (complete) reply.

    The swarm's backends answer whole replies, so streaming here is
    transport-level: one content chunk carrying the reply, then the
    terminal chunk with ``finish_reason: stop``. Clients see a valid
    OpenAI stream either way.
    """
    base = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    first = dict(base)
    first["choices"] = [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": reply},
            "finish_reason": None,
        }
    ]
    last = dict(base)
    last["id"] = first["id"]
    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    return [first, last]


class _BackendHolder:
    """Lazily build the real distributor-backed backend, thread-safely."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._backend: Optional[ChatBackend] = None
        self._lock = threading.Lock()

    def __call__(self, messages: list, model: Optional[str]) -> str:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    from . import agent  # deferred: discovery is heavy

                    distributor = agent.get_distributor(self._config)

                    def _ask(msgs: list, mdl: Optional[str], _d=distributor) -> str:
                        return _d.ask(msgs, model=mdl)

                    self._backend = _ask
        return self._backend(messages, model)


def make_handler(
    model: str,
    backend: ChatBackend,
    api_key: Optional[str] = None,
):
    """Build a request-handler class bound to one backend + model."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "swarm-agent"

        # -- plumbing ------------------------------------------------------
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # quiet: the CLI prints its own startup banner

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
            self._send_json(
                status,
                {"error": {"message": message, "type": err_type, "code": None}},
            )

        def _authed(self) -> bool:
            if not api_key:
                return True
            header = self.headers.get("Authorization") or ""
            return header == f"Bearer {api_key}"

        # -- routes ----------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path in ("/", "/health"):
                self._send_json(200, {"status": "ok", "model": model})
                return
            if self.path == "/v1/models":
                if not self._authed():
                    self._error(401, "missing or wrong API key", "authentication_error")
                    return
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "swarm-agent",
                            }
                        ],
                    },
                )
                return
            self._error(404, f"no such route: {self.path}", "not_found")

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path != "/v1/chat/completions":
                self._error(404, f"no such route: {self.path}", "not_found")
                return
            if not self._authed():
                self._error(401, "missing or wrong API key", "authentication_error")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError):
                self._error(400, "request body is not valid JSON")
                return
            try:
                messages = validate_messages(payload)
            except SwarmError as exc:
                self._error(400, str(exc))
                return
            requested_model = payload.get("model") or model
            try:
                reply = backend(messages, requested_model)
            except SwarmError as exc:
                self._error(503, str(exc), "service_unavailable")
                return
            except Exception as exc:  # a backend failure is an answer too
                self._error(500, f"{type(exc).__name__}: {exc}", "internal_error")
                return
            if payload.get("stream"):
                self._send_stream(requested_model, reply)
            else:
                self._send_json(200, completion_object(requested_model, reply, messages))

        def _send_stream(self, requested_model: str, reply: str) -> None:
            chunks = stream_chunks(requested_model, reply)
            body = "".join(
                f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
            ) + "data: [DONE]\n\n"
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def make_server(
    config: Config,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    backend: Optional[ChatBackend] = None,
    api_key: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Create (but do not start) the gateway HTTP server."""
    if backend is None:
        backend = _BackendHolder(config)
    handler = make_handler(config.model, backend, api_key=api_key)
    return ThreadingHTTPServer((host, port), handler)


def opencode_snippet(host: str, port: int, model: str) -> str:
    """The OpenCode provider config that points the IDE at this gateway."""
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return json.dumps(
        {
            "provider": {
                "swarm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Swarm (LAN)",
                    "options": {"baseURL": f"http://{shown_host}:{port}/v1"},
                    "models": {model: {"name": model}},
                }
            }
        },
        indent=2,
    )


def serve(
    config: Config,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    api_key: Optional[str] = None,
) -> int:
    """Run the gateway in the foreground. Returns a process exit code."""
    key = api_key or os.environ.get("SWARM_SERVE_API_KEY") or None
    server = make_server(config, host=host, port=port, api_key=key)
    bound_host, bound_port = server.server_address[:2]
    print(f"swarm-agent serving the swarm as an OpenAI-compatible API", flush=True)
    print(f"  endpoint : http://{bound_host}:{bound_port}/v1", flush=True)
    print(f"  model    : {config.model}", flush=True)
    print(f"  auth     : {'API key required' if key else 'open (localhost is the safe default)'}", flush=True)
    print(f"\nPoint OpenCode at it - add to opencode.json:\n{opencode_snippet(bound_host, bound_port, config.model)}", flush=True)
    print("\nCtrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
