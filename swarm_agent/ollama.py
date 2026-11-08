"""HTTP clients for swarm peers, stdlib-only (``urllib``).

Two peer flavours are supported:

- :class:`OllamaClient` - a stock Ollama server (``/api/tags``, ``/api/chat``,
  ``/api/generate``).
- :class:`MLXClient` - an ``mlx_lm.server`` instance, which exposes an
  OpenAI-compatible ``/v1/chat/completions`` endpoint (used on Apple Silicon).

Both expose the same small surface - ``tags()``, ``chat()``, ``healthy()`` -
so the distributor can treat them interchangeably.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


class SwarmError(Exception):
    """A user-facing failure (unreachable peer, bad response, no peers...)."""


class OllamaClient:
    """Client for a single Ollama server."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        default_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model

    # -- low-level helpers -------------------------------------------------
    def _request(
        self,
        path: str,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ):
        url = self.base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(
                req, timeout=timeout if timeout is not None else self.timeout
            ) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:  # server answered, but badly
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise SwarmError(
                f"Ollama at {self.base_url} returned HTTP {exc.code} "
                f"for {path}: {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise SwarmError(
                f"cannot reach Ollama at {self.base_url} ({path}): {exc}"
            ) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            # Some endpoints (GET /) return plain text.
            return raw.decode("utf-8", "replace")

    def _model(self, model: Optional[str]) -> str:
        chosen = model or self.default_model
        if not chosen:
            raise SwarmError("no model specified for this Ollama client")
        return chosen

    # -- public API ----------------------------------------------------------
    def healthy(self, timeout: float = 2.0) -> bool:
        """True if the server answers ``GET /`` quickly."""
        try:
            self._request("/", timeout=timeout)
            return True
        except SwarmError:
            return False

    def tags(self, timeout: Optional[float] = None) -> list[str]:
        """Names of the models this server has pulled."""
        data = self._request("/api/tags", timeout=timeout)
        if not isinstance(data, dict):
            raise SwarmError(f"unexpected /api/tags response from {self.base_url}")
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def chat(self, messages: list[dict], model: Optional[str] = None) -> str:
        """One non-streaming chat round-trip; returns the reply text."""
        payload = {
            "model": self._model(model),
            "messages": messages,
            "stream": False,
        }
        data = self._request("/api/chat", payload=payload)
        if not isinstance(data, dict) or "message" not in data:
            raise SwarmError(f"unexpected /api/chat response from {self.base_url}")
        return data["message"].get("content", "")

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Single-prompt completion via ``/api/generate``."""
        payload = {"model": self._model(model), "prompt": prompt, "stream": False}
        data = self._request("/api/generate", payload=payload)
        if not isinstance(data, dict) or "response" not in data:
            raise SwarmError(
                f"unexpected /api/generate response from {self.base_url}"
            )
        return data["response"]

    def embed(
        self, texts: list[str], model: Optional[str] = None
    ) -> list[list[float]]:
        """Embed texts via ``/api/embed``; returns one vector per input text."""
        payload = {"model": self._model(model), "input": texts}
        data = self._request("/api/embed", payload=payload)
        if not isinstance(data, dict) or "embeddings" not in data:
            raise SwarmError(
                f"unexpected /api/embed response from {self.base_url}"
            )
        vectors = data["embeddings"]
        if len(vectors) != len(texts):
            raise SwarmError(
                f"/api/embed from {self.base_url} returned {len(vectors)} "
                f"vectors for {len(texts)} texts"
            )
        return [[float(x) for x in vec] for vec in vectors]


class MLXClient:
    """Client for an ``mlx_lm.server`` (OpenAI-compatible) peer."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        default_model: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model

    def _request(self, path: str, payload: Optional[dict] = None,
                 timeout: Optional[float] = None):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(
                req, timeout=timeout if timeout is not None else self.timeout
            ) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise SwarmError(
                f"MLX server at {self.base_url} returned HTTP {exc.code} for {path}"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise SwarmError(
                f"cannot reach MLX server at {self.base_url} ({path}): {exc}"
            ) from exc
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError as exc:
            raise SwarmError(
                f"invalid JSON from MLX server at {self.base_url}"
            ) from exc

    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            self._request("/v1/models", timeout=timeout)
            return True
        except SwarmError:
            return False

    def tags(self, timeout: Optional[float] = None) -> list[str]:
        data = self._request("/v1/models", timeout=timeout)
        if not isinstance(data, dict):
            return []
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

    def chat(self, messages: list[dict], model: Optional[str] = None) -> str:
        payload = {
            "model": model or self.default_model or "mlx",
            "messages": messages,
        }
        data = self._request("/v1/chat/completions", payload=payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SwarmError(
                f"unexpected /v1/chat/completions response from {self.base_url}"
            ) from exc


def client_for_peer(peer, model: Optional[str] = None):
    """Build the right client for a discovered :class:`~.discovery.Peer`.

    The effective default model is resolved per peer: the requested model if
    the peer serves it, otherwise the first model the peer does serve (so a
    unit of work is never sent asking for a model the peer does not have).
    """
    models = list(getattr(peer, "models", []) or [])
    if model and model in models:
        effective = model
    elif models:
        effective = models[0]
    else:
        effective = model
    if getattr(peer, "kind", "ollama") == "mlx":
        return MLXClient(peer.base_url, default_model=effective)
    return OllamaClient(peer.base_url, default_model=effective)
