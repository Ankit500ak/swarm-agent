"""Finding the swarm: which machines on the LAN can run inference?

Discovery starts from the configured peers. With ``auto_discover`` enabled it
also scans the local /24 subnet for Ollama's port (11434), and it picks up an
MLX server (``mlx_lm.server``, port 8080) when one is configured or already
listening on localhost. Everything is probed with short timeouts and the port
scan runs concurrently, so discovery stays fast even on a quiet network.
"""

from __future__ import annotations

import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .config import Config, normalize_url
from .ollama import MLXClient, OllamaClient

OLLAMA_PORT = 11434
MLX_PORT = 8080


@dataclass
class Peer:
    """One inference peer on the local network."""

    base_url: str
    models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    kind: str = "ollama"  # "ollama" | "mlx"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.base_url} ({self.kind}, {self.latency_ms:.0f} ms)"


# -- probes (module-level so tests can monkeypatch them) ---------------------

def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fetch_ollama_peer(base_url: str, timeout: float) -> Optional[Peer]:
    client = OllamaClient(base_url)
    start = time.perf_counter()
    try:
        models = client.tags(timeout=timeout)
    except Exception:
        return None
    latency = (time.perf_counter() - start) * 1000.0
    return Peer(base_url=base_url, models=models, latency_ms=latency, kind="ollama")


def _fetch_mlx_peer(base_url: str) -> Optional[Peer]:
    host = base_url.split("://", 1)[-1].split(":")[0]
    if not _tcp_open(host, MLX_PORT, timeout=1.0):
        return None
    client = MLXClient(base_url)
    start = time.perf_counter()
    try:
        models = client.tags(timeout=2.0)
    except Exception:
        models = []
    latency = (time.perf_counter() - start) * 1000.0
    return Peer(base_url=base_url, models=models, latency_ms=latency, kind="mlx")


def _local_ip() -> str:
    """Best-effort LAN IP of this machine (no traffic is actually sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def _scan_subnet(config: Config) -> list[Peer]:
    ip = _local_ip()
    parts = ip.split(".")
    if len(parts) != 4:
        return []
    prefix = ".".join(parts[:3])
    hosts = [f"{prefix}.{i}" for i in range(1, 255)]

    def probe(host: str) -> Optional[str]:
        return host if _tcp_open(host, OLLAMA_PORT, config.scan_timeout) else None

    responders: list[str] = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        for hit in pool.map(probe, hosts):
            if hit:
                responders.append(hit)

    peers: list[Peer] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for peer in pool.map(
            lambda h: _fetch_ollama_peer(f"http://{h}:{OLLAMA_PORT}", config.peer_timeout),
            responders,
        ):
            if peer is not None:
                peers.append(peer)
    return peers


# -- main entry point ----------------------------------------------------------

def discover(config: Config) -> list[Peer]:
    """Return the reachable peers, fastest first, de-duplicated by URL."""
    found: dict[str, Peer] = {}

    for url in config.peers:
        url = normalize_url(url)
        peer = _fetch_ollama_peer(url, config.peer_timeout)
        if peer is not None:
            found[peer.base_url] = peer

    if config.auto_discover:
        for peer in _scan_subnet(config):
            found.setdefault(peer.base_url, peer)

    mlx_url = config.mlx_url
    if not mlx_url and _tcp_open("127.0.0.1", MLX_PORT, timeout=0.5):
        mlx_url = f"http://localhost:{MLX_PORT}"
    if mlx_url:
        peer = _fetch_mlx_peer(normalize_url(mlx_url))
        if peer is not None:
            found.setdefault(peer.base_url, peer)

    return sorted(found.values(), key=lambda p: p.latency_ms)
