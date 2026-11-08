"""Configuration for swarm-agent.

Load order (later sources win):

1. Built-in defaults.
2. ``~/.config/swarm-agent/config.json`` (if it exists).
3. Environment variables:

   - ``SWARM_MODEL``  - model name to request from peers.
   - ``SWARM_PEERS``  - comma-separated list of peer base URLs (replaces the
     configured peer list).
   - ``OLLAMA_HOST``  - treated as the *primary* peer and prepended to the
     peer list (Ollama's own convention).

Peer URLs may be written with or without a scheme (``localhost:11434`` is
normalised to ``http://localhost:11434``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_PEER = "http://localhost:11434"
DEFAULT_CONFIG_PATH = Path("~/.config/swarm-agent/config.json").expanduser()


def normalize_url(url: str) -> str:
    """Normalise a peer URL: add a scheme if missing, drop trailing slash."""
    url = url.strip()
    if not url:
        return url
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def _split_peers(raw: str) -> list[str]:
    return [normalize_url(p) for p in raw.split(",") if p.strip()]


@dataclass
class Config:
    """Runtime configuration for swarm-agent."""

    model: str = DEFAULT_MODEL
    peers: list[str] = field(default_factory=lambda: [DEFAULT_PEER])
    chunk_tokens: int = 1500
    request_timeout: float = 120.0
    peer_timeout: float = 2.0
    scan_timeout: float = 0.4
    test_command: Optional[str] = "python -m pytest -q"
    auto_discover: bool = False
    mlx_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.peers = [normalize_url(p) for p in self.peers if p and p.strip()]
        if self.mlx_url:
            self.mlx_url = normalize_url(self.mlx_url)


def _from_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    known = {f.name for f in fields(Config)}
    return {k: v for k, v in data.items() if k in known}


def load_config(
    config_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Config:
    """Build a :class:`Config` from defaults, the config file and the env."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if env is None:
        env = os.environ

    values: dict = {}
    if config_path is not None and Path(config_path).exists():
        values.update(_from_file(Path(config_path)))

    cfg = Config(**values)

    if env.get("SWARM_MODEL"):
        cfg.model = env["SWARM_MODEL"].strip()
    if env.get("SWARM_PEERS"):
        cfg.peers = _split_peers(env["SWARM_PEERS"])
    if env.get("OLLAMA_HOST"):
        primary = normalize_url(env["OLLAMA_HOST"])
        # OLLAMA_HOST names the primary peer: put it first, de-duplicated.
        cfg.peers = [primary] + [p for p in cfg.peers if p != primary]
    if env.get("SWARM_MLX_URL"):
        cfg.mlx_url = normalize_url(env["SWARM_MLX_URL"])

    cfg.__post_init__()
    return cfg
