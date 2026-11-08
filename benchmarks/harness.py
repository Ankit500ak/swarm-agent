"""Benchmark harness: analyse a pinned sample repo against 1, 2, 3 peers.

The project's core claim is that a LAN swarm is faster. This module
measures it: it runs the real :func:`swarm_agent.agent.analyze_flow`
over the pinned sample repo (``benchmarks/sample_repo`` — 4 files,
16 chunks at the default 1500-token budget) with the first 1, then
2, then 3 configured Ollama peers, and records wall time, chunks
summarised and per-peer task counts.

The numbers come from the distributor itself: :func:`run_once`
captures the distributor the flow builds (via a wrapper around
:func:`swarm_agent.agent.get_distributor`) and reads its ``counts``
— one entry per completed chunk, keyed by peer URL. Chunks are
chunks actually summarised, not chunks attempted.

Honest numbers only. The harness records what actually ran: if an
MLX server is listening on localhost:8080 it joins discovery
automatically (that is ``discover``'s behaviour, not the harness's)
and shows up in the per-peer column — stop it for a pure Ollama
comparison. A run that raises is recorded as ``failed`` with the
error, never silently dropped. No row is ever invented: with no
peers the harness runs nothing and writes nothing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from swarm_agent import agent, backends
from swarm_agent.config import Config

SAMPLE_REPO = Path(__file__).parent / "sample_repo"
RESULTS_PATH = Path(__file__).parent / "RESULTS.md"
MAX_PEERS = 3

RESULTS_HEADER = """# Benchmark results — analyse, 1 vs 2 vs 3 peers

Methodology: see [README.md](README.md). Each row is one run of
`analyze` over the pinned sample repo (4 files, 16 chunks at the
default 1500-token budget). Wall time covers the whole analyse call,
including the final synthesis. Chunks are chunks actually summarised
(the distributor's per-peer counts summed). Per-peer tasks are
`host:count` pairs in peer order.

| date | machine | model | ollama peers | wall (s) | chunks | chunks/min | per-peer tasks |
|---|---|---|---|---|---|---|---|
"""


def plan_peer_counts(peer_urls: list[str], max_peers: int = MAX_PEERS) -> list[int]:
    """Peer counts to benchmark: 1..min(max_peers, peers given)."""
    n = min(max_peers, len(peer_urls))
    return list(range(1, n + 1))


def chunks_per_minute(chunks: int, seconds: float) -> float:
    """Throughput of completed chunks; 0 when nothing ran."""
    if chunks <= 0 or seconds <= 0:
        return 0.0
    return chunks / (seconds / 60.0)


def machine_label() -> str:
    """Short machine description for the results table."""
    report = backends.hardware_report()
    chip = report.get("chip") or report.get("machine") or "unknown"
    mem = report.get("unified_memory_gb") or report.get("memory_gb")
    return f"{chip}, {mem} GB" if mem is not None else str(chip)


def _short_host(url: str) -> str:
    return url.split("://", 1)[-1]


def format_row(result: dict) -> str:
    """One markdown table row for a run's metrics dict."""
    if result["status"] == "failed":
        wall = chunks = cpm = "—"
        tasks = f"failed: {result['error']}"
    else:
        wall = f"{result['wall_s']:.1f}"
        chunks = str(result["chunks"])
        cpm = f"{result['chunks_per_min']:.1f}"
        tasks = result["per_peer"]
    return (
        f"| {result['date']} | {result['machine']} | {result['model']} | "
        f"{result['ollama_peers']} | {wall} | {chunks} | {cpm} | {tasks} |"
    )


def append_results(path: str | Path, rows: list[str]) -> None:
    """Append rows to the results table, creating it with a header first."""
    path = Path(path)
    if not path.exists():
        path.write_text(RESULTS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row + "\n")


@contextmanager
def capture_distributor():
    """Capture the distributor :func:`agent.analyze_flow` builds.

    The flow does not return its distributor, and the per-peer task
    counts live on it — so this wrapper records the instance the
    flow creates, then restores the original function. It changes
    nothing about the run itself.
    """
    captured: dict = {}
    original = agent.get_distributor

    def wrapper(config):
        dist = original(config)
        captured["distributor"] = dist
        return dist

    agent.get_distributor = wrapper
    try:
        yield captured
    finally:
        agent.get_distributor = original


def run_once(
    peer_urls: list[str],
    model: str,
    sample_repo: str | Path = SAMPLE_REPO,
) -> dict:
    """Analyse the sample repo with the given peers; return metrics.

    ``peer_urls`` are the Ollama peers to configure (the first N of
    the benchmark's list). The distributor the flow builds is the
    source of truth for what actually ran: its ``peers`` and
    ``counts``. A run that raises is returned as ``status: failed``
    with the error, so a failure is visible in the table.
    """
    cfg = Config(model=model, peers=list(peer_urls))
    started = time.perf_counter()
    try:
        with capture_distributor() as captured:
            agent.analyze_flow(sample_repo, cfg)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {
            "date": datetime.now().isoformat(timespec="seconds"),
            "machine": machine_label(),
            "model": model,
            "ollama_peers": len(peer_urls),
            "wall_s": time.perf_counter() - started,
            "chunks": None,
            "chunks_per_min": None,
            "per_peer": "",
            "status": "failed",
            "error": str(exc),
        }
    wall = time.perf_counter() - started
    dist = captured.get("distributor")
    counts = dict(dist.counts) if dist is not None else {}
    peers = list(dist.peers) if dist is not None else []
    chunks = sum(counts.values())
    per_peer = ", ".join(
        f"{_short_host(p.base_url)}:{counts.get(p.base_url, 0)}" for p in peers
    )
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": machine_label(),
        "model": model,
        "ollama_peers": len(peer_urls),
        "wall_s": wall,
        "chunks": chunks,
        "chunks_per_min": chunks_per_minute(chunks, wall),
        "per_peer": per_peer,
        "status": "ok",
        "error": "",
    }


def run_benchmark(
    peer_urls: list[str],
    model: str,
    sample_repo: str | Path = SAMPLE_REPO,
    results_path: str | Path = RESULTS_PATH,
    max_peers: int = MAX_PEERS,
) -> list[dict]:
    """Run the 1 vs 2 vs 3 comparison and append the rows to RESULTS.md.

    Returns the list of metrics dicts (one per peer count). With no
    peers given, runs nothing and writes nothing.
    """
    counts = plan_peer_counts(peer_urls, max_peers)
    if not counts:
        return []
    rows = [run_once(peer_urls[:n], model, sample_repo) for n in counts]
    append_results(results_path, [format_row(r) for r in rows])
    return rows
