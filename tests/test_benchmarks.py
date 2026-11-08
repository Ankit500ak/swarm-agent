"""Tests for the benchmark harness (Issue #4).

All tests are pure: no live Ollama peer is ever contacted. The live
runs happen on real hardware; what is pinned here is the planning,
the metrics maths, the results table, the distributor capture, and
the pinned sample workload itself.
"""

from pathlib import Path

import pytest

from benchmarks import harness
from swarm_agent import agent
from swarm_agent.chunking import chunk_file
from swarm_agent.config import Config
from swarm_agent.discovery import Peer
from swarm_agent.ollama import SwarmError

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_REPO = REPO_ROOT / "benchmarks" / "sample_repo"


# -- planning -------------------------------------------------------------


def test_plan_peer_counts():
    assert harness.plan_peer_counts([]) == []
    assert harness.plan_peer_counts(["a"]) == [1]
    assert harness.plan_peer_counts(["a", "b"]) == [1, 2]
    assert harness.plan_peer_counts(["a", "b", "c"]) == [1, 2, 3]
    assert harness.plan_peer_counts(["a", "b", "c", "d"]) == [1, 2, 3]
    assert harness.plan_peer_counts(["a", "b", "c"], max_peers=2) == [1, 2]


# -- metrics ---------------------------------------------------------------


def test_chunks_per_minute():
    assert harness.chunks_per_minute(60, 30.0) == 120.0
    assert harness.chunks_per_minute(16, 60.0) == 16.0
    assert harness.chunks_per_minute(0, 10.0) == 0.0
    assert harness.chunks_per_minute(10, 0.0) == 0.0


def test_machine_label_is_nonempty_and_has_memory():
    label = harness.machine_label()
    assert label
    assert "GB" in label


# -- results table ---------------------------------------------------------


def _ok_result():
    return {
        "date": "2026-11-08T15:00:00",
        "machine": "Apple M5, 16.0 GB",
        "model": "qwen2.5-coder:7b",
        "ollama_peers": 2,
        "wall_s": 30.0,
        "chunks": 16,
        "chunks_per_min": 32.0,
        "per_peer": "localhost:11434:10, 192.168.1.5:11434:6",
        "status": "ok",
        "error": "",
    }


def test_format_row_ok():
    assert harness.format_row(_ok_result()) == (
        "| 2026-11-08T15:00:00 | Apple M5, 16.0 GB | qwen2.5-coder:7b | 2 | "
        "30.0 | 16 | 32.0 | localhost:11434:10, 192.168.1.5:11434:6 |"
    )


def test_format_row_failed_shows_error_not_numbers():
    result = _ok_result()
    result.update(status="failed", wall_s=1.5, chunks=None, chunks_per_min=None,
                  per_peer="", error="no Ollama peer found")
    assert harness.format_row(result) == (
        "| 2026-11-08T15:00:00 | Apple M5, 16.0 GB | qwen2.5-coder:7b | 2 | "
        "— | — | — | failed: no Ollama peer found |"
    )


def test_append_results_creates_header_once(tmp_path):
    path = tmp_path / "RESULTS.md"
    harness.append_results(path, [harness.format_row(_ok_result())])
    harness.append_results(path, [harness.format_row(_ok_result())])
    text = path.read_text(encoding="utf-8")
    assert text.count("# Benchmark results") == 1
    assert text.count("| 2026-11-08T15:00:00 |") == 2


# -- distributor capture ---------------------------------------------------


class _FakeDistributor:
    def __init__(self, peers, counts):
        self.peers = peers
        self.counts = counts


def _peer(url):
    return Peer(base_url=url, models=["qwen2.5-coder:7b"], latency_ms=5.0, kind="ollama")


def test_capture_distributor_records_instance_and_restores(monkeypatch):
    fake = _FakeDistributor([_peer("http://localhost:11434")],
                            {"http://localhost:11434": 16})
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: fake)
    original = agent.get_distributor
    with harness.capture_distributor() as captured:
        agent.get_distributor(Config())
    assert captured["distributor"] is fake
    # The wrapper is removed again afterwards.
    assert agent.get_distributor is original


# -- run_once / run_benchmark with stubbed flow ----------------------------


def test_run_once_reads_metrics_from_captured_distributor(monkeypatch):
    fake = _FakeDistributor(
        [_peer("http://localhost:11434"), _peer("http://192.168.1.5:11434")],
        {"http://localhost:11434": 10, "http://192.168.1.5:11434": 6},
    )
    monkeypatch.setattr(agent, "get_distributor", lambda cfg: fake)

    def fake_analyze(path, config):
        agent.get_distributor(config)  # the real flow builds the distributor
        return "report"

    monkeypatch.setattr(agent, "analyze_flow", fake_analyze)
    result = harness.run_once(
        ["http://localhost:11434", "http://192.168.1.5:11434"],
        "qwen2.5-coder:7b",
        SAMPLE_REPO,
    )
    assert result["status"] == "ok"
    assert result["ollama_peers"] == 2
    assert result["chunks"] == 16
    assert result["per_peer"] == "localhost:11434:10, 192.168.1.5:11434:6"
    assert result["wall_s"] >= 0
    assert result["chunks_per_min"] >= 0


def test_run_once_failure_is_recorded_not_raised(monkeypatch):
    def failing_analyze(path, config):
        raise SwarmError("no Ollama peer found")

    monkeypatch.setattr(agent, "analyze_flow", failing_analyze)
    result = harness.run_once(["http://localhost:11434"], "qwen2.5-coder:7b", SAMPLE_REPO)
    assert result["status"] == "failed"
    assert result["error"] == "no Ollama peer found"
    assert result["chunks"] is None


def test_run_benchmark_no_peers_runs_nothing(tmp_path):
    results = tmp_path / "RESULTS.md"
    assert harness.run_benchmark([], "qwen2.5-coder:7b", SAMPLE_REPO, results) == []
    assert not results.exists()


# -- pinned sample workload --------------------------------------------------


def test_sample_repo_is_pinned():
    files = sorted(SAMPLE_REPO.glob("*.py"))
    assert [f.name for f in files] == ["calc.py", "cli.py", "store.py", "test_calc.py"]
    total = sum(len(chunk_file(f, budget_tokens=1500)) for f in files)
    assert total == 16
