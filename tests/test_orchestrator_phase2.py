"""Tests for the Level 2 orchestrator, phase 2: process-isolated runs,
mid-run stop, and per-session logs.

Flows are module-level functions so they stay valid under any
multiprocessing start method; the orchestrator itself prefers fork.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from swarm_agent.cli import main
from swarm_agent.config import Config
from swarm_agent.ollama import SwarmError
from swarm_agent.orchestrator import Orchestrator, Session


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "proj"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "init")
    return path


# -- flows (module-level on purpose) ------------------------------------------


def _fast(session: Session, config: Config) -> str:
    return "pong"


def _slow(session: Session, config: Config) -> str:
    time.sleep(30)
    return "never reached"


def _boom(session: Session, config: Config) -> str:
    raise RuntimeError("child exploded")


def _flows(flow) -> dict:
    return {kind: flow for kind in ("ask", "analyze", "gen-tests", "fix")}


# -- process execution ----------------------------------------------------------


def test_run_executes_in_child_process_and_clears_pid(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_fast))
    session = orch.spawn("ask", {"prompt": "ping"})
    finished = orch.run(session.id)
    assert finished.status == "done"
    assert finished.result == "pong"
    assert finished.pid is None  # cleared once finished


def test_child_exception_becomes_failed_session(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_boom))
    session = orch.spawn("ask", {"prompt": "ping"})
    finished = orch.run(session.id)
    assert finished.status == "failed"
    assert "child exploded" in (finished.error or "")


def test_run_writes_session_log(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_fast))
    session = orch.spawn("ask", {"prompt": "ping"})
    finished = orch.run(session.id)
    log = orch.log_path(finished)
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "pong" in text
    assert session.id in text


# -- mid-run stop -----------------------------------------------------------------


def test_stop_terminates_running_session_from_another_instance(
    repo: Path,
) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_slow))
    session = orch.spawn("ask", {"prompt": "ping"})

    outcome: dict = {}

    def _run() -> None:
        outcome["session"] = orch.run(session.id)

    thread = threading.Thread(target=_run, daemon=True)
    started = time.monotonic()
    thread.start()

    # Wait until the run is visibly running with a live pid recorded.
    deadline = started + 10
    while time.monotonic() < deadline:
        current = orch.get(session.id)
        if current.status == "running" and current.pid:
            break
        time.sleep(0.05)
    else:
        pytest.fail("session never reached running state with a pid")

    # A *different* orchestrator instance issues the stop, like a
    # second terminal would.
    stopper = Orchestrator(repo, Config(), flows=_flows(_slow))
    stopped = stopper.stop(session.id)
    assert stopped.status == "stopped"

    thread.join(timeout=10)
    assert not thread.is_alive(), "run() did not return after the stop"
    assert time.monotonic() - started < 20  # did not sleep the full 30 s
    final = orch.get(session.id)
    assert final.status == "stopped"  # the owner respected the stop mark
    assert final.pid is None


def test_stop_with_stale_pid_still_marks_stopped(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_fast))
    session = orch.spawn("ask", {"prompt": "ping"})
    # Simulate a running session whose process is already gone.
    record = orch.get(session.id)
    record.status = "running"
    record.pid = 999_999
    orch.store.upsert(record)
    stopped = orch.stop(session.id)
    assert stopped.status == "stopped"


def test_stop_running_without_pid_raises(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_fast))
    session = orch.spawn("ask", {"prompt": "ping"})
    record = orch.get(session.id)
    record.status = "running"
    record.pid = None
    orch.store.upsert(record)
    with pytest.raises(SwarmError):
        orch.stop(session.id)


# -- CLI ---------------------------------------------------------------------------


def test_cli_sessions_show(repo: Path, capsys: pytest.CaptureFixture) -> None:
    orch = Orchestrator(repo, Config(), flows=_flows(_fast))
    session = orch.spawn("ask", {"prompt": "ping"})
    orch.run(session.id)
    rc = main(["sessions", "show", session.id, "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert session.id in out
    assert "done" in out
    assert "pong" in out
