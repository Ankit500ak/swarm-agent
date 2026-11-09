"""Tests for the Level 2 multi-session orchestrator (phase 1).

Everything runs against real git worktrees in a tmp repo, with fake
flows injected - no peers, no network.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

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


def _fake_flows(record: list, reply: str = "ok") -> dict:
    def flow(session: Session, config: Config) -> str:
        record.append((session.id, session.kind, session.worktree, config))
        return reply

    return {kind: flow for kind in ("ask", "analyze", "gen-tests", "fix")}


def _orchestrator(repo: Path, record: list, reply: str = "ok") -> Orchestrator:
    # Phase 1 semantics are about dispatch, so these run flows
    # in-process where the record list can observe them; process
    # isolation itself is covered by test_orchestrator_phase2.py.
    return Orchestrator(
        repo, Config(), flows=_fake_flows(record, reply), in_process=True
    )


# -- spawn / worktrees -------------------------------------------------------


def test_spawn_creates_worktree_branch_and_record(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hello"})
    assert session.status == "queued"
    worktree = Path(session.worktree)
    assert worktree.is_dir()
    assert (worktree / "mod.py").is_file()
    assert session.branch == f"swarm/{session.id}"
    assert session.branch in _git(repo, "branch", "--list")
    # The record persists across instances.
    assert Orchestrator(repo, Config()).get(session.id).kind == "ask"


def test_spawn_rejects_unknown_kind_and_missing_payload(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    with pytest.raises(SwarmError):
        orch.spawn("dance", {})
    with pytest.raises(SwarmError):
        orch.spawn("ask", {})
    with pytest.raises(SwarmError):
        orch.spawn("gen-tests", {})


def test_spawn_requires_a_git_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SwarmError):
        Orchestrator(plain, Config(), flows=_fake_flows([]))


# -- run ---------------------------------------------------------------------


def test_run_dispatches_flow_in_worktree_and_stores_result(repo: Path) -> None:
    record: list = []
    orch = _orchestrator(repo, record, reply="the answer")
    session = orch.spawn("analyze", {"question": "what does mod.py do?"})
    finished = orch.run(session.id)
    assert finished.status == "done"
    assert finished.result == "the answer"
    assert finished.started_at and finished.finished_at
    sid, kind, worktree, _cfg = record[0]
    assert sid == session.id and kind == "analyze"
    assert worktree == session.worktree


def test_run_pinned_peer_restricts_config(repo: Path) -> None:
    record: list = []
    orch = _orchestrator(repo, record)
    session = orch.spawn("ask", {"prompt": "hi"}, peer="mac-studio:11434")
    orch.run(session.id)
    cfg = record[0][3]
    assert cfg.peers == ["http://mac-studio:11434"]
    assert cfg.auto_discover is False


def test_run_unpinned_session_keeps_config(repo: Path) -> None:
    record: list = []
    orch = _orchestrator(repo, record)
    session = orch.spawn("ask", {"prompt": "hi"})
    orch.run(session.id)
    assert record[0][3] is orch.config


def test_failed_flow_marks_session_failed(repo: Path) -> None:
    def boom(session: Session, config: Config) -> str:
        raise RuntimeError("peer exploded")

    flows = {kind: boom for kind in ("ask", "analyze", "gen-tests", "fix")}
    orch = Orchestrator(repo, Config(), flows=flows, in_process=True)
    session = orch.spawn("ask", {"prompt": "hi"})
    finished = orch.run(session.id)
    assert finished.status == "failed"
    assert "peer exploded" in (finished.error or "")


def test_run_refuses_non_queued_session(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hi"})
    orch.run(session.id)
    with pytest.raises(SwarmError):
        orch.run(session.id)


def test_run_all_executes_sessions_concurrently(repo: Path) -> None:
    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    def slow(session: Session, config: Config) -> str:
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.3)
        with lock:
            state["current"] -= 1
        return "done"

    flows = {kind: slow for kind in ("ask", "analyze", "gen-tests", "fix")}
    orch = Orchestrator(repo, Config(), flows=flows, in_process=True)
    ids = [orch.spawn("ask", {"prompt": f"q{i}"}).id for i in range(3)]
    finished = orch.run_all(max_workers=3)
    assert [s.status for s in finished] == ["done"] * 3
    assert {s.id for s in finished} == set(ids)
    assert state["peak"] >= 2  # genuinely parallel, not serial


# -- stop / clean --------------------------------------------------------------


def test_stop_queued_session(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hi"})
    stopped = orch.stop(session.id)
    assert stopped.status == "stopped"
    assert orch.run_all() == []  # nothing left to run


def test_stop_running_or_finished_session_raises(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hi"})
    orch.run(session.id)
    with pytest.raises(SwarmError):
        orch.stop(session.id)


def test_clean_removes_worktree_keeps_branch_by_default(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hi"})
    orch.run(session.id)
    cleaned = orch.clean(session.id)
    assert cleaned.cleaned is True
    assert not Path(session.worktree).exists()
    assert session.branch in _git(repo, "branch", "--list")


def test_clean_can_delete_branch_and_refuses_active_session(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    session = orch.spawn("ask", {"prompt": "hi"})
    with pytest.raises(SwarmError):
        orch.clean(session.id)  # still queued
    orch.run(session.id)
    orch.clean(session.id, delete_branch=True)
    assert session.branch not in _git(repo, "branch", "--list")


def test_unknown_session_id_raises(repo: Path) -> None:
    orch = _orchestrator(repo, [])
    with pytest.raises(SwarmError):
        orch.get("deadbeef0000")
