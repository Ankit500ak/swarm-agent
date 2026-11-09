"""Tests for Level 2 phase 3: the live dashboard / TUI panes.

Rendering is pure (sessions in, frame out), so everything here runs
without a terminal, curses, or a running session.
"""

from __future__ import annotations

import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from swarm_agent import tui
from swarm_agent.cli import main
from swarm_agent.config import Config
from swarm_agent.orchestrator import Orchestrator, Session, SessionStore


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


def _session(**overrides) -> Session:
    base = dict(
        id="abc123def456",
        kind="ask",
        payload={"prompt": "explain the auth flow"},
        repo="/tmp/proj",
        worktree="/tmp/proj.swarm-worktrees/abc123def456",
        branch="swarm/abc123def456",
        peer="http://mac-studio:11434",
        status="running",
        created_at="2026-11-09T06:00:00+00:00",
        started_at="2026-11-09T06:00:10+00:00",
    )
    base.update(overrides)
    return Session(**base)


# -- pane rendering -------------------------------------------------------------


def test_pane_shows_identity_status_peer_and_branch() -> None:
    pane = tui.render_pane(_session(), width=80)
    assert "abc123def456" in pane
    assert "running" in pane
    assert "ask" in pane
    assert "mac-studio" in pane
    assert "swarm/abc123def456" in pane
    # boxed: every line is exactly the pane width
    assert all(len(line) == 80 for line in pane.splitlines())


def test_pane_preview_prefers_log_tail_then_error_then_result() -> None:
    with_log = _session(result="result text", error="boom")
    pane = tui.render_pane(with_log, log_tail=["log line one", "log line two"])
    assert "log line one" in pane and "result text" not in pane

    with_error = _session(status="failed", error="kaboom happened")
    assert "kaboom happened" in tui.render_pane(with_error)

    with_result = _session(status="done", result="all good here")
    assert "all good here" in tui.render_pane(with_result)

    queued = _session(status="queued", result=None, error=None)
    assert "explain the auth flow" in tui.render_pane(queued)


def test_pane_truncates_long_preview_with_more_marker() -> None:
    session = _session(result="\n".join(f"line {i}" for i in range(20)))
    pane = tui.render_pane(session, preview_lines=3)
    assert "more lines" in pane
    assert "line 19" not in pane


def test_session_elapsed_formats() -> None:
    now = datetime(2026, 11, 9, 6, 2, 10, tzinfo=timezone.utc)
    session = _session()  # started 06:00:10, no finish -> uses now
    assert tui.session_elapsed(session, now=now) == "2m00s"
    finished = _session(finished_at="2026-11-09T06:00:45+00:00")
    assert tui.session_elapsed(finished, now=now) == "35s"
    unstarted = _session(started_at=None)
    assert tui.session_elapsed(unstarted, now=now) == "-"


# -- dashboard ------------------------------------------------------------------


def test_dashboard_counts_statuses_and_stacks_panes() -> None:
    sessions = [
        _session(id="aaa111bbb222", status="done", result="ok"),
        _session(id="ccc333ddd444", status="running"),
        _session(id="eee555fff666", status="running"),
    ]
    frame = tui.render_dashboard(sessions, width=72)
    assert "done: 1" in frame
    assert "running: 2" in frame
    for sid in ("aaa111bbb222", "ccc333ddd444", "eee555fff666"):
        assert sid in frame


def test_dashboard_empty_state() -> None:
    frame = tui.render_dashboard([], width=72)
    assert "no sessions" in frame


# -- watch loop -------------------------------------------------------------------


def test_watch_once_prints_frame_from_store(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), flows={})
    store = SessionStore(repo)
    store.upsert(_session(id="watchme12345", repo=str(repo), status="done", result="seen"))
    out = io.StringIO()
    rc = tui.watch(repo, iterations=1, width=72, out=out, clear=False)
    assert rc == 0
    text = out.getvalue()
    assert "watchme12345" in text
    assert "seen" in text or "done" in text
    assert orch  # fixture sanity: a real repo backs the store


def test_collect_log_tails_reads_session_log(repo: Path) -> None:
    orch = Orchestrator(repo, Config(), in_process=True, flows={
        kind: (lambda s, c: "logged result")
        for kind in ("ask", "analyze", "gen-tests", "fix")
    })
    session = orch.spawn("ask", {"prompt": "ping"})
    finished = orch.run(session.id)
    tails = tui.collect_log_tails(orch.store, tail=50)
    assert finished.id in tails
    assert any("logged result" in line for line in tails[finished.id])


def test_cli_sessions_watch_once(repo: Path, capsys: pytest.CaptureFixture) -> None:
    store = SessionStore(repo)
    store.upsert(_session(id="cli987watch1", repo=str(repo), status="queued"))
    rc = main(["sessions", "watch", "--once", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cli987watch1" in out
    assert "queued" in out
