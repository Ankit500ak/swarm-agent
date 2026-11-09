"""Level 2 - the multi-session orchestrator.

Run several agent tasks at once, each as its own session, each
isolated in its own **git worktree**, and each optionally **pinned to
one peer** on the LAN - so one machine crunches the refactor while
another writes tests, and no two sessions ever fight over the same
files.

A session wraps one of the existing flows (``ask`` / ``analyze`` /
``gen-tests`` / ``fix``) with:

- a worktree on its own branch (``swarm/<session-id>``) created next
  to the repository as ``<repo>.swarm-worktrees/<session-id>``;
- a persistent record in ``<repo>/.swarm/sessions.json`` so sessions
  survive across CLI invocations;
- a per-session :class:`~swarm_agent.config.Config`: when a peer is
  pinned, discovery is restricted to that peer only;
- its own log file at ``<repo>/.swarm/logs/<session-id>.log`` - the
  session's "terminal scrollback".

Phase 2 execution model: every session flow runs in a **child
process** (fork where available). The session record carries the
child's PID, so ``stop`` works even from a different terminal /
invocation than the one that started the session: it sends SIGTERM to
the flow process and marks the session stopped, and the owning
:meth:`Orchestrator.run` respects that mark instead of overwriting it
with a failure when its child dies.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Callable, Optional

from . import agent, prompts
from .config import Config
from .ollama import SwarmError

SESSION_KINDS = ("ask", "analyze", "gen-tests", "fix")
TERMINAL_STATUSES = ("done", "failed", "stopped")
MAX_RESULT_CHARS = 20_000

# A flow runs one session against a (possibly peer-pinned) config and
# returns a human-readable result string.
Flow = Callable[["Session", Config], str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + "\n... [truncated]"


def _mp_context():
    """Fork where available (children inherit flows, no pickling)."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix fallback
        return multiprocessing.get_context()


def _session_child_main(
    session: "Session", config: Config, flow: Flow, queue
) -> None:
    """Child-process entry point: run the flow, report via the queue."""
    try:
        result = flow(session, config)
        queue.put({"status": "done", "result": result or ""})
    except BaseException as exc:  # the parent turns this into data
        queue.put({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Sessions and their persistent store
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """One orchestrated task: a flow, a worktree, optionally a peer."""

    id: str
    kind: str
    payload: dict = field(default_factory=dict)
    repo: str = ""
    worktree: str = ""
    branch: str = ""
    peer: Optional[str] = None
    status: str = "queued"
    result: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    cleaned: bool = False
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class SessionStore:
    """JSON-backed session records at ``<repo>/.swarm/sessions.json``."""

    def __init__(self, repo: str | Path) -> None:
        self.repo = Path(repo)
        self.path = self.repo / ".swarm" / "sessions.json"
        self._lock = threading.Lock()

    @property
    def log_dir(self) -> Path:
        return self.path.parent / "logs"

    def _load(self) -> dict[str, Session]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        sessions: dict[str, Session] = {}
        for sid, raw in data.items():
            if isinstance(raw, dict):
                sessions[sid] = Session.from_dict(raw)
        return sessions

    def _save(self, sessions: dict[str, Session]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {sid: s.to_dict() for sid, s in sessions.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def upsert(self, session: Session) -> None:
        with self._lock:
            sessions = self._load()
            sessions[session.id] = session
            self._save(sessions)

    def get(self, session_id: str) -> Session:
        session = self._load().get(session_id)
        if session is None:
            raise SwarmError(f"no such session: {session_id}")
        return session

    def all(self) -> list[Session]:
        return sorted(self._load().values(), key=lambda s: s.created_at)


# ---------------------------------------------------------------------------
# Git worktrees
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise SwarmError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


class WorktreeManager:
    """One worktree per session, on branch ``swarm/<session-id>``."""

    def __init__(self, repo: str | Path) -> None:
        self.repo = Path(repo).resolve()
        try:
            inside = _git(self.repo, "rev-parse", "--is-inside-work-tree")
        except SwarmError:
            inside = ""
        if inside != "true":
            raise SwarmError(f"not a git repository: {self.repo}")

    @property
    def root(self) -> Path:
        """Worktrees live *next to* the repo, never inside it."""
        return self.repo.parent / f"{self.repo.name}.swarm-worktrees"

    def create(self, session_id: str) -> tuple[Path, str]:
        branch = f"swarm/{session_id}"
        path = self.root / session_id
        self.root.mkdir(parents=True, exist_ok=True)
        _git(self.repo, "worktree", "add", "-b", branch, str(path), "HEAD")
        return path, branch

    def remove(self, session: Session, delete_branch: bool = False) -> None:
        if session.worktree and Path(session.worktree).exists():
            _git(
                self.repo,
                "worktree",
                "remove",
                "--force",
                session.worktree,
            )
        if delete_branch and session.branch:
            _git(self.repo, "branch", "-D", session.branch)


# ---------------------------------------------------------------------------
# The default flows - thin adapters over swarm_agent.agent
# ---------------------------------------------------------------------------


def _flow_ask(session: Session, config: Config) -> str:
    distributor = agent.get_distributor(config)
    return distributor.ask([
        {"role": "system", "content": prompts.ASK_SYSTEM},
        {"role": "user", "content": session.payload["prompt"]},
    ])


def _flow_analyze(session: Session, config: Config) -> str:
    return agent.analyze_flow(
        session.worktree, config, question=session.payload.get("question")
    )


def _flow_gen_tests(session: Session, config: Config) -> str:
    target = Path(session.worktree) / session.payload["target"]
    return agent.gen_tests_flow(target, config, write=True, run=False)


def _flow_fix(session: Session, config: Config) -> str:
    rc = agent.fix_flow(
        session.payload["error"],
        repo_root=session.worktree,
        config=config,
        apply=bool(session.payload.get("apply", False)),
        test_cmd=session.payload.get("test_cmd"),
    )
    if rc != 0:
        raise SwarmError(f"fix flow failed with exit code {rc}")
    return "fix flow completed"


DEFAULT_FLOWS: dict[str, Flow] = {
    "ask": _flow_ask,
    "analyze": _flow_analyze,
    "gen-tests": _flow_gen_tests,
    "fix": _flow_fix,
}

_REQUIRED_PAYLOAD = {
    "ask": ("prompt",),
    "analyze": (),
    "gen-tests": ("target",),
    "fix": ("error",),
}


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Spawn, run, stop and clean up sessions for one repository."""

    def __init__(
        self,
        repo: str | Path,
        config: Config,
        flows: Optional[dict[str, Flow]] = None,
        in_process: bool = False,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.config = config
        self.flows = flows if flows is not None else DEFAULT_FLOWS
        # in_process exists for tests and embeddings that need to
        # observe a flow's side effects directly; the CLI always runs
        # flows in child processes so they can be stopped mid-run.
        self.in_process = in_process
        self.store = SessionStore(self.repo)
        self.worktrees = WorktreeManager(self.repo)

    # -- lifecycle -----------------------------------------------------------

    def spawn(
        self,
        kind: str,
        payload: Optional[dict] = None,
        peer: Optional[str] = None,
    ) -> Session:
        if kind not in self.flows:
            raise SwarmError(
                f"unknown session kind: {kind!r} "
                f"(choose from {', '.join(sorted(self.flows))})"
            )
        payload = dict(payload or {})
        for key in _REQUIRED_PAYLOAD.get(kind, ()):
            if not payload.get(key):
                raise SwarmError(f"session kind {kind!r} needs payload[{key!r}]")
        session_id = _new_id()
        worktree, branch = self.worktrees.create(session_id)
        session = Session(
            id=session_id,
            kind=kind,
            payload=payload,
            repo=str(self.repo),
            worktree=str(worktree),
            branch=branch,
            peer=peer,
            created_at=_now(),
        )
        self.store.upsert(session)
        return session

    def run(self, session_id: str) -> Session:
        session = self.store.get(session_id)
        if session.status != "queued":
            raise SwarmError(
                f"session {session_id} is {session.status}, not queued"
            )
        session.status = "running"
        session.started_at = _now()
        self.store.upsert(session)

        if self.in_process:
            try:
                result = self.flows[session.kind](
                    session, self._config_for(session)
                )
                session.result = _truncate(result or "")
                session.status = "done"
            except Exception as exc:  # a flow failure is data, not a crash
                session.error = str(exc)
                session.status = "failed"
            session.finished_at = _now()
            self.store.upsert(session)
            self._write_log(session)
            return session

        ctx = _mp_context()
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_session_child_main,
            args=(
                session,
                self._config_for(session),
                self.flows[session.kind],
                queue,
            ),
        )
        proc.start()
        session.pid = proc.pid
        self.store.upsert(session)

        outcome: Optional[dict] = None
        while proc.is_alive():
            try:
                outcome = queue.get(timeout=0.05)
                break
            except Empty:
                pass
        proc.join(timeout=5)
        if outcome is None:
            try:
                outcome = queue.get_nowait()
            except Empty:
                outcome = None
        session.pid = None

        if outcome is not None:
            if outcome["status"] == "done":
                session.result = _truncate(outcome.get("result") or "")
                session.status = "done"
            else:
                session.error = outcome.get("error") or "flow failed"
                session.status = "failed"
        else:
            # No result from the child: either it was stopped from
            # another invocation (respect that mark) or it died.
            fresh = self.store.get(session_id)
            if fresh.status == "stopped":
                session.status = "stopped"
                session.error = fresh.error
            else:
                session.error = (
                    "flow process exited without a result "
                    f"(exit code {proc.exitcode})"
                )
                session.status = "failed"
        session.finished_at = _now()
        self.store.upsert(session)
        self._write_log(session)
        return session

    def run_all(self, max_workers: int = 4) -> list[Session]:
        queued = [s for s in self.store.all() if s.status == "queued"]
        if not queued:
            return []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            return list(pool.map(lambda s: self.run(s.id), queued))

    def stop(self, session_id: str) -> Session:
        session = self.store.get(session_id)
        if session.status in TERMINAL_STATUSES:
            raise SwarmError(
                f"session {session_id} is already {session.status}"
            )
        if session.status == "running":
            if not session.pid:
                raise SwarmError(
                    f"session {session_id} is running without a recorded "
                    "pid; it cannot be stopped from here"
                )
            try:
                os.kill(session.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone - still mark it stopped
            except PermissionError as exc:
                raise SwarmError(
                    f"cannot signal session {session_id} (pid "
                    f"{session.pid}): {exc}"
                ) from exc
        session.status = "stopped"
        session.pid = None
        session.finished_at = _now()
        self.store.upsert(session)
        return session

    def clean(self, session_id: str, delete_branch: bool = False) -> Session:
        session = self.store.get(session_id)
        if session.status not in TERMINAL_STATUSES:
            raise SwarmError(
                f"session {session_id} is {session.status}; "
                "only finished sessions can be cleaned"
            )
        self.worktrees.remove(session, delete_branch=delete_branch)
        session.cleaned = True
        self.store.upsert(session)
        return session

    # -- queries -------------------------------------------------------------

    def get(self, session_id: str) -> Session:
        return self.store.get(session_id)

    def list(self) -> list[Session]:
        return self.store.all()

    def log_path(self, session: Session) -> Path:
        return self.store.log_dir / f"{session.id}.log"

    # -- internals -----------------------------------------------------------

    def _config_for(self, session: Session) -> Config:
        if not session.peer:
            return self.config
        # Pin the session to one peer: no discovery, no substitutes.
        return replace(
            self.config, peers=[session.peer], auto_discover=False
        )

    def _write_log(self, session: Session) -> None:
        try:
            self.store.log_dir.mkdir(parents=True, exist_ok=True)
            lines = [
                f"session  : {session.id}",
                f"kind     : {session.kind}",
                f"peer     : {session.peer or '(swarm)'}",
                f"branch   : {session.branch}",
                f"worktree : {session.worktree}",
                f"created  : {session.created_at}",
                f"started  : {session.started_at}",
                f"finished : {session.finished_at}",
                f"status   : {session.status}",
                "",
            ]
            if session.result:
                lines += ["--- result ---", session.result, ""]
            if session.error:
                lines += ["--- error ---", session.error, ""]
            self.log_path(session).write_text(
                "\n".join(lines), encoding="utf-8"
            )
        except OSError:
            pass  # a log must never fail the session itself
