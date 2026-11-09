"""Level 2 - the multi-session orchestrator.

Phase 1: run several agent tasks at once, each in its own terminal
session, each isolated in its own **git worktree**, and each optionally
**pinned to one peer** on the LAN - so one machine crunches the
refactor while another writes tests, and no two sessions ever fight
over the same files.

A session wraps one of the existing flows (``ask`` / ``analyze`` /
``gen-tests`` / ``fix``) with:

- a worktree on its own branch (``swarm/<session-id>``) created next
  to the repository as ``<repo>.swarm-worktrees/<session-id>``;
- a persistent record in ``<repo>/.swarm/sessions.json`` so sessions
  survive across CLI invocations;
- a per-session :class:`~swarm_agent.config.Config`: when a peer is
  pinned, discovery is restricted to that peer only.

Sessions are executed by :meth:`Orchestrator.run` (one) or
:meth:`Orchestrator.run_all` (all queued, on a thread pool). Phase 1
stopping is cooperative: only a *queued* session can be stopped - a
running flow is never killed mid-flight.
"""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
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
    ) -> None:
        self.repo = Path(repo).resolve()
        self.config = config
        self.flows = flows if flows is not None else DEFAULT_FLOWS
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
        try:
            result = self.flows[session.kind](session, self._config_for(session))
            session.result = _truncate(result or "")
            session.status = "done"
        except Exception as exc:  # a flow failure is data, not a crash
            session.error = str(exc)
            session.status = "failed"
        session.finished_at = _now()
        self.store.upsert(session)
        return session

    def run_all(self, max_workers: int = 4) -> list[Session]:
        queued = [s for s in self.store.all() if s.status == "queued"]
        if not queued:
            return []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            return list(pool.map(lambda s: self.run(s.id), queued))

    def stop(self, session_id: str) -> Session:
        session = self.store.get(session_id)
        if session.status == "running":
            raise SwarmError(
                f"session {session_id} is running; phase 1 cannot stop a "
                "running flow - wait for it to finish"
            )
        if session.status in TERMINAL_STATUSES:
            raise SwarmError(
                f"session {session_id} is already {session.status}"
            )
        session.status = "stopped"
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

    # -- internals -----------------------------------------------------------

    def _config_for(self, session: Session) -> Config:
        if not session.peer:
            return self.config
        # Pin the session to one peer: no discovery, no substitutes.
        return replace(
            self.config, peers=[session.peer], auto_discover=False
        )
