"""Level 2, phase 3 - the live session dashboard ("TUI panes").

A visual layer over the orchestrator, built only on the standard
library so swarm-agent stays dependency-free. Each session gets its
own **pane**: a boxed card showing its status, kind, pinned peer,
branch and a live preview of its log / result - the terminal
equivalent of one OpenCode pane per task.

Two entry points:

- :func:`render_dashboard` - a pure function: sessions (+ optional
  log tails) in, one frame of text out. Deterministic and testable.
- :func:`watch` - the live loop: re-reads the session store on an
  interval, redraws the frame, and stops after a bounded number of
  iterations when asked (``--once`` in the CLI is exactly 1).

The dashboard only *reads* orchestrator state. Starting, stopping
and cleaning sessions stay with the ``sessions`` CLI commands, so
several terminals can watch while others drive.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, TextIO

from .orchestrator import Session, SessionStore

STATUS_MARKS = {
    "queued": "[ ]",
    "running": "[*]",
    "done": "[x]",
    "failed": "[!]",
    "stopped": "[-]",
}

DEFAULT_WIDTH = 80
MIN_WIDTH = 40
MAX_WIDTH = 200


def _clamp_width(width: int) -> int:
    return max(MIN_WIDTH, min(MAX_WIDTH, int(width)))


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def session_elapsed(session: Session, now: Optional[datetime] = None) -> str:
    """Human duration for a session: finished-started, or - if unstarted."""
    start = _parse_ts(session.started_at)
    if start is None:
        return "-"
    end = _parse_ts(session.finished_at) or now or datetime.now(timezone.utc)
    seconds = max(0.0, (end - start).total_seconds())
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m"


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word-wrap to width, hard-clipping overlong words."""
    if width <= 0:
        return [""]
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current or not lines:
        lines.append(current)
    return lines


def pane_preview(session: Session, log_tail: Optional[list[str]] = None) -> list[str]:
    """The text a pane shows: log tail wins, then error, then result,
    then the task payload, then a plain status line."""
    if log_tail:
        cleaned = [line.rstrip() for line in log_tail if line.strip()]
        if cleaned:
            return cleaned
    if session.error:
        return [f"error: {session.error}"]
    if session.result:
        return session.result.splitlines()
    payload = session.payload or {}
    for key in ("prompt", "question", "target", "error"):
        if payload.get(key):
            return [f"{key}: {payload[key]}"]
    return [f"status: {session.status}"]


def render_pane(
    session: Session,
    width: int = DEFAULT_WIDTH,
    preview_lines: int = 4,
    log_tail: Optional[list[str]] = None,
    now: Optional[datetime] = None,
) -> str:
    """Render one session as a boxed pane of exactly ``width`` columns."""
    width = _clamp_width(width)
    inner = width - 4  # "| " + " |"
    mark = STATUS_MARKS.get(session.status, "[?]")
    border = "+" + "-" * (width - 2) + "+"

    def row(text: str) -> str:
        return "| " + _clip(text, inner).ljust(inner) + " |"

    title = (
        f"{mark} {session.id}  {session.status}  {session.kind}  "
        f"peer: {session.peer or '(swarm)'}  {session_elapsed(session, now=now)}"
    )
    lines = [border, row(title), row(f"branch: {session.branch or '-'}")]
    preview = pane_preview(session, log_tail=log_tail)
    wrapped: list[str] = []
    for raw in preview:
        wrapped.extend(_wrap(raw, inner))
    for line in wrapped[: max(0, preview_lines)]:
        lines.append(row(line))
    if len(wrapped) > preview_lines:
        lines.append(row(f"... ({len(wrapped) - preview_lines} more lines)"))
    lines.append(border)
    return "\n".join(lines)


def render_dashboard(
    sessions: Iterable[Session],
    width: int = DEFAULT_WIDTH,
    preview_lines: int = 4,
    log_tails: Optional[dict[str, list[str]]] = None,
    now: Optional[datetime] = None,
) -> str:
    """Render the full dashboard frame: header counts + one pane each."""
    width = _clamp_width(width)
    sessions = list(sessions)
    log_tails = log_tails or {}
    counts: dict[str, int] = {}
    for session in sessions:
        counts[session.status] = counts.get(session.status, 0) + 1
    summary = "  ".join(
        f"{status}: {counts[status]}" for status in sorted(counts)
    ) or "no sessions"
    header = _clip(f"swarm-agent sessions - {summary}", width)
    frame = [header, "=" * len(header)]
    if not sessions:
        frame.append("(no sessions yet - spawn one with: swarm-agent sessions spawn ...)")
        return "\n".join(frame)
    for session in sessions:
        frame.append("")
        frame.append(
            render_pane(
                session,
                width=width,
                preview_lines=preview_lines,
                log_tail=log_tails.get(session.id),
                now=now,
            )
        )
    return "\n".join(frame)


def collect_log_tails(store: SessionStore, tail: int = 6) -> dict[str, list[str]]:
    """Read the last ``tail`` lines of each session's log, if it exists."""
    tails: dict[str, list[str]] = {}
    for session in store.all():
        path = store.log_dir / f"{session.id}.log"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if lines:
            tails[session.id] = lines[-tail:] if tail > 0 else lines
    return tails


def watch(
    repo: str | Path,
    interval: float = 1.0,
    iterations: Optional[int] = None,
    width: int = DEFAULT_WIDTH,
    preview_lines: int = 4,
    out: Optional[TextIO] = None,
    clear: bool = True,
) -> int:
    """Live loop: redraw the dashboard from the store until done.

    With ``iterations`` set the loop is bounded (tests and ``--once``);
    otherwise it runs until every session is terminal or the user
    interrupts with Ctrl-C. Returns 0.
    """
    out = out if out is not None else sys.stdout
    store = SessionStore(repo)
    done_count = 0
    while True:
        sessions = store.all()
        frame = render_dashboard(
            sessions,
            width=width,
            preview_lines=preview_lines,
            log_tails=collect_log_tails(store),
        )
        if clear and iterations is None:
            out.write("\x1b[2J\x1b[H")
        out.write(frame + "\n")
        out.flush()
        done_count += 1
        if iterations is not None and done_count >= iterations:
            return 0
        if sessions and all(
            s.status in ("done", "failed", "stopped") for s in sessions
        ) and iterations is None:
            return 0
        try:
            time.sleep(max(0.05, interval))
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            return 0
