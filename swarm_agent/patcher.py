"""Applying unified diffs - carefully.

The engine's one hard rule: **verify before you write**. Every hunk's context
and removed lines must match the current file content before anything is
written. All files in a diff are first computed in memory; only when every
hunk of every file verifies are the results written out (atomically, per
file). If anything does not match, :class:`PatchError` is raised and the
working tree is left exactly as it was.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class PatchError(Exception):
    """The diff is malformed, or its context does not match the files."""


@dataclass
class FileChange:
    """One file's outcome inside a (possibly multi-file) patch."""

    path: str  # relative to the repo root, as named in the diff
    hunks_applied: int  # hunks verified (and, unless dry-run, written)
    action: str  # "modified" | "created" | "deleted"


@dataclass
class MultiFileResult:
    """Structured outcome of :func:`apply_multi_file_diff`.

    ``files`` has one :class:`FileChange` per file section in the diff, in
    diff order. On any verification failure no result is returned at all -
    :class:`PatchError` is raised and nothing is written.
    """

    files: list[FileChange] = field(default_factory=list)
    dry_run: bool = False

    @property
    def changed_paths(self) -> list[str]:
        return [f.path for f in self.files]


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def extract_diff(text: str) -> Optional[str]:
    """Pull the first unified diff out of a model response.

    Accepts a ```diff fenced block, or a bare diff starting at a ``--- ``
    header. Returns ``None`` when the response contains no diff.
    """
    if not text:
        return None
    fence = re.search(r"```(?:diff|patch)\s*\n(.*?)```", text, re.DOTALL)
    if fence and "---" in fence.group(1) and "@@" in fence.group(1):
        return fence.group(1).strip("\n")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            return "\n".join(lines[i:]).strip("\n")
    return None


class _Hunk:
    def __init__(self, old_start: int, old_count: int, new_start: int, new_count: int):
        self.old_start = old_start
        self.old_count = old_count
        self.new_start = new_start
        self.new_count = new_count
        self.lines: list[str] = []  # raw hunk body lines, prefix char included


class _FilePatch:
    def __init__(self, old_path: Optional[str], new_path: Optional[str]):
        self.old_path = old_path  # None means /dev/null (creation)
        self.new_path = new_path  # None means /dev/null (deletion)
        self.hunks: list[_Hunk] = []


def _clean_path(raw: str) -> Optional[str]:
    raw = raw.strip().split("\t")[0].strip()
    if raw == "/dev/null":
        return None
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    return raw


def _parse(diff_text: str) -> list[_FilePatch]:
    patches: list[_FilePatch] = []
    current: Optional[_FilePatch] = None
    hunk: Optional[_Hunk] = None
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            current = None
            hunk = None
            i += 1
            continue
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise PatchError(f"malformed diff: '---' not followed by '+++': {line!r}")
            current = _FilePatch(_clean_path(line[4:]), _clean_path(lines[i + 1][4:]))
            patches.append(current)
            hunk = None
            i += 2
            continue
        m = _HUNK_RE.match(line)
        if m:
            if current is None:
                raise PatchError("hunk found before any file header")
            hunk = _Hunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2)) if m.group(2) is not None else 1,
                new_start=int(m.group(3)),
                new_count=int(m.group(4)) if m.group(4) is not None else 1,
            )
            current.hunks.append(hunk)
            i += 1
            continue
        if hunk is not None and line[:1] in (" ", "-", "+", "\\"):
            hunk.lines.append(line)
            i += 1
            continue
        # Anything else (index lines, mode lines, prose) is ignored.
        i += 1
    if not patches:
        raise PatchError("no file sections found in diff")
    for p in patches:
        if not p.hunks and p.old_path is not None and p.new_path is not None:
            raise PatchError(f"file section for {p.new_path} has no hunks")
    return patches


def _find(lines: list[str], pattern: list[str], expected: int) -> int:
    """Locate ``pattern`` in ``lines``: at ``expected`` if possible, else the
    unique occurrence, else -1."""
    def at(pos: int) -> bool:
        return 0 <= pos <= len(lines) - len(pattern) and lines[pos : pos + len(pattern)] == pattern

    if at(expected):
        return expected
    hits = [i for i in range(0, len(lines) - len(pattern) + 1) if at(i)]
    if len(hits) == 1:
        return hits[0]
    return -1


def _apply_to_lines(lines: list[str], patch: _FilePatch, name: str) -> list[str]:
    result = list(lines)
    offset = 0
    for hunk in patch.hunks:
        old_segment = [l[1:] for l in hunk.lines if l[:1] in (" ", "-")]
        new_segment = [l[1:] for l in hunk.lines if l[:1] in (" ", "+")]
        if len(old_segment) != hunk.old_count or len(new_segment) != hunk.new_count:
            raise PatchError(
                f"{name}: hunk @@ -{hunk.old_start},{hunk.old_count} "
                f"+{hunk.new_start},{hunk.new_count} @@ has "
                f"{len(old_segment)} old / {len(new_segment)} new lines"
            )
        expected = hunk.old_start - 1 + offset
        if not old_segment:  # pure insertion
            pos = min(max(hunk.old_start + offset, 0), len(result))
            # old_start for a pure insertion points *after* that line number
            pos = min(max(hunk.old_start - 1 + offset + (1 if hunk.old_count == 0 else 0), 0), len(result))
            result[pos:pos] = new_segment
            offset += len(new_segment)
            continue
        pos = _find(result, old_segment, expected)
        if pos < 0:
            raise PatchError(
                f"{name}: context does not match near line {hunk.old_start}; "
                "refusing to apply a stale diff"
            )
        result[pos : pos + len(old_segment)] = new_segment
        offset += len(new_segment) - len(old_segment)
    return result


def _resolve(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise PatchError(f"diff path escapes the repository: {rel}")
    return target


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# A planned change: (target, new content | None = delete, rel path,
# hunks in the file's section, action).
_Planned = tuple[Path, Optional[str], str, int, str]


def _plan_changes(root: Path, patches: list[_FilePatch]) -> list[_Planned]:
    """Phase 1 of every apply: verify *every* hunk of *every* file against
    the current contents and compute the new contents fully in memory.

    Nothing is written here. Any mismatch raises :class:`PatchError`, so a
    caller that plans first and writes second can never leave a
    half-applied diff behind because of a stale hunk.
    """
    planned: list[_Planned] = []
    for patch in patches:
        rel = patch.new_path if patch.new_path is not None else patch.old_path
        assert rel is not None
        hunks = len(patch.hunks)
        if patch.old_path is None:  # creation (--- /dev/null)
            target = _resolve(root, patch.new_path)  # type: ignore[arg-type]
            if target.exists() and target.read_text(encoding="utf-8", errors="replace").strip():
                raise PatchError(f"cannot create {rel}: file already exists")
            new_lines = _apply_to_lines([], patch, rel)
            content = "\n".join(new_lines) + ("\n" if new_lines else "")
            planned.append((target, content, rel, hunks, "created"))
        elif patch.new_path is None:  # deletion (+++ /dev/null)
            target = _resolve(root, patch.old_path)
            if not target.exists():
                raise PatchError(f"cannot delete {rel}: file does not exist")
            current = target.read_text(encoding="utf-8", errors="replace").splitlines()
            remaining = _apply_to_lines(current, patch, rel)
            if any(line.strip() for line in remaining):
                raise PatchError(f"deletion diff for {rel} leaves content behind")
            planned.append((target, None, rel, hunks, "deleted"))
        else:
            target = _resolve(root, patch.new_path)
            if not target.exists():
                raise PatchError(f"cannot patch {rel}: file does not exist")
            original = target.read_text(encoding="utf-8", errors="replace")
            new_lines = _apply_to_lines(original.splitlines(), patch, rel)
            content = "\n".join(new_lines)
            if new_lines:
                content += "\n"
            planned.append((target, content, rel, hunks, "modified"))
    return planned


def _write_plan(planned: list[_Planned]) -> None:
    """Phase 2: write a fully verified plan. Each file is replaced
    atomically; the plan as a whole is not transactional - a failure
    *while writing* (disk error, permissions) can still leave earlier
    files written. Verification failures can never reach this phase.
    """
    for target, content, _rel, _hunks, _action in planned:
        if content is None:
            target.unlink()
        else:
            _atomic_write(target, content)


def apply_unified_diff(
    repo_root: str | Path,
    diff_text: str,
    dry_run: bool = False,
) -> list[str]:
    """Apply ``diff_text`` under ``repo_root``.

    Verifies every hunk against the current files first; on any mismatch
    raises :class:`PatchError` and changes nothing. With ``dry_run=True``
    the verification runs but no file is written. Returns the list of
    changed paths (relative to the repo root, as named in the diff).
    """
    planned = _plan_changes(Path(repo_root), _parse(diff_text))
    if not dry_run:
        _write_plan(planned)
    return [rel for _t, _c, rel, _h, _a in planned]


def apply_multi_file_diff(
    repo_root: str | Path,
    diff_text: str,
    dry_run: bool = False,
) -> MultiFileResult:
    """Apply a (possibly multi-file) ``diff_text`` under ``repo_root``,
    all-or-nothing.

    Every file section is parsed, then *every* hunk of *every* file is
    verified against the current contents in memory. If any file fails
    verification, :class:`PatchError` is raised and **nothing is
    written** - not even the files that verified cleanly. Only when the
    whole diff verifies is each file written (atomically per file).

    New files (``--- /dev/null``) are created and files whose section
    ends in ``+++ /dev/null`` are deleted, provided the deletion's hunks
    consume the file's entire content.

    Returns a :class:`MultiFileResult` with one :class:`FileChange` per
    file section (``path``, ``hunks_applied``, ``action``). With
    ``dry_run=True`` the full verification runs but no file is written
    or deleted.
    """
    planned = _plan_changes(Path(repo_root), _parse(diff_text))
    if not dry_run:
        _write_plan(planned)
    return MultiFileResult(
        files=[
            FileChange(path=rel, hunks_applied=hunks, action=action)
            for _t, _c, rel, hunks, action in planned
        ],
        dry_run=dry_run,
    )
