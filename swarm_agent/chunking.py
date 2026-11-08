"""Code-aware chunking.

Models reason best about whole units of code, so Python files are split with
``ast`` into top-level function/class units (module preamble and inter-unit
code become ``module`` chunks). Everything else falls back to line windows
with a small overlap so context is not lost at the seams. Every chunk is
guaranteed to fit the configured token budget - oversized units are split at
line boundaries.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    path: str
    start_line: int  # 1-based line number of the first line of the chunk
    text: str
    kind: str  # "function" | "class" | "module" | "window" | "file"

    def label(self) -> str:
        return f"{self.path}:{self.start_line} ({self.kind})"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for code."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def _split_lines(
    path: str,
    lines: list[str],
    start_line: int,
    budget_tokens: int,
    kind: str,
    overlap: int = 0,
) -> list[Chunk]:
    """Split ``lines`` into chunks that each fit ``budget_tokens``."""
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        # A single line that alone busts the budget: hard-split it by chars.
        if estimate_tokens(lines[i]) > budget_tokens:
            max_chars = max(1, budget_tokens * 4)
            line = lines[i]
            for k in range(0, len(line), max_chars):
                chunks.append(
                    Chunk(path, start_line + i, line[k : k + max_chars], kind)
                )
            i += 1
            continue
        buf: list[str] = []
        j = i
        while j < n and estimate_tokens(lines[j]) <= budget_tokens:
            if buf and estimate_tokens("\n".join(buf + [lines[j]])) > budget_tokens:
                break
            buf.append(lines[j])
            j += 1
        if not buf:  # pragma: no cover - guarded above, kept for safety
            buf = [lines[i]]
        chunks.append(Chunk(path, start_line + i, "\n".join(buf), kind))
        step = len(buf) - overlap if overlap else len(buf)
        i += max(1, step)
    return chunks


def _python_chunks(path: str, text: str, lines: list[str], budget: int) -> list[Chunk]:
    tree = ast.parse(text)
    units: list[tuple[int, int, str]] = []  # (start, end, kind) 1-based inclusive
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        start = node.lineno
        for deco in getattr(node, "decorator_list", []):
            start = min(start, deco.lineno)
        units.append((start, node.end_lineno or start, kind))
    if not units:
        return _split_lines(path, lines, 1, budget, "file")

    chunks: list[Chunk] = []
    cursor = 1  # next uncovered line (1-based)
    for start, end, kind in units:
        if start > cursor:
            gap = lines[cursor - 1 : start - 1]
            if any(line.strip() for line in gap):
                chunks.extend(_split_lines(path, gap, cursor, budget, "module"))
        segment = lines[start - 1 : end]
        if estimate_tokens("\n".join(segment)) <= budget:
            chunks.append(Chunk(path, start, "\n".join(segment), kind))
        else:
            chunks.extend(_split_lines(path, segment, start, budget, kind))
        cursor = end + 1
    if cursor <= len(lines):
        tail = lines[cursor - 1 :]
        if any(line.strip() for line in tail):
            chunks.extend(_split_lines(path, tail, cursor, budget, "module"))
    return chunks


def chunk_file(path: str | Path, budget_tokens: int = 1500) -> list[Chunk]:
    """Split one file into budget-sized, code-aware chunks."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    path = str(path)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    lines = text.splitlines()
    if estimate_tokens(text) <= budget_tokens and Path(path).suffix != ".py":
        return [Chunk(path, 1, text, "file")]
    if Path(path).suffix == ".py":
        try:
            return _python_chunks(path, text, lines, budget_tokens)
        except SyntaxError:
            pass  # fall through to plain windows
    overlap = 10
    return _split_lines(path, lines, 1, budget_tokens, "window", overlap=overlap)
