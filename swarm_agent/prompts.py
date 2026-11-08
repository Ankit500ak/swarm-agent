"""Prompts tuned for Qwen2.5-Coder-class open-weight models.

These models follow *blunt, explicit* format instructions well and ramble
when instructions are vague. Every prompt below therefore states the exact
output contract - fences, structure, and what NOT to include - up front.
"""

from __future__ import annotations

FIX_SYSTEM = """You are a senior software engineer fixing a bug in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it.
- The diff must use standard unified format with --- a/path, +++ b/path and @@ hunks, exactly as `git diff` produces.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- Use the file paths exactly as given in the prompt.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

ANALYZE_SYSTEM = """You are a software architect analysing a local codebase, one chunk at a time.

You will be given ONE chunk of one file. Summarise only that chunk.

Output exactly these lines, nothing else:
PURPOSE: <one sentence: what this chunk is for>
KEY: <comma-separated names of the important functions/classes/concepts, or "none">
RISKS: <one short phrase about a risk, smell, or coupling you notice, or "none">
Be factual. Do not invent behaviour that is not in the chunk."""

ANALYZE_SYNTHESIS_SYSTEM = """You are a software architect writing the summary of a repository analysis.

You will be given per-chunk summaries collected from across the repository. Merge them into ONE structured report with exactly these markdown sections:

## Purpose
## Architecture
## Key modules
## Risks
## Suggested next steps

Base every statement only on the summaries provided. Where the summaries conflict or are thin, say so instead of guessing. If a specific question is included, answer it first under a "## Answer" section, then give the report."""

TESTGEN_SYSTEM = """You are a senior Python engineer writing tests.

You will be given one Python module. Write a pytest test suite for it.

Rules:
- Output ONLY Python code inside one ```python code fence. No prose, no markdown outside the fence.
- Import the module under test exactly as instructed in the prompt.
- Cover the normal cases, the edge cases (empty, zero, None where sensible), and the error cases.
- Every test must be a real assertion about behaviour - no trivial `assert True`, no testing the mock.
- Use plain pytest functions. Use pytest.raises for expected exceptions.
- Do not modify the module under test. Do not add sleeps, network calls, or filesystem writes outside tmp_path."""

ASK_SYSTEM = """You are a helpful senior software engineer answering a developer's question. Be direct and concrete. If you show code, keep it minimal and correct."""


def build_fix_prompt(error_text: str, code_blocks: list[tuple[str, str]]) -> str:
    """User message for the fix workflow: the error, then the files."""
    parts = ["ERROR:\n```\n" + error_text.strip() + "\n```\n"]
    for path, text in code_blocks:
        parts.append(f"FILE: {path}\n```python\n{text}\n```\n")
    parts.append(
        "Fix the root cause. Reply with the unified diff only, in a ```diff fence."
    )
    return "\n".join(parts)


def build_repair_prompt(
    error_text: str, test_output: str, code_blocks: list[tuple[str, str]]
) -> str:
    """Second attempt: the original error plus the failing test output."""
    parts = [
        "Your previous fix was applied but the test suite now fails.\n",
        "ORIGINAL ERROR:\n```\n" + error_text.strip() + "\n```\n",
        "TEST OUTPUT AFTER YOUR FIX:\n```\n" + test_output.strip()[-3000:] + "\n```\n",
    ]
    for path, text in code_blocks:
        parts.append(f"FILE (current state): {path}\n```python\n{text}\n```\n")
    parts.append(
        "Fix the remaining failure. Reply with the unified diff only, in a ```diff fence."
    )
    return "\n".join(parts)


def build_chunk_prompt(path: str, start_line: int, kind: str, text: str) -> str:
    return (
        f"FILE: {path} (lines from {start_line}, chunk kind: {kind})\n"
        f"```\n{text}\n```"
    )


def build_synthesis_prompt(summaries: list[str], question: str | None) -> str:
    parts = []
    if question:
        parts.append(f"QUESTION: {question}\n")
    parts.append("PER-CHUNK SUMMARIES:\n" + "\n\n".join(summaries))
    return "\n".join(parts)


def build_testgen_prompt(module_name: str, source: str) -> str:
    return (
        f"The module under test is imported as: `import {module_name}`\n\n"
        f"MODULE SOURCE:\n```python\n{source}\n```\n\n"
        "Write the pytest suite now, in a single ```python fence."
    )
