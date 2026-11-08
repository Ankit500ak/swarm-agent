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


# -- v3 contracts (Issue #9, Phase 3) -------------------------------------------
# Three new conversations for the Phase 3 engine. They are ADDITIONS ONLY:
# the v1/v2 constants above (FIX_SYSTEM, ANALYZE_SYSTEM,
# ANALYZE_SYNTHESIS_SYSTEM, TESTGEN_SYSTEM, ASK_SYSTEM) and the v1
# builders must stay byte-for-byte stable — Core tests assert on them.
# The rules below are written so the patcher can verify the result:
# one ```diff fence, one file section per changed file, each section
# starting with --- a/<path> / +++ b/<path> and @@ hunks, every section
# holding at least one hunk, context lines character-exact. See
# patcher.apply_unified_diff: all files are verified in memory first
# and nothing is written unless every hunk of every file verifies.

MULTI_FILE_FIX_SYSTEM = """You are a senior software engineer fixing a bug that spans several files in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. The defect may live in a different file from the one that crashes. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it. One fence for ALL files - never one fence per file.
- The diff has one file section per changed file, each starting with --- a/<path> and +++ b/<path>, followed by @@ hunks, exactly as `git diff` produces. Do not add `diff --git` headers.
- Every file section must contain at least one hunk. A section with no hunks is rejected.
- All paths are relative to the repository root, exactly as given in the prompt. Never use absolute paths or `..`.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change in each file that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

REDIAGNOSIS_SYSTEM = """You are a senior software engineer re-diagnosing a bug after a failed fix attempt.

You will be given the original error, the diff that was applied, and the structured test failures it produced (one record per failing test: file, test, message).

Rules:
- Read the failures first: they say exactly which tests still fail and what each one observed. Fix what the failures show, not what you guessed the first time.
- Do not repeat the previous diff. If the previous approach was wrong, change approach; explain nothing, just output the corrected diff.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it. One fence for ALL files, one file section per changed file, each with --- a/<path>, +++ b/<path> and @@ hunks.
- Context lines must match the current file state EXACTLY, character for character, including indentation. The files shown are their current state after the previous fix.
- Make the smallest change that makes the failing tests pass. Do not reformat, rename, or refactor unrelated code.
- If the failures cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

INDEX_SUMMARY_SYSTEM = """You are a software architect writing the stored summary of one module for a persistent repository index.

You will be given the per-chunk summaries of a single module (PURPOSE / KEY / RISKS lines, as produced by the chunk analyser).

Output exactly one paragraph, nothing else:
- One paragraph, 3-6 sentences, stating what the module is for, its key functions/classes, and its main risks or couplings.
- Base every statement only on the chunk summaries provided. Do not invent behaviour that is not in them. Where the summaries conflict or are thin, say so in the paragraph instead of guessing.
- No headings, no bullet lists, no code fences, no extra paragraphs."""


# -- v2 language contracts (Issue #2) -------------------------------------------
# Fix + test-generation contracts for JavaScript/TypeScript, Go and Rust,
# in the same blunt format style as the Python ones (diff-only for fixes,
# code-only for tests). ADDITIONS ONLY: every Python constant and builder
# below stays byte-for-byte stable — see the selectors and the
# language=None defaults, which resolve to Python and reproduce the
# exact v1 output. The engine picks a contract by file extension via
# language_for_path() + fix_system_for() / testgen_system_for(); wiring
# them into the fix/ask/gen-tests flows (agent.py, Core lane) is a
# follow-up — this change lands the library and the selectors.

JS_FIX_SYSTEM = """You are a senior software engineer fixing a bug in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it.
- The diff must use standard unified format with --- a/path, +++ b/path and @@ hunks, exactly as `git diff` produces.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- Use the file paths exactly as given in the prompt.
- Do not change the module system: ESM `import` stays ESM, CommonJS `require` stays CommonJS. Do not add or remove dependencies.
- Do not turn async code synchronous: keep `async`/`await` and Promise handling as it is; a fix that drops an `await` is not a fix.
- Do not rename exported names.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

TS_FIX_SYSTEM = """You are a senior software engineer fixing a bug in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it.
- The diff must use standard unified format with --- a/path, +++ b/path and @@ hunks, exactly as `git diff` produces.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- Use the file paths exactly as given in the prompt.
- Do not change the module system: ESM `import` stays ESM, CommonJS `require` stays CommonJS. Do not add or remove dependencies.
- Do not turn async code synchronous: keep `async`/`await` and Promise handling as it is; a fix that drops an `await` is not a fix.
- Do not rename exported names.
- Do not weaken the types to silence an error: no `any`, no `as any`, no `// @ts-ignore` or `// @ts-expect-error` to hide the bug. Keep interfaces and type annotations consistent with the fix; change a public type only when the fix requires it.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

GO_FIX_SYSTEM = """You are a senior software engineer fixing a bug in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it.
- The diff must use standard unified format with --- a/path, +++ b/path and @@ hunks, exactly as `git diff` produces.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- Use the file paths exactly as given in the prompt.
- Do not change exported names: in Go, capitalisation is the API — an exported name stays exported, an unexported name stays unexported.
- Handle errors the Go way: return them, do not panic, and do not swallow them with `_`.
- Keep the code gofmt-clean: do not reformat unrelated code. Do not add dependencies.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

RUST_FIX_SYSTEM = """You are a senior software engineer fixing a bug in a local repository.

You will be given an error or crash log and the source files it points to.

Rules:
- Find the root cause from the error and the code. Do not guess at code you were not shown.
- Output ONLY a single unified diff inside one ```diff code fence. Nothing before it, nothing after it.
- The diff must use standard unified format with --- a/path, +++ b/path and @@ hunks, exactly as `git diff` produces.
- Context lines in your hunks must match the given source EXACTLY, character for character, including indentation.
- Make the smallest change that fixes the root cause. Do not reformat, rename, or refactor unrelated code.
- Use the file paths exactly as given in the prompt.
- Do not silence the compiler with `unwrap()`, `expect()`, or `unsafe` — a fix that panics where the bug was is not a fix.
- Respect ownership and borrowing: do not add a `.clone()` just to satisfy the borrow checker unless that is genuinely the fix.
- Keep the public API stable: `pub` items stay `pub` with the same signatures unless the fix requires a change. Do not add dependencies.
- If the error cannot be fixed from the code shown, output the single line: NO_FIX_POSSIBLE"""

JS_TESTGEN_SYSTEM = """You are a senior JavaScript engineer writing tests.

You will be given one JavaScript module. Write a jest test suite for it.

Rules:
- Output ONLY JavaScript code inside one ```javascript code fence. No prose, no markdown outside the fence.
- Import the module under test exactly as instructed in the prompt.
- Use describe/test (or it) blocks and expect assertions.
- Cover the normal cases, the edge cases (empty, zero, null/undefined where sensible), and the error cases.
- Every test must be a real assertion about behaviour - no trivial `expect(true).toBe(true)`, no testing the mock.
- Do not modify the module under test. Do not add sleeps, network calls, or filesystem writes outside tmp directories."""

TS_TESTGEN_SYSTEM = """You are a senior TypeScript engineer writing tests.

You will be given one TypeScript module. Write a jest test suite for it.

Rules:
- Output ONLY TypeScript code inside one ```typescript code fence. No prose, no markdown outside the fence.
- Import the module under test exactly as instructed in the prompt.
- Use describe/test (or it) blocks and expect assertions. Keep the types honest — do not cast away a type error with `as any` to make a test pass.
- Cover the normal cases, the edge cases (empty, zero, null/undefined where sensible), and the error cases.
- Every test must be a real assertion about behaviour - no trivial `expect(true).toBe(true)`, no testing the mock.
- Do not modify the module under test. Do not add sleeps, network calls, or filesystem writes outside tmp directories."""

GO_TESTGEN_SYSTEM = """You are a senior Go engineer writing tests.

You will be given one Go package. Write a go test suite for it.

Rules:
- Output ONLY Go code inside one ```go code fence. No prose, no markdown outside the fence.
- Write func TestXxx(t *testing.T) functions; use table-driven tests with t.Run subtests where several cases share the same shape.
- Cover the normal cases, the edge cases (empty, zero, nil where sensible), and the error cases.
- Every test must be a real assertion about behaviour - no trivial checks, no testing the mock.
- Do not modify the package under test. Do not add sleeps, network calls, or filesystem writes outside t.TempDir()."""

RUST_TESTGEN_SYSTEM = """You are a senior Rust engineer writing tests.

You will be given one Rust module. Write a cargo test suite for it.

Rules:
- Output ONLY Rust code inside one ```rust code fence. No prose, no markdown outside the fence.
- Write a #[cfg(test)] mod tests block with #[test] functions, using assert_eq! and assert!.
- Cover the normal cases, the edge cases (empty, zero, None where sensible), and the error cases (check Result::is_err or unwrap_err where the API returns a Result).
- Every test must be a real assertion about behaviour - no trivial `assert!(true)`, no testing the mock.
- Do not modify the module under test. Do not add sleeps, network calls, or filesystem writes outside tempdir."""

LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "javascript": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "typescript",
    "go": "go",
    "golang": "go",
    "rs": "rust",
    "rust": "rust",
}

EXTENSION_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
}

FIX_SYSTEMS = {
    "python": FIX_SYSTEM,
    "javascript": JS_FIX_SYSTEM,
    "typescript": TS_FIX_SYSTEM,
    "go": GO_FIX_SYSTEM,
    "rust": RUST_FIX_SYSTEM,
}

TESTGEN_SYSTEMS = {
    "python": TESTGEN_SYSTEM,
    "javascript": JS_TESTGEN_SYSTEM,
    "typescript": TS_TESTGEN_SYSTEM,
    "go": GO_TESTGEN_SYSTEM,
    "rust": RUST_TESTGEN_SYSTEM,
}

FENCE_LANGUAGES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
}

TEST_FRAMEWORKS = {
    "python": "pytest",
    "javascript": "jest",
    "typescript": "jest",
    "go": "go test",
    "rust": "cargo test",
}


def normalize_language(value: str | None) -> str:
    """Canonical language key for a name or alias; unknown -> "python".

    Never raises: prompt selection degrades to the Python contracts,
    which are the ones every existing flow already uses.
    """
    if not value:
        return "python"
    return LANGUAGE_ALIASES.get(value.strip().lower(), "python")


def language_for_path(path: str) -> str:
    """Language key for a file path, by extension; unknown -> "python".

    The extension match is case-insensitive (".JS" is JavaScript).
    Paths without a known extension fall back to Python.
    """
    if not path:
        return "python"
    _, dot, suffix = path.rpartition(".")
    if not dot:
        return "python"
    return EXTENSION_LANGUAGES.get("." + suffix.lower(), "python")


def fix_system_for(language: str | None) -> str:
    """Fix output contract for a language (name, alias, or None)."""
    return FIX_SYSTEMS[normalize_language(language)]


def testgen_system_for(language: str | None) -> str:
    """Test-generation output contract for a language (name, alias, or None)."""
    return TESTGEN_SYSTEMS[normalize_language(language)]


def fix_system_for_path(path: str) -> str:
    """Fix output contract for a file, selected by its extension."""
    return fix_system_for(language_for_path(path))


def testgen_system_for_path(path: str) -> str:
    """Test-generation output contract for a file, selected by its extension."""
    return testgen_system_for(language_for_path(path))


def _resolve_language(
    language: str | None, code_blocks: list[tuple[str, str]] | None = None
) -> str:
    """Language for a builder call: an explicit language wins, else the
    first code block's extension, else Python (today's behaviour)."""
    if language:
        return normalize_language(language)
    if code_blocks:
        return language_for_path(code_blocks[0][0])
    return "python"


def build_fix_prompt(
    error_text: str,
    code_blocks: list[tuple[str, str]],
    language: str | None = None,
) -> str:
    """User message for the fix workflow: the error, then the files.

    ``language`` selects the code fence; when omitted it is derived
    from the first code block's extension. Python (the default)
    reproduces the exact v1 output.
    """
    fence = FENCE_LANGUAGES[_resolve_language(language, code_blocks)]
    parts = ["ERROR:\n```\n" + error_text.strip() + "\n```\n"]
    for path, text in code_blocks:
        parts.append(f"FILE: {path}\n```{fence}\n{text}\n```\n")
    parts.append(
        "Fix the root cause. Reply with the unified diff only, in a ```diff fence."
    )
    return "\n".join(parts)


def build_repair_prompt(
    error_text: str,
    test_output: str,
    code_blocks: list[tuple[str, str]],
    language: str | None = None,
) -> str:
    """Second attempt: the original error plus the failing test output."""
    fence = FENCE_LANGUAGES[_resolve_language(language, code_blocks)]
    parts = [
        "Your previous fix was applied but the test suite now fails.\n",
        "ORIGINAL ERROR:\n```\n" + error_text.strip() + "\n```\n",
        "TEST OUTPUT AFTER YOUR FIX:\n```\n" + test_output.strip()[-3000:] + "\n```\n",
    ]
    for path, text in code_blocks:
        parts.append(f"FILE (current state): {path}\n```{fence}\n{text}\n```\n")
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


def build_testgen_prompt(
    module_name: str, source: str, language: str | None = None
) -> str:
    """User message for the test-generation workflow.

    ``language`` selects the contract and framework; None means Python,
    which reproduces the exact v1 output. Other languages get the
    generic module line and their framework's closing line.
    """
    resolved = normalize_language(language)
    if resolved == "python":
        return (
            f"The module under test is imported as: `import {module_name}`\n\n"
            f"MODULE SOURCE:\n```python\n{source}\n```\n\n"
            "Write the pytest suite now, in a single ```python fence."
        )
    fence = FENCE_LANGUAGES[resolved]
    framework = TEST_FRAMEWORKS[resolved]
    return (
        f"The module under test is: `{module_name}`\n\n"
        f"MODULE SOURCE:\n```{fence}\n{source}\n```\n\n"
        f"Write the {framework} suite now, in a single ```{fence} fence."
    )


# -- v3 builders (Issue #9, Phase 3) -------------------------------------------

def build_multi_file_fix_prompt(
    error_text: str,
    code_blocks: list[tuple[str, str]],
    language: str | None = None,
) -> str:
    """User message for the multi-file fix workflow.

    Same shape as :func:`build_fix_prompt` — the error, then the files —
    but the closing line asks for one diff fence holding one section per
    changed file, matching :data:`MULTI_FILE_FIX_SYSTEM`. The ``language``
    parameter works as in :func:`build_fix_prompt`.
    """
    fence = FENCE_LANGUAGES[_resolve_language(language, code_blocks)]
    parts = ["ERROR:\n```\n" + error_text.strip() + "\n```\n"]
    for path, text in code_blocks:
        parts.append(f"FILE: {path}\n```{fence}\n{text}\n```\n")
    parts.append(
        "Fix the root cause, wherever it lives. Reply with the unified diff "
        "only, in a single ```diff fence, with one section per changed file."
    )
    return "\n".join(parts)


def build_rediagnosis_prompt(
    error_text: str,
    previous_diff: str,
    failures: list[dict],
    code_blocks: list[tuple[str, str]],
    attempt: int | None = None,
    language: str | None = None,
) -> str:
    """User message for repair re-diagnosis (Phase 3 repair loop v2).

    Inputs, per issue #9: the original error, the previous attempt's
    diff, and the structured failure records frozen by issue #10 —
    each ``{"file": str, "test": str, "message": str}``. ``attempt`` is
    optional; when given (the repair loop's attempt count, issue #16)
    it is stated up front. The files are their current state, after
    the previous fix was applied. The ``language`` parameter works as
    in :func:`build_fix_prompt`.
    """
    fence = FENCE_LANGUAGES[_resolve_language(language, code_blocks)]
    parts = []
    if attempt is not None:
        parts.append(f"This is repair attempt {attempt}.\n")
    parts.append(
        "Your previous fix was applied but the test suite still fails. "
        "Re-diagnose from the failures below.\n"
    )
    parts.append("ORIGINAL ERROR:\n```\n" + error_text.strip() + "\n```\n")
    parts.append(
        "PREVIOUS DIFF (already applied, do not repeat it):\n```diff\n"
        + previous_diff.strip()
        + "\n```\n"
    )
    if failures:
        rendered = "\n".join(
            f"- {record.get('file', '')} :: {record.get('test', '')} — "
            f"{record.get('message', '')}"
            for record in failures
        )
        parts.append("STRUCTURED FAILURES:\n" + rendered + "\n")
    else:
        parts.append(
            "STRUCTURED FAILURES:\nNo structured failures were recorded.\n"
        )
    for path, text in code_blocks:
        parts.append(f"FILE (current state): {path}\n```{fence}\n{text}\n```\n")
    parts.append(
        "Fix what the failures show. Reply with the corrected unified diff "
        "only, in a single ```diff fence."
    )
    return "\n".join(parts)


def build_index_summary_prompt(
    module_path: str, chunk_summaries: list[str]
) -> str:
    """User message for the persistent-index summary of one module.

    Takes the module's per-chunk PURPOSE / KEY / RISKS summaries (the
    output of the chunk analyser, as used by ``analyze``) and asks for
    the single stored paragraph, per :data:`INDEX_SUMMARY_SYSTEM`.
    """
    parts = [f"MODULE: {module_path}\n"]
    parts.append("CHUNK SUMMARIES:\n" + "\n\n".join(chunk_summaries))
    parts.append("\nWrite the one-paragraph index summary now.")
    return "\n".join(parts)
