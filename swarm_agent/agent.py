"""The three workflows: fix, analyze, gen-tests.

Each workflow follows the same shape: prepare context locally, ask the swarm
(a single synthesis call on the fastest peer, or a fan-out across all of
them), then act locally - patch files, run tests, write the test file. The
models only ever *propose*; everything that touches the disk goes through
the verifying patcher.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from . import prompts
from .chunking import chunk_file, estimate_tokens
from .config import Config
from .discovery import discover
from .distributor import Distributor
from .ollama import SwarmError
from .patcher import PatchError, apply_multi_file_diff, apply_unified_diff, extract_diff
from .runner import parse_failures, run_tests

NO_PEER_HINT = (
    "no Ollama peer found - start one with: ollama serve, or set SWARM_PEERS"
)

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache", "dist",
    "build", ".idea", ".next", "target",
}
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".pyc", ".class", ".jar",
    ".woff", ".woff2", ".ttf", ".mp3", ".mp4", ".mov", ".db", ".sqlite",
}
MAX_FILE_BYTES = 1_000_000


def get_distributor(config: Config) -> Distributor:
    """Discover peers and build a distributor, or fail with a clear hint."""
    peers = discover(config)
    if not peers:
        raise SwarmError(NO_PEER_HINT)
    return Distributor(peers, config.model)


# --------------------------------------------------------------------------
# fix
# --------------------------------------------------------------------------

_TRACEBACK_RE = re.compile(r'File "([^"]+)", line (\d+)')


def _gather_traceback_files(
    error_text: str, repo_root: Path, config: Config
) -> list[tuple[str, str]]:
    """Files named by the traceback, as (display_path, content) pairs."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_path, _line in _TRACEBACK_RE.findall(error_text):
        candidates = [Path(raw_path), repo_root / raw_path]
        target: Path | None = None
        for cand in candidates:
            if cand.is_file():
                target = cand
                break
        if target is None:
            # Last resort: find the basename anywhere under the repo root.
            matches = [
                p for p in repo_root.rglob(Path(raw_path).name)
                if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
            ]
            if matches:
                target = matches[0]
        if target is None:
            continue
        try:
            rel = str(target.relative_to(repo_root))
        except ValueError:
            rel = str(target)
        if rel in seen:
            continue
        seen.add(rel)
        chunks = chunk_file(target, budget_tokens=config.chunk_tokens)
        text = "\n".join(c.text for c in chunks)
        found.append((rel, text))
    return found


#: Index hits considered for related context before budget filtering.
RELATED_SEARCH_K = 20

_TRACE_NAME_RE = re.compile(r'File "[^"]+", line \d+, in ([A-Za-z_][\w.]*)')


def _related_query(error_text: str, code_blocks: list[tuple[str, str]]) -> str:
    """Search text for the index: the error plus what it points at."""
    parts = [error_text]
    parts.extend(path for path, _ in code_blocks)
    parts.extend(_TRACE_NAME_RE.findall(error_text))
    return "\n".join(parts)


def _gather_related_context(
    error_text: str,
    repo_root: Path,
    code_blocks: list[tuple[str, str]],
    distributor: Distributor,
    config: Config,
) -> list[tuple[str, int, str]]:
    """Chunks from *other* files that the repo index ranks as related.

    Builds (or incrementally refreshes) the persistent index for
    ``repo_root`` with the same client and model as the fix call itself,
    searches it with the error text plus the traceback's file names and
    identifiers, and returns ``(path, start_line, text)`` triples in the
    index's ranking order. Files already in the prompt are excluded and
    no chunk is returned twice.

    The section is capped at the flow's chunk budget
    (``config.chunk_tokens``, estimated with :func:`estimate_tokens`):
    when the hits do not all fit, the related context is what shrinks -
    the traceback files are never trimmed to make room for it.

    Any failure - the index cannot be built, the search raises, the
    client cannot embed - degrades to an empty list with a printed
    note. Retrieval is a hint for the model, never a reason for the
    fix itself to fail.
    """
    try:
        # Deferred import: swarm_agent.index imports constants from
        # this module, so it cannot be imported at module level.
        from .index import RepoIndex

        client = distributor.client_for(distributor.best_peer())
        index = RepoIndex.build(
            repo_root,
            client=client,
            model=config.model,
            budget_tokens=config.chunk_tokens,
        )
        hits = index.search(
            _related_query(error_text, code_blocks), k=RELATED_SEARCH_K
        )
    except Exception as exc:
        print(
            f"swarm-agent: repo index unavailable ({exc}); "
            "continuing without related context"
        )
        return []

    included = {path for path, _ in code_blocks}
    seen: set[tuple[str, int]] = set()
    remaining = config.chunk_tokens
    related: list[tuple[str, int, str]] = []
    for hit in hits:
        path = hit["path"]
        if path in included:
            continue
        key = (path, hit["start_line"])
        if key in seen:
            continue
        seen.add(key)
        cost = estimate_tokens(hit["text"])
        if cost > remaining:
            continue  # shrink the related context, not the traceback
        remaining -= cost
        related.append((path, hit["start_line"], hit["text"]))
    return related


def _format_related_context(related: list[tuple[str, int, str]]) -> str:
    """The related chunks as a labelled section of the fix prompt."""
    lines = [
        "--- Related context (from repo index) ---",
        "The chunks below are from other files in this repository that "
        "the index ranked as most related to the error. They were not "
        "named in the traceback, but the root cause may live in one of "
        "them. If the fix belongs in one of these files, use its path "
        "exactly as shown.",
        "",
    ]
    for path, start_line, text in related:
        lines.append(
            f"FILE: {path} (lines from {start_line})\n```python\n{text}\n```"
        )
    lines.append(
        "Fix the root cause. Reply with the unified diff only, "
        "in a ```diff fence."
    )
    return "\n".join(lines)


def fix_flow(
    error_text: str,
    repo_root: str | Path,
    config: Config,
    apply: bool = False,
    test_cmd: str | None = None,
) -> int:
    """Diagnose ``error_text`` against the repo; optionally apply + verify.

    Returns a process-style exit code: 0 on success (or a clean dry run),
    1 when no fix could be produced or the tests still fail.
    """
    repo_root = Path(repo_root)
    distributor = get_distributor(config)
    code_blocks = _gather_traceback_files(error_text, repo_root, config)
    if code_blocks:
        print(f"swarm-agent: context from {len(code_blocks)} file(s): "
              + ", ".join(p for p, _ in code_blocks))
    else:
        print("swarm-agent: no traceback files found; diagnosing from the error alone")

    related = _gather_related_context(
        error_text, repo_root, code_blocks, distributor, config
    )
    if related:
        print("swarm-agent: related context from index: "
              + ", ".join(f"{p}:{ln}" for p, ln, _ in related))

    user_content = prompts.build_fix_prompt(error_text, code_blocks)
    if related:
        user_content += "\n\n" + _format_related_context(related)
    messages = [
        {"role": "system", "content": prompts.FIX_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    reply = distributor.ask(messages)
    if "NO_FIX_POSSIBLE" in reply and extract_diff(reply) is None:
        print("swarm-agent: the swarm could not fix this from the code shown.")
        print(reply)
        return 1
    diff = extract_diff(reply)
    if diff is None:
        print("swarm-agent: the model did not return a usable diff. Raw reply:")
        print(reply)
        return 1

    if not apply:
        print("\n--- proposed fix (dry run; re-run with --apply to write it) ---")
        print(diff)
        # Verify it *would* apply, so a dry run is a meaningful signal.
        try:
            changed = apply_unified_diff(repo_root, diff, dry_run=True)
            print(f"--- dry run OK: would update {', '.join(changed)} ---")
        except PatchError as exc:
            print(f"--- warning: this diff does not apply cleanly: {exc} ---")
            return 1
        return 0

    try:
        changed = apply_unified_diff(repo_root, diff)
    except PatchError as exc:
        print(f"swarm-agent: refusing to apply the diff: {exc}")
        return 1
    print(f"swarm-agent: applied changes to {', '.join(changed)}")

    cmd = test_cmd if test_cmd is not None else config.test_command
    if not cmd:
        print("swarm-agent: no test command configured; skipping verification")
        return 0

    code, tail = run_tests(cmd, repo_root)
    if code == 0:
        print(f"swarm-agent: tests pass ({cmd})")
        return 0

    return _repair_loop(
        error_text=error_text,
        repo_root=repo_root,
        config=config,
        distributor=distributor,
        cmd=cmd,
        initial_diff=diff,
        changed=changed,
        related=related,
        code=code,
        tail=tail,
    )


# --------------------------------------------------------------------------
# repair loop v2 (issue #16)
# --------------------------------------------------------------------------

#: Repair attempts allowed after the initial fix, unless configured.
DEFAULT_MAX_REPAIRS = 2


def _max_repairs(config: Config) -> int:
    """How many repair attempts may follow the initial fix.

    A ``max_repairs`` attribute on the config object wins if one is
    ever added to :class:`Config`; otherwise the ``SWARM_MAX_REPAIRS``
    environment variable; otherwise :data:`DEFAULT_MAX_REPAIRS`.
    Unparseable values degrade to the default and negative values to
    zero - a loop bound is not a place to crash the flow.
    """
    value = getattr(config, "max_repairs", None)
    if value is None:
        raw = os.environ.get("SWARM_MAX_REPAIRS", "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = None
    if value is None:
        return DEFAULT_MAX_REPAIRS
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_REPAIRS


def _infer_ecosystem(cmd: str) -> str:
    """Best-guess ecosystem for :func:`parse_failures` from a test command."""
    lowered = cmd.lower()
    if "pytest" in lowered or "python" in lowered:
        return "pytest"
    if "jest" in lowered or "npm" in lowered or "node" in lowered:
        return "jest"
    return ""


_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def _diff_summary(diff_text: str) -> str:
    """One-line summary of a diff, for the attempt history."""
    if not diff_text:
        return "no diff"
    files = _DIFF_FILE_RE.findall(diff_text)
    added = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    names = ", ".join(files) if files else "unknown file(s)"
    return f"{names} (+{added}/-{removed})"


def _current_state_blocks(
    repo_root: Path, paths: list[str], config: Config
) -> list[tuple[str, str]]:
    """Current contents of ``paths`` as (display_path, text) blocks.

    Chunked exactly like the traceback gatherer chunks files. Missing
    files are skipped - a repair may have deleted one.
    """
    blocks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rel in paths:
        if rel in seen:
            continue
        seen.add(rel)
        target = repo_root / rel
        if not target.is_file():
            continue
        chunks = chunk_file(target, budget_tokens=config.chunk_tokens)
        blocks.append((rel, "\n".join(c.text for c in chunks)))
    return blocks


def _repair_loop(
    *,
    error_text: str,
    repo_root: Path,
    config: Config,
    distributor: Distributor,
    cmd: str,
    initial_diff: str,
    changed: list[str],
    related: list[tuple[str, int, str]],
    code: int,
    tail: str,
) -> int:
    """Re-diagnose from structured failures until the tests pass or
    the repair bound is reached. Returns the flow's exit code.

    Each round feeds the model the original error, the previous
    attempt's diff, and the structured failure records parsed from the
    last test run (the frozen ``{"file", "test", "message"}`` shape of
    issue #10, rendered by the re-diagnosis contract of issue #9),
    plus the current contents of every file patched so far and of the
    files the failures name.

    Corrected diffs are applied with
    :func:`apply_multi_file_diff`, which verifies against the
    *current* contents - i.e. the state the previous attempts left
    behind. A diff written against stale (pre-attempt) contents fails
    verification, counts as a failed attempt, and leaves every file on
    disk untouched; the loop then stops rather than guessing.
    """
    ecosystem = _infer_ecosystem(cmd)
    max_repairs = _max_repairs(config)
    failures = parse_failures(tail, ecosystem)
    history: list[dict] = [
        {
            "attempt": 1,
            "diff": initial_diff,
            "outcome": f"tests failed (exit {code})",
            "failures": failures,
        }
    ]
    if max_repairs == 0:
        print(
            f"swarm-agent: tests still failing (exit {code}); "
            "repairs disabled (max repairs = 0). Last output:"
        )
        print(tail)
        return 1

    changed_paths = list(changed)
    previous_diff = initial_diff
    for repair in range(1, max_repairs + 1):
        attempt = repair + 1
        print(
            f"swarm-agent: tests failed (exit {code}); repair "
            f"{repair}/{max_repairs} - re-diagnosing from "
            f"{len(failures)} structured failure(s)"
        )
        block_paths = list(changed_paths)
        for record in failures:
            named = record.get("file") or ""
            if (
                named
                and named not in block_paths
                and (repo_root / named).is_file()
            ):
                block_paths.append(named)
        blocks = _current_state_blocks(repo_root, block_paths, config)
        user_content = prompts.build_rediagnosis_prompt(
            error_text, previous_diff, failures, blocks, attempt=attempt
        )
        # The index context gathered for the first attempt can still
        # help - but only for files no attempt has rewritten since;
        # their gathered text would be stale.
        fresh_related = [t for t in related if t[0] not in changed_paths]
        if fresh_related:
            user_content += "\n\n" + _format_related_context(fresh_related)
        reply = distributor.ask(
            [
                {"role": "system", "content": prompts.REDIAGNOSIS_SYSTEM},
                {"role": "user", "content": user_content},
            ]
        )
        new_diff = extract_diff(reply)
        if new_diff is None:
            reason = (
                "the swarm reported NO_FIX_POSSIBLE"
                if "NO_FIX_POSSIBLE" in reply
                else "the model returned no usable diff"
            )
            history.append(
                {
                    "attempt": attempt,
                    "diff": "",
                    "outcome": reason,
                    "failures": failures,
                }
            )
            print(f"swarm-agent: repair attempt {attempt}: {reason}; giving up")
            break
        try:
            result = apply_multi_file_diff(repo_root, new_diff)
        except PatchError as exc:
            history.append(
                {
                    "attempt": attempt,
                    "diff": new_diff,
                    "outcome": (
                        f"diff did not apply against current contents: {exc}"
                    ),
                    "failures": failures,
                }
            )
            print(
                f"swarm-agent: repair attempt {attempt}: diff did not "
                f"apply ({exc}); nothing was written - giving up"
            )
            break
        for path in result.changed_paths:
            if path not in changed_paths:
                changed_paths.append(path)
        print(
            f"swarm-agent: repair attempt {attempt} applied to "
            f"{', '.join(result.changed_paths)}"
        )
        previous_diff = new_diff
        code, tail = run_tests(cmd, repo_root)
        if code == 0:
            print(
                f"swarm-agent: tests pass after repair - "
                f"fixed on attempt {attempt} ({cmd})"
            )
            return 0
        failures = parse_failures(tail, ecosystem)
        history.append(
            {
                "attempt": attempt,
                "diff": new_diff,
                "outcome": f"tests failed (exit {code})",
                "failures": failures,
            }
        )

    print(
        f"swarm-agent: giving up after {len(history)} attempt(s) "
        f"(initial fix + up to {max_repairs} repair(s)). Attempt history:"
    )
    for entry in history:
        print(
            f"  attempt {entry['attempt']}: {_diff_summary(entry['diff'])}"
            f" - {entry['outcome']} "
            f"[{len(entry['failures'])} failing record(s)]"
        )
    if failures:
        print("swarm-agent: last failures:")
        for record in failures:
            print(
                f"  - {record.get('file', '')} :: {record.get('test', '')}"
                f" - {record.get('message', '')}"
            )
    print("swarm-agent: last test output:")
    print(tail)
    return 1


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _iter_source_files(root: Path):
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in BINARY_EXTS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                path.read_text(encoding="utf-8")  # probe: must decode as text
            except (OSError, UnicodeDecodeError):
                continue
            yield path


def analyze_flow(path: str | Path, config: Config, question: str | None = None) -> str:
    """Fan chunk summaries out across the swarm, then merge them into a report."""
    root = Path(path)
    if not root.exists():
        raise SwarmError(f"path does not exist: {root}")
    distributor = get_distributor(config)

    chunks = []
    files = list(_iter_source_files(root))
    for file in files:
        try:
            chunks.extend(chunk_file(file, budget_tokens=config.chunk_tokens))
        except (OSError, ValueError):
            continue
    if not chunks:
        raise SwarmError(f"no readable source files found under {root}")
    print(
        f"swarm-agent: {len(files)} file(s), {len(chunks)} chunk(s), "
        f"{len(distributor.peers)} peer(s) - analysing...",
        file=sys.stderr,
    )

    def summarise(client, chunk):
        reply = client.chat([
            {"role": "system", "content": prompts.ANALYZE_SYSTEM},
            {"role": "user", "content": prompts.build_chunk_prompt(
                chunk.path, chunk.start_line, chunk.kind, chunk.text)},
        ])
        return f"[{chunk.label()}]\n{reply.strip()}"

    results = distributor.map_units(chunks, summarise)
    summaries: list[str] = []
    failures = 0
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            failures += 1
            summaries.append(f"[{chunk.label()}]\nPURPOSE: (analysis failed: {result})")
        else:
            summaries.append(result)
    if failures:
        print(f"swarm-agent: {failures} chunk(s) failed on every peer", file=sys.stderr)

    # Keep the synthesis prompt bounded even for very large repos:
    # keep the first and last summaries, note what was elided.
    if len("\n\n".join(summaries)) > 48000 and len(summaries) > 30:
        omitted = len(summaries) - 30
        summaries = (
            summaries[:20]
            + [f"[... {omitted} further chunk summaries omitted for length ...]"]
            + summaries[-10:]
        )

    report = distributor.ask([
        {"role": "system", "content": prompts.ANALYZE_SYNTHESIS_SYSTEM},
        {"role": "user", "content": prompts.build_synthesis_prompt(summaries, question)},
    ])
    print(report)
    return report


# --------------------------------------------------------------------------
# gen-tests
# --------------------------------------------------------------------------

def _extract_python(text: str) -> str | None:
    m = re.search(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    if "def test_" in text:
        return text.strip()
    return None


def gen_tests_flow(
    target: str | Path,
    config: Config,
    write: bool = False,
    run: bool = False,
) -> str:
    """Generate pytest tests for one module; optionally write and run them."""
    target = Path(target)
    if not target.is_file():
        raise SwarmError(f"no such file: {target}")
    source = target.read_text(encoding="utf-8")
    distributor = get_distributor(config)

    reply = distributor.ask([
        {"role": "system", "content": prompts.TESTGEN_SYSTEM},
        {"role": "user", "content": prompts.build_testgen_prompt(target.stem, source)},
    ])
    code = _extract_python(reply)
    if code is None:
        raise SwarmError("the model did not return a usable test file")

    if not write:
        print(code)
        return code

    tests_dir = Path.cwd() / "tests"
    tests_dir.mkdir(exist_ok=True)
    out = tests_dir / f"test_{target.stem}_generated.py"
    out.write_text(code.rstrip() + "\n", encoding="utf-8")
    print(f"swarm-agent: wrote {out}")
    if run:
        code_rc, tail = run_tests(
            f"{sys.executable} -m pytest {out} -q", Path.cwd()
        )
        print(tail)
        print(f"swarm-agent: generated tests exit code {code_rc}")
    return code
