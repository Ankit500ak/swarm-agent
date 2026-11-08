"""The three workflows: fix, analyze, gen-tests.

Each workflow follows the same shape: prepare context locally, ask the swarm
(a single synthesis call on the fastest peer, or a fan-out across all of
them), then act locally - patch files, run tests, write the test file. The
models only ever *propose*; everything that touches the disk goes through
the verifying patcher.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import prompts
from .chunking import chunk_file
from .config import Config
from .discovery import discover
from .distributor import Distributor
from .ollama import SwarmError
from .patcher import PatchError, apply_unified_diff, extract_diff
from .runner import run_tests

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

    messages = [
        {"role": "system", "content": prompts.FIX_SYSTEM},
        {"role": "user", "content": prompts.build_fix_prompt(error_text, code_blocks)},
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

    print(f"swarm-agent: tests failed (exit {code}); asking the swarm for one repair")
    fresh_blocks = []
    for rel in changed:
        path = repo_root / rel
        if path.is_file():
            chunks = chunk_file(path, budget_tokens=config.chunk_tokens)
            fresh_blocks.append((rel, "\n".join(c.text for c in chunks)))
    repair_messages = [
        {"role": "system", "content": prompts.FIX_SYSTEM},
        {"role": "user", "content": prompts.build_repair_prompt(error_text, tail, fresh_blocks)},
    ]
    repair_diff = extract_diff(distributor.ask(repair_messages))
    if repair_diff:
        try:
            changed2 = apply_unified_diff(repo_root, repair_diff)
            print(f"swarm-agent: repair applied to {', '.join(changed2)}")
            code, tail = run_tests(cmd, repo_root)
            if code == 0:
                print(f"swarm-agent: tests pass after repair ({cmd})")
                return 0
        except PatchError as exc:
            print(f"swarm-agent: repair diff did not apply: {exc}")

    print(f"swarm-agent: tests still failing (exit {code}). Last output:")
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
