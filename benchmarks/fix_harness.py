"""Fix benchmark: run ``swarm-agent fix --apply`` against seeded bugs.

Five tiny repos, each with one seeded bug (see :data:`FIX_REPOS`):
the two multi-file examples from issue #12, and three new one-file
repos under ``benchmarks/fix_repos/``. The runner copies each repo
to a scratch directory (the committed repo is never mutated), runs
its tests to capture the failure, pipes that failure to
``swarm-agent fix --apply`` against a local model, and then
independently re-runs the tests.

The outcome is grounded in that independent re-run, not in what the
fix run claimed:

- ``fixed attempt 1`` — the tests pass after the first fix, no
  repair round was needed.
- ``fixed after repair`` — the tests pass, but only after the repair
  loop ran (the attempt number comes from fix's own output).
- ``failed`` — the tests still fail after the fix run, whatever it
  claimed. (A fix that exits 0 while the tests still fail is
  recorded as failed, with a note saying so.)

Honest numbers only. A repo whose baseline tests pass is recorded
as failed with a note, never silently skipped. No row is ever
invented: with no live model the runner does nothing — the fix
subprocess fails at peer discovery and every repo is recorded
failed with that error, which is the truthful result for that run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from swarm_agent import backends
from swarm_agent.config import DEFAULT_MODEL
from swarm_agent.runner import detect_test_command, run_tests

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).parent / "FIX_RESULTS.md"

#: (path relative to the repo root, one-line bug description)
#:
#: The five are the two multi-file examples from issue #12 plus three
#: new one-file repos under ``benchmarks/fix_repos/``. The demo-kit
#: project (issue #5, ``examples/broken-demo``) is deliberately NOT
#: one of the five: it lives on its own unmerged branch, and this
#: harness must be self-contained on main.
FIX_REPOS: list[tuple[str, str]] = [
    (
        "examples/cross_file_crash",
        "two-file: prices stored as strings in catalog.py, crash in checkout.py",
    ),
    (
        "examples/signature_change",
        "two-file: function signature changed in store.py, caller in report.py not updated",
    ),
    (
        "benchmarks/fix_repos/off_by_one",
        "one-line slice overshoot: items[-n - 1:] returns n+1 items",
    ),
    (
        "benchmarks/fix_repos/empty_average",
        "missing empty-list guard: ZeroDivisionError on average([])",
    ),
    (
        "benchmarks/fix_repos/threshold",
        "boundary bug: is_adult(18) returns False (age > 18 instead of >= 18)",
    ),
]

FIX_RESULTS_HEADER = """# Fix-benchmark results — swarm-agent fix --apply on seeded bugs

Methodology: see [README.md](README.md). Each row is one run of
`swarm-agent fix --apply` against a seeded-bug repo, copied to a
scratch directory first (the committed repo is never mutated). The
outcome is grounded in an independent re-run of the repo's tests
after the fix, not in what the fix run claimed: `fixed attempt 1`
means the first fix made the tests pass, `fixed after repair` means
the repair loop had to run, `failed` means the tests still fail.

| date | repo | bug | outcome | attempts | model | machine | notes |
|---|---|---|---|---|---|---|---|
"""

_REPAIR_SUCCESS = re.compile(r"tests pass after repair - fixed on attempt (\d+)")
_REPAIR_START = re.compile(r"; repair \d+/\d+ - re-diagnosing")


def machine_label() -> str:
    """Short machine description for the results table.

    (Duplicated from the analyse harness in ``benchmarks/harness.py``
    rather than imported: the two benchmark PRs are independent, and
    this module must work on its own until both are merged.)
    """
    report = backends.hardware_report()
    chip = report.get("chip") or report.get("machine") or "unknown"
    mem = report.get("unified_memory_gb") or report.get("memory_gb")
    return f"{chip}, {mem} GB" if mem is not None else str(chip)


def classify_fix_output(output: str, verified: bool) -> tuple[str, int]:
    """Classify a fix run from its output and the independent re-run.

    ``verified`` is whether the repo's tests pass after the fix run
    (re-run by the harness itself). It is the ground truth: a fix
    that claimed success while the tests still fail is ``failed``.
    """
    repair_rounds = len(_REPAIR_START.findall(output))
    if not verified:
        return "failed", 1 + repair_rounds
    match = _REPAIR_SUCCESS.search(output)
    if match:
        return "fixed after repair", int(match.group(1))
    if repair_rounds:
        return "fixed after repair", 1 + repair_rounds
    return "fixed attempt 1", 1


def format_fix_row(result: dict) -> str:
    """One markdown table row for a fix run's metrics dict."""
    return (
        f"| {result['date']} | {result['repo']} | {result['bug']} | "
        f"{result['outcome']} | {result['attempts']} | {result['model']} | "
        f"{result['machine']} | {result['notes']} |"
    )


def append_fix_results(path: str | Path, rows: list[str]) -> None:
    """Append rows to the fix-results table, creating it with a header first."""
    path = Path(path)
    if not path.exists():
        path.write_text(FIX_RESULTS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row + "\n")


def _fix_command() -> list[str]:
    """How to invoke swarm-agent: the console script, else the module."""
    script = shutil.which("swarm-agent")
    if script:
        return [script]
    return [sys.executable, "-m", "swarm_agent.cli"]


def _run_fix(
    repo_copy: Path,
    crash_text: str,
    model: str,
    peers: list[str] | None,
) -> tuple[int, str]:
    """Pipe ``crash_text`` to ``swarm-agent fix --apply`` in ``repo_copy``.

    Returns (exit code, combined output). The model and peers reach
    the subprocess through the environment (``SWARM_MODEL`` /
    ``SWARM_PEERS``), because the fix command takes no model flag.
    """
    env = dict(os.environ)
    env["SWARM_MODEL"] = model
    if peers:
        env["SWARM_PEERS"] = ",".join(peers)
    try:
        proc = subprocess.run(
            _fix_command() + ["fix", "--apply", "--path", str(repo_copy)],
            input=crash_text,
            capture_output=True,
            text=True,
            cwd=repo_copy,
            env=env,
            timeout=900,
        )
        return proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial += (exc.stderr or b"") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 1, partial + "\nswarm-agent: fix run timed out after 900s"


def run_repo_fix(
    repo_path: str | Path,
    bug: str,
    model: str,
    peers: list[str] | None = None,
    work_root: str | Path | None = None,
) -> dict:
    """Fix one seeded-bug repo in a scratch copy; return the metrics dict.

    The committed repo is never touched: it is copied to a scratch
    directory, the fix runs there, and the scratch directory is
    removed afterwards.
    """
    repo_path = Path(repo_path)
    base = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "repo": repo_path.name,
        "bug": bug,
        "model": model,
        "machine": machine_label(),
    }
    scratch = Path(tempfile.mkdtemp(prefix="fixbench-", dir=work_root))
    copy = scratch / repo_path.name
    shutil.copytree(
        repo_path,
        copy,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".venv"),
    )
    try:
        cmd = detect_test_command(copy)
        code, tail = run_tests(cmd, copy)
        if code == 0:
            return {
                **base,
                "outcome": "failed",
                "attempts": 0,
                "notes": "baseline tests pass — repo not broken",
            }
        crash_file = copy / "crash.log"
        crash_text = crash_file.read_text(encoding="utf-8") if crash_file.is_file() else tail
        exit_code, output = _run_fix(copy, crash_text, model, peers)
        verify_code, _ = run_tests(cmd, copy)
        verified = verify_code == 0
        outcome, attempts = classify_fix_output(output, verified)
        notes = ""
        if not verified:
            notes = f"fix exit {exit_code}"
            if exit_code == 0:
                notes += " (reported success, tests still fail)"
        return {**base, "outcome": outcome, "attempts": attempts, "notes": notes}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_fix_benchmark(
    repos: list[tuple[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
    peers: list[str] | None = None,
    results_path: str | Path = RESULTS_PATH,
    work_root: str | Path | None = None,
) -> list[dict]:
    """Fix every repo in ``repos`` (default: all five) and append the rows.

    Returns the list of metrics dicts. With no repos, runs nothing
    and writes nothing.
    """
    repos = list(repos) if repos is not None else FIX_REPOS
    if not repos:
        return []
    rows = []
    for rel_path, bug in repos:
        result = run_repo_fix(REPO_ROOT / rel_path, bug, model, peers, work_root)
        rows.append(result)
    append_fix_results(results_path, [format_fix_row(r) for r in rows])
    return rows
