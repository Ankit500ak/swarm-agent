"""Tests for the fix benchmark (Issue #11).

All tests are pure: no live model is ever contacted. The fix
subprocess is stubbed; the test runs are real (pytest via the
interpreter running the suite) but only ever run in scratch copies.
"""

import subprocess
import sys
from pathlib import Path

from benchmarks import fix_harness

REPO_ROOT = Path(__file__).resolve().parent.parent


def _real_pytest_run_tests(cmd, path):
    """Stand-in for runner.run_tests: real pytest, via this interpreter."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


# -- classification ----------------------------------------------------------


def test_classify_first_try_success():
    outcome, attempts = fix_harness.classify_fix_output(
        "swarm-agent: tests pass (python -m pytest -q)\n", verified=True
    )
    assert (outcome, attempts) == ("fixed attempt 1", 1)


def test_classify_repair_success_uses_printed_attempt():
    output = (
        "swarm-agent: tests failed (exit 1); repair 1/2 - re-diagnosing "
        "from 2 structured failure(s)\n"
        "swarm-agent: tests pass after repair - fixed on attempt 2 "
        "(python -m pytest -q)\n"
    )
    assert fix_harness.classify_fix_output(output, verified=True) == (
        "fixed after repair",
        2,
    )


def test_classify_failure_counts_repair_rounds():
    output = (
        "swarm-agent: tests failed (exit 1); repair 1/2 - re-diagnosing "
        "from 1 structured failure(s)\n"
        "swarm-agent: repair attempt 2: the model returned no usable diff; giving up\n"
    )
    assert fix_harness.classify_fix_output(output, verified=False) == ("failed", 2)


def test_classify_failure_with_no_repair_is_one_attempt():
    assert fix_harness.classify_fix_output(
        "swarm-agent: the swarm could not fix this from the code shown.\n",
        verified=False,
    ) == ("failed", 1)


def test_classify_claimed_success_is_failed_when_tests_still_fail():
    # The ground truth is the independent re-run, not the claim.
    assert fix_harness.classify_fix_output(
        "swarm-agent: tests pass (python -m pytest -q)\n", verified=False
    ) == ("failed", 1)


# -- table -------------------------------------------------------------------


def _result():
    return {
        "date": "2026-11-08T15:00:00",
        "repo": "off_by_one",
        "bug": "one-line slice overshoot",
        "outcome": "fixed attempt 1",
        "attempts": 1,
        "model": "qwen2.5-coder:7b",
        "machine": "Apple M5, 16.0 GB",
        "notes": "",
    }


def test_format_fix_row():
    assert fix_harness.format_fix_row(_result()) == (
        "| 2026-11-08T15:00:00 | off_by_one | one-line slice overshoot | "
        "fixed attempt 1 | 1 | qwen2.5-coder:7b | Apple M5, 16.0 GB |  |"
    )


def test_append_fix_results_creates_header_once(tmp_path):
    path = tmp_path / "FIX_RESULTS.md"
    fix_harness.append_fix_results(path, [fix_harness.format_fix_row(_result())])
    fix_harness.append_fix_results(path, [fix_harness.format_fix_row(_result())])
    text = path.read_text(encoding="utf-8")
    assert text.count("# Fix-benchmark results") == 1
    assert text.count("| 2026-11-08T15:00:00 |") == 2


def test_machine_label():
    label = fix_harness.machine_label()
    assert label
    assert "GB" in label


# -- run_repo_fix with a stubbed fix -----------------------------------------


def _repo(rel):
    return REPO_ROOT / rel


def test_run_repo_fix_first_try(monkeypatch):
    monkeypatch.setattr(fix_harness, "run_tests", _real_pytest_run_tests)

    def honest_fix(copy, crash_text, model, peers):
        assert crash_text.strip(), "the crash log should reach the fix"
        p = copy / "recent.py"
        p.write_text(p.read_text().replace("items[-n - 1:]", "items[-n:]"))
        return 0, "swarm-agent: tests pass (python -m pytest -q)\n"

    monkeypatch.setattr(fix_harness, "_run_fix", honest_fix)
    result = fix_harness.run_repo_fix(
        _repo("benchmarks/fix_repos/off_by_one"), "slice bug", "qwen2.5-coder:7b"
    )
    assert result["outcome"] == "fixed attempt 1"
    assert result["attempts"] == 1
    assert result["notes"] == ""


def test_run_repo_fix_claimed_success_is_failed(monkeypatch):
    monkeypatch.setattr(fix_harness, "run_tests", _real_pytest_run_tests)

    def liar_fix(copy, crash_text, model, peers):
        return 0, "swarm-agent: tests pass (python -m pytest -q)\n"

    monkeypatch.setattr(fix_harness, "_run_fix", liar_fix)
    result = fix_harness.run_repo_fix(
        _repo("benchmarks/fix_repos/off_by_one"), "slice bug", "qwen2.5-coder:7b"
    )
    assert result["outcome"] == "failed"
    assert "reported success" in result["notes"]


def test_run_repo_fix_after_repair(monkeypatch):
    monkeypatch.setattr(fix_harness, "run_tests", _real_pytest_run_tests)

    def repair_fix(copy, crash_text, model, peers):
        p = copy / "recent.py"
        p.write_text(p.read_text().replace("items[-n - 1:]", "items[-n:]"))
        return 0, (
            "swarm-agent: tests failed (exit 1); repair 1/2 - re-diagnosing "
            "from 2 structured failure(s)\n"
            "swarm-agent: tests pass after repair - fixed on attempt 2 "
            "(python -m pytest -q)\n"
        )

    monkeypatch.setattr(fix_harness, "_run_fix", repair_fix)
    result = fix_harness.run_repo_fix(
        _repo("benchmarks/fix_repos/off_by_one"), "slice bug", "qwen2.5-coder:7b"
    )
    assert result["outcome"] == "fixed after repair"
    assert result["attempts"] == 2


def test_run_repo_fix_gives_up(monkeypatch):
    monkeypatch.setattr(fix_harness, "run_tests", _real_pytest_run_tests)

    def giving_up_fix(copy, crash_text, model, peers):
        return 1, (
            "swarm-agent: tests failed (exit 1); repair 1/2 - re-diagnosing "
            "from 2 structured failure(s)\n"
            "swarm-agent: repair attempt 2: the model returned no usable diff; giving up\n"
        )

    monkeypatch.setattr(fix_harness, "_run_fix", giving_up_fix)
    result = fix_harness.run_repo_fix(
        _repo("benchmarks/fix_repos/off_by_one"), "slice bug", "qwen2.5-coder:7b"
    )
    assert result["outcome"] == "failed"
    assert result["attempts"] == 2
    assert result["notes"] == "fix exit 1"


def test_run_repo_fix_baseline_passing_is_recorded(monkeypatch):
    monkeypatch.setattr(fix_harness, "run_tests", lambda cmd, path: (0, "3 passed"))
    monkeypatch.setattr(fix_harness, "_run_fix", lambda *a: (0, ""))
    result = fix_harness.run_repo_fix(
        _repo("benchmarks/fix_repos/off_by_one"), "slice bug", "qwen2.5-coder:7b"
    )
    assert result["outcome"] == "failed"
    assert result["attempts"] == 0
    assert "not broken" in result["notes"]


def test_run_fix_benchmark_empty_runs_nothing(tmp_path):
    results = tmp_path / "FIX_RESULTS.md"
    assert fix_harness.run_fix_benchmark([], "qwen2.5-coder:7b", results_path=results) == []
    assert not results.exists()


# -- the five repos are pinned -------------------------------------------------


def test_five_repos_exist_and_fail_baseline():
    assert len(fix_harness.FIX_REPOS) == 5
    for rel_path, bug in fix_harness.FIX_REPOS:
        repo = REPO_ROOT / rel_path
        assert repo.is_dir(), rel_path
        assert bug
        assert (repo / "crash.log").is_file(), rel_path
        assert (repo / "README.md").is_file(), rel_path
        code, _ = _real_pytest_run_tests("python -m pytest -q", repo)
        assert code != 0, f"{rel_path} baseline should fail"
