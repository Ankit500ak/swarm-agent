"""Running the project's test suite after a fix.

Two jobs live here:

- :func:`run_tests` — run the suite, return ``(exit_code, tail)``.
- :func:`parse_failures` — turn that tail (or any captured test output)
  into structured records, ``{"file": str, "test": str, "message": str}``,
  one per failing test. This record shape is a **shared contract**: the
  Core lane's repair loop v2 (Phase 3, issue #16) builds against it, and
  the prompts lane's re-diagnosis contract (issue #9) consumes it. Do
  not change the shape without commenting on issue #10 first.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

TAIL_CHARS = 4000
_MAX_MESSAGE_CHARS = 500


def run_tests(
    command: str,
    cwd: str | Path,
    timeout: float = 600.0,
) -> tuple[int, str]:
    """Run ``command`` in ``cwd``; return (exit_code, tail of combined output).

    A timeout returns exit code 124 with whatever output was captured.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "")[-TAIL_CHARS:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, (f"[timed out after {timeout:.0f}s]\n" + out)[-TAIL_CHARS:]
    except OSError as exc:
        return 127, f"could not run {command!r}: {exc}"


# -- test-command detection (Issue #3) ----------------------------------------
#
# The default test command is the config default, "python -m pytest -q".
# detect_test_command() picks the right command for the target repo by
# its markers, so fix/verify work on JS/TS, Go and Rust repos too.
# The explicit --test-cmd always wins; wiring this into the agent's
# `cmd = test_cmd if test_cmd is not None else config.test_command`
# (agent.py, Core lane) is a follow-up — this lands the detector.

PYTEST_COMMAND = "python -m pytest -q"
NPM_COMMAND = "npm test"
GO_COMMAND = "go test ./..."
CARGO_COMMAND = "cargo test"


def _has_test_script(package_json: Path) -> bool:
    """True when package.json defines a non-empty "test" script.

    A missing file, unreadable content, or malformed JSON counts as
    no script — detection never raises.
    """
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return False
    test = scripts.get("test")
    return isinstance(test, str) and bool(test.strip())


def detect_test_command(path: str | Path | None, explicit: str | None = None) -> str:
    """Detect the test command for the repo at ``path``.

    ``explicit`` (from --test-cmd) always wins when given. Otherwise
    the repo is inspected for markers, in this order:

    - ``pyproject.toml`` or ``pytest.ini`` -> ``python -m pytest -q``
    - ``package.json`` with a ``"test"`` script -> ``npm test``
    - ``go.mod`` -> ``go test ./...``
    - ``Cargo.toml`` -> ``cargo test``

    A repo with no marker defaults to pytest (today's behaviour).
    Never raises: a missing or unreadable path degrades to the default.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    root = Path(path) if path else None
    if root is not None and root.is_dir():
        if (root / "pyproject.toml").is_file() or (root / "pytest.ini").is_file():
            return PYTEST_COMMAND
        if _has_test_script(root / "package.json"):
            return NPM_COMMAND
        if (root / "go.mod").is_file():
            return GO_COMMAND
        if (root / "Cargo.toml").is_file():
            return CARGO_COMMAND
    return PYTEST_COMMAND


# -- structured failure parsing (Issue #10, Phase 3) ---------------------------
#
# Record shape (frozen contract — see module docstring):
#   {"file": str, "test": str, "message": str}
# `file` is the path as the runner reported it (pytest nodeids keep their
# relative path; jest's "./" prefix is stripped). `test` is the test's
# own name (pytest keeps the full nodeid tail, e.g.
# "TestDivide::test_divide"; jest keeps the describe chain, e.g.
# "calculator › subtracts 5 - 3 to equal 1"). `message` is the first
# assertion detail line(s), capped at _MAX_MESSAGE_CHARS.
#
# Unparseable output degrades to a single record carrying the raw tail,
# never an exception — the repair loop still gets something to work
# with. Output that shows no failure at all (a passing run, empty
# output) yields [] instead: no failures, no records.

_FAILURE_SIGNALS = ("FAILED", "failed", "✕", "●", "AssertionError", "Error:")


def _looks_like_failure(output: str) -> bool:
    return any(signal in output for signal in _FAILURE_SIGNALS)


def _fallback_record(output: str) -> list[dict]:
    return [{"file": "", "test": "", "message": output.strip()[-TAIL_CHARS:]}]


def _split_nodeid(nodeid: str) -> tuple[str, str]:
    """Split a pytest nodeid into (file, test)."""
    parts = nodeid.split("::")
    return parts[0], "::".join(parts[1:]) if len(parts) > 1 else ""


def _parse_pytest(output: str) -> list[dict]:
    records: dict[tuple[str, str], dict] = {}

    # Failure sections (___ test_name ___ headers) carry the "E ..." detail
    # lines. Index them by test name so verbose FAILED lines that carry no
    # message can be enriched from their section.
    section_messages: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        if current_section is None:
            return
        detail = " ".join(
            line[2:].strip() for line in current_lines if line.startswith("E ")
        ).strip()
        if detail:
            section_messages.setdefault(current_section, detail)

    for line in output.splitlines():
        header = re.match(r"^_+\s+(.+?)\s_+$", line)
        if header:
            flush_section()
            current_section = header.group(1).strip()
            current_lines = []
            continue
        if current_section is not None:
            if line.startswith("E "):
                current_lines.append(line)
            elif line.strip() == "" and current_lines:
                # End of the detail block; keep the section open for the
                # "file:line: Exception" trailer, which adds nothing.
                pass
    flush_section()

    def message_for(test: str) -> str:
        # Section headers use dots for classes ("TestDivide.test_divide")
        # while nodeids use "::" — match on the dotted tail.
        dotted = test.replace("::", ".")
        if dotted in section_messages:
            return section_messages[dotted][: _MAX_MESSAGE_CHARS]
        short = test.split("::")[-1]
        if short in section_messages:
            return section_messages[short][: _MAX_MESSAGE_CHARS]
        return "failed"

    # Primary source: the short test summary info —
    #   FAILED tests/test_calc.py::test_add - assert 3 == 4
    #   ERROR tests/test_x.py::test_y - some setup error
    for line in output.splitlines():
        match = re.match(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$", line)
        if not match:
            continue
        nodeid, message = match.group(1), (match.group(2) or "").strip()
        file, test = _split_nodeid(nodeid)
        if not test:
            continue
        key = (file, test)
        if key not in records:
            records[key] = {
                "file": file,
                "test": test,
                "message": (message or message_for(test))[:_MAX_MESSAGE_CHARS],
            }

    # Secondary source: verbose per-test lines ("...::test_add FAILED [ 33%]").
    # These carry no message, so enrich from the failure section.
    for line in output.splitlines():
        match = re.match(r"^(\S+::\S+)\s+(?:FAILED|ERROR)\b", line)
        if not match:
            continue
        file, test = _split_nodeid(match.group(1))
        if not test:
            continue
        key = (file, test)
        if key not in records:
            records[key] = {
                "file": file,
                "test": test,
                "message": message_for(test),
            }

    return list(records.values())


def _parse_jest(output: str) -> list[dict]:
    records: dict[tuple[str, str], dict] = {}
    current_file = ""
    lines = output.splitlines()

    # Detail sections ("● test name") carry the message; the suite summary
    # ("✕ test name (5 ms)") only names the test. Two passes: ● sections
    # first (they have the message and always win), then ✕ lines only
    # fill in tests the ● sections did not cover — matching by exact name
    # or by suffix, because the summary drops the describe prefix that
    # the ● line keeps ("subtracts ..." vs "calculator › subtracts ...").
    i = 0
    while i < len(lines):
        line = lines[i]
        fail_match = re.match(r"^FAIL\s+(\S+)", line)
        if fail_match:
            current_file = fail_match.group(1)
            if current_file.startswith("./"):
                current_file = current_file[2:]
            i += 1
            continue
        bullet_match = re.match(r"^\s*●\s+(.+?)\s*$", line)
        if bullet_match and current_file:
            test = bullet_match.group(1).strip()
            # Collect the message: following lines until the next ●, the
            # "Test Suites:" summary, or the end. Skip code-frame lines
            # ("  2 | ...", "  > 2 | ...", "  at ...") — the assertion
            # detail ("Expected: ...", "Received: ...") is what matters.
            message_lines: list[str] = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                # A new suite (FAIL/PASS) or the summary ends this section.
                # Without the FAIL break, the next suite's FAIL line was
                # swallowed into this test's message and its file was lost.
                if re.match(r"^\s*●\s+", nxt) or nxt.startswith("Test Suites:") \
                        or re.match(r"^(FAIL|PASS)\s+", nxt):
                    break
                stripped = nxt.strip()
                if stripped and not re.match(r"^>? ?\d+ \|", stripped) \
                        and not stripped.startswith("|") \
                        and not stripped.startswith("at "):
                    message_lines.append(stripped)
                j += 1
            message = "\n".join(message_lines).strip()[:_MAX_MESSAGE_CHARS]
            records[(current_file, test)] = {
                "file": current_file,
                "test": test,
                "message": message or "failed",
            }
            i = j
            continue
        i += 1

    current_file = ""
    for line in lines:
        fail_match = re.match(r"^FAIL\s+(\S+)", line)
        if fail_match:
            current_file = fail_match.group(1)
            if current_file.startswith("./"):
                current_file = current_file[2:]
            continue
        cross_match = re.match(r"^\s*✕\s+(.+?)\s*(?:\(\d+\s*ms\))?\s*$", line)
        if cross_match and current_file:
            test = cross_match.group(1).strip()
            covered = any(
                file == current_file
                and (name == test or name.endswith(" › " + test))
                for file, name in records
            )
            if not covered:
                records[(current_file, test)] = {
                    "file": current_file,
                    "test": test,
                    "message": "failed",
                }

    return list(records.values())


def parse_failures(output: str, ecosystem: str) -> list[dict]:
    """Parse test output into ``{"file", "test", "message"}`` records.

    ``ecosystem`` is ``"pytest"`` (also accepts ``"python"``) or
    ``"jest"`` (also accepts ``"npm"`` / ``"node"``). An unknown
    ecosystem, or output in no recognisable shape, never raises:
    output that looks like a failure degrades to a single record
    carrying the raw tail, and output with no failure in it yields [].
    """
    if not output or not output.strip():
        return []
    eco = (ecosystem or "").strip().lower()
    if eco in ("pytest", "python"):
        records = _parse_pytest(output)
    elif eco in ("jest", "npm", "node"):
        records = _parse_jest(output)
    else:
        records = []
    if records:
        return records
    if _looks_like_failure(output):
        return _fallback_record(output)
    return []
