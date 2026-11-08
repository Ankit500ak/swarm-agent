"""Running the project's test suite after a fix."""

from __future__ import annotations

import subprocess
from pathlib import Path

TAIL_CHARS = 4000


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
