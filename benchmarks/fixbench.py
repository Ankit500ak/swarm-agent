#!/usr/bin/env python3
"""Run the fix benchmark and append the results.

Usage, from the repo root, with an Ollama peer up and serving the
model:

    python benchmarks/fixbench.py --peers http://localhost:11434

Each of the five seeded-bug repos is copied to a scratch directory,
its tests are run to capture the failure, the failure is piped to
``swarm-agent fix --apply``, and the tests are re-run independently.
Results are appended to ``benchmarks/FIX_RESULTS.md``; see
``benchmarks/README.md`` for the methodology.
"""

import argparse
import os
import sys
from pathlib import Path

# The script lives in benchmarks/, the package lives at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm_agent.config import DEFAULT_MODEL, DEFAULT_PEER  # noqa: E402
from benchmarks import fix_harness  # noqa: E402


def _parse_peers(raw: str | None) -> list[str]:
    value = raw or os.environ.get("SWARM_PEERS") or DEFAULT_PEER
    return [p.strip() for p in value.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peers",
        default=None,
        help="comma-separated Ollama peer URLs "
        "(default: SWARM_PEERS, else http://localhost:11434)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model to request")
    parser.add_argument(
        "--repos",
        default=None,
        help="comma-separated repo names to run (default: all five)",
    )
    parser.add_argument(
        "--results",
        default=str(fix_harness.RESULTS_PATH),
        help="results table to append rows to",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="parent directory for the scratch copies (default: system temp)",
    )
    args = parser.parse_args(argv)

    peers = _parse_peers(args.peers)
    repos = list(fix_harness.FIX_REPOS)
    if args.repos:
        wanted = {name.strip() for name in args.repos.split(",") if name.strip()}
        repos = [r for r in repos if Path(r[0]).name in wanted]
        if not repos:
            print(f"no matching repos: {args.repos}", file=sys.stderr)
            return 1
    print(
        f"fix benchmark: {len(repos)} repo(s), model {args.model}, "
        f"peers {', '.join(peers)}",
        file=sys.stderr,
    )
    rows = fix_harness.run_fix_benchmark(repos, args.model, peers, args.results, args.work_dir)
    if not rows:
        print("nothing ran", file=sys.stderr)
        return 1
    for row in rows:
        print(fix_harness.format_fix_row(row))
    print(f"results appended to {args.results}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
