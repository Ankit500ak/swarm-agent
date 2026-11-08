#!/usr/bin/env python3
"""Run the 1 vs 2 vs 3 peer benchmark and append the results.

Usage, from the repo root:

    python benchmarks/bench.py --peers http://localhost:11434,http://192.168.1.5:11434

Peers can also come from the ``SWARM_PEERS`` environment variable
(comma-separated), and the model from ``--model`` (default
qwen2.5-coder:7b). Results are appended to ``benchmarks/RESULTS.md``;
see ``benchmarks/README.md`` for the methodology.
"""

import argparse
import os
import sys
from pathlib import Path

# The script lives in benchmarks/, the package lives at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm_agent.config import DEFAULT_MODEL, DEFAULT_PEER  # noqa: E402
from benchmarks import harness  # noqa: E402


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
        "--sample-repo",
        default=str(harness.SAMPLE_REPO),
        help="pinned sample repo to analyse",
    )
    parser.add_argument(
        "--results",
        default=str(harness.RESULTS_PATH),
        help="results table to append rows to",
    )
    parser.add_argument(
        "--max-peers", type=int, default=harness.MAX_PEERS, help="largest swarm to test"
    )
    args = parser.parse_args(argv)

    peer_urls = _parse_peers(args.peers)
    if not peer_urls:
        print("no peers given: pass --peers URL,... or set SWARM_PEERS", file=sys.stderr)
        return 1
    print(f"benchmark: {len(peer_urls)} peer(s), model {args.model}", file=sys.stderr)
    rows = harness.run_benchmark(
        peer_urls, args.model, args.sample_repo, args.results, args.max_peers
    )
    if not rows:
        print("nothing ran: no peers to benchmark", file=sys.stderr)
        return 1
    for row in rows:
        print(harness.format_row(row))
    print(f"results appended to {args.results}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
