"""Command-line entry point for the sample workload.

Part of the pinned benchmark workload (see ``benchmarks/README.md``).
It wires the helpers in ``calc`` and ``store`` together; it is never
executed by the benchmark, only analysed.
"""

import argparse
import json
import sys

from calc import add, divide, mean
from store import Store


def build_parser():
    """Build the argument parser for the sample CLI."""
    parser = argparse.ArgumentParser(prog="sample")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("add", help="add two numbers")
    sub.add_parser("mean", help="mean of a list of numbers")
    return parser


def main(argv=None):
    """Parse arguments, run the command, print the result as JSON."""
    parser = build_parser()
    args = parser.parse_args(argv)
    store = Store()
    if args.command == "add":
        result = add(2, 3)
    else:
        result = mean([1, 2, 3, 4])
    store.put("last", result)
    print(json.dumps({"result": store.get("last")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
