"""Tiny gradebook summary — the demo target for `swarm-agent fix`.

Reads ``data.json`` and prints the average score, the best score and
how many scores there are. It is intentionally broken: see
``crash.log`` and the README.
"""

import json
from pathlib import Path


def summarize(path="data.json"):
    """Summarise the scores in ``path`` as a dict."""
    data = json.loads(Path(path).read_text())
    scores = data["score"]
    return {
        "average": sum(scores) / len(scores),
        "best": max(scores),
        "count": len(scores),
    }


def main():
    print(json.dumps(summarize()))


if __name__ == "__main__":
    main()
