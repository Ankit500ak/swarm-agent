"""Tests for the gradebook summary.

They fail while the demo is broken (``summarize`` raises before it
returns anything) and pass once the fix lands.
"""

from pathlib import Path

from demo import summarize

DATA = Path(__file__).resolve().parent.parent / "data.json"


def test_average():
    assert summarize(DATA)["average"] == 87.6


def test_best():
    assert summarize(DATA)["best"] == 95


def test_count():
    assert summarize(DATA)["count"] == 5
