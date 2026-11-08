"""Tests for the sample workload's calculator.

Part of the pinned benchmark workload (see ``benchmarks/README.md``).
These tests are analysed like any other source file; they are not
part of the repo's own suite (``pyproject.toml`` sets
``testpaths = ["tests"]``).
"""

from calc import add, divide, mean


def test_add():
    assert add(2, 3) == 5


def test_divide_by_zero_returns_none():
    assert divide(10, 0) is None


def test_mean():
    assert mean([1, 2, 3]) == 2
