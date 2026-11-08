"""Tests for ``average``.

They fail while the bug is present and pass once it is fixed.
"""

from stats import average


def test_average():
    assert average([1, 2, 3]) == 2


def test_empty_returns_zero():
    assert average([]) == 0.0
