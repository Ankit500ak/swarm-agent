"""Tests for ``recent_items``.

They fail while the bug is present and pass once it is fixed.
"""

from recent import recent_items


def test_last_three():
    assert recent_items([1, 2, 3, 4, 5], 3) == [3, 4, 5]


def test_last_one():
    assert recent_items([1, 2, 3], 1) == [3]


def test_all():
    assert recent_items([1, 2], 2) == [1, 2]
