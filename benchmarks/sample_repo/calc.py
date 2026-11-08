"""Small arithmetic helpers with one deliberate edge case each.

This file is part of the pinned benchmark workload (see
``benchmarks/README.md``): it is analysed, never executed, so the
helpers only need to look like a real module.
"""


def add(a, b):
    """Return a + b."""
    return a + b


def subtract(a, b):
    """Return a - b."""
    return a - b


def multiply(a, b):
    """Return a * b, computed by repeated addition."""
    total = 0
    for _ in range(abs(b)):
        total += a
    return total if b >= 0 else -total


def divide(a, b):
    """Return a / b, or None when b is zero instead of raising."""
    if b == 0:
        return None
    return a / b


def mean(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    if not values:
        raise ValueError("mean of an empty list is undefined")
    return sum(values) / len(values)
