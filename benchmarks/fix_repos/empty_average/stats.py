"""Average of a list of numbers, with an empty-list guard missing.

Part of the fix-benchmark workload (see ``benchmarks/README.md``):
one seeded bug, fixed by the swarm.
"""


def average(values):
    """Return the mean of ``values``; 0.0 when the list is empty."""
    return sum(values) / len(values)
