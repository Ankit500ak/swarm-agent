"""Return the most recent items from a list.

Part of the fix-benchmark workload (see ``benchmarks/README.md``):
one seeded bug, fixed by the swarm.
"""


def recent_items(items, n):
    """Return the last ``n`` items of ``items``, in order."""
    return items[-n - 1:]
