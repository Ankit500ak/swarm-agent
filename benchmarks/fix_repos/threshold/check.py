"""Age check with a boundary bug.

Part of the fix-benchmark workload (see ``benchmarks/README.md``):
one seeded bug, fixed by the swarm.
"""


def is_adult(age):
    """Return True when ``age`` is 18 or older."""
    return age > 18
