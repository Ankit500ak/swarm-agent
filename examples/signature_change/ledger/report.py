"""Monthly reporting over ledger entries."""

from .store import record


def monthly_total(entries):
    """Total of all entries, in USD."""
    total = 0.0
    for entry in entries:
        total += record(entry)["amount"]
    return total
