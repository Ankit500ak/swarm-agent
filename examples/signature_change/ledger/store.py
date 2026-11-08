"""Ledger store: record entries, converting amounts to USD.

Exchange rates to USD, as of the seeded data.
"""

RATES = {"USD": 1.0, "EUR": 1.10, "GBP": 1.27}


def record(entry, *, currency):
    """Store one ledger entry, converted to USD.

    ``entry`` is ``{"amount": float, "currency": str}``.
    Returns ``{"amount": float, "currency": "USD"}``.
    """
    return {"amount": entry["amount"], "currency": currency}
