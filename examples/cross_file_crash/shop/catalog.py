"""Product catalogue.

Prices are stored in ``PRICE_TABLE`` and returned by :func:`get_price`.
The contract is: prices are integer paise.
"""

PRICE_TABLE = {
    "pen": "499",
    "notebook": "1299",
    "backpack": "2499",
}


def get_price(sku: str):
    """Return the price of a SKU, in paise."""
    return PRICE_TABLE[sku]
