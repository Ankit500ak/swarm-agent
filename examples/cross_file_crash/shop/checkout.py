"""Checkout: total a basket with a 10% member discount."""

from .catalog import get_price


def checkout_total(items):
    """Return the basket total in paise, after the discount.

    ``items`` is a list of ``(sku, qty)`` pairs.
    """
    total = 0
    for sku, qty in items:
        price = get_price(sku)
        total += price * qty * 0.9
    return total
