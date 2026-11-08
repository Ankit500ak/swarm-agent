"""Basket totals, with the 10% member discount applied."""

import pytest

from shop.checkout import checkout_total


def test_total_applies_discount():
    # 499 paise x 2, less 10%.
    assert checkout_total([("pen", 2)]) == pytest.approx(898.2)


def test_total_single_item():
    # 1299 paise x 1, less 10%.
    assert checkout_total([("notebook", 1)]) == pytest.approx(1169.1)
