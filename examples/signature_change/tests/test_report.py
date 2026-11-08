"""Monthly totals, converted to USD."""

import pytest

from ledger.report import monthly_total


def test_monthly_total_converts_eur():
    # 100 EUR at 1.10 -> 110 USD.
    entries = [{"amount": 100.0, "currency": "EUR"}]
    assert monthly_total(entries) == pytest.approx(110.0)


def test_monthly_total_mixed():
    # 100 EUR -> 110 USD, plus 50 USD -> 160 USD.
    entries = [
        {"amount": 100.0, "currency": "EUR"},
        {"amount": 50.0, "currency": "USD"},
    ]
    assert monthly_total(entries) == pytest.approx(160.0)
