"""Currency conversion. Rates are read from ``library/currency.yaml``."""

from __future__ import annotations

from collections.abc import Mapping

from teklif.common.models import Param


class MissingRateError(ValueError):
    """Raised when a conversion needs a rate that is still TODO."""


def convert(amount: float, from_ccy: str, to_ccy: str, rates: Mapping[str, Param]) -> float:
    """Convert ``amount`` between currencies.

    ``rates[c].value`` is how many units of ``c`` one unit of the base currency buys,
    so ``amount_base = amount / rate[from]`` and ``result = amount_base * rate[to]``.
    """
    if from_ccy == to_ccy:
        return amount
    for ccy in (from_ccy, to_ccy):
        if ccy not in rates:
            raise MissingRateError(f"Unknown currency '{ccy}'.")
        if rates[ccy].value is None:
            raise MissingRateError(f"Exchange rate for '{ccy}' is TODO.")
    rate_from = rates[from_ccy].value
    rate_to = rates[to_ccy].value
    assert rate_from is not None and rate_to is not None
    return amount / rate_from * rate_to
