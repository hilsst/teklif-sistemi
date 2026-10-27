"""Unit vocabulary.

Units are plain strings such as ``min/m2`` or ``g/m2``. Only known base tokens are
accepted so that a typo in a library file fails at load time instead of silently
producing a wrong number. Conversions are not performed here; formulas expect the
units declared in the library schema.
"""

from __future__ import annotations

BASE_UNITS: frozenset[str] = frozenset(
    {
        # length / area / volume
        "mm",
        "m",
        "m2",
        "m3",
        "L",
        # mass
        "g",
        "kg",
        # time
        "s",
        "min",
        "h",
        "day",
        # temperature / pressure / angle
        "degC",
        "K",
        "bar",
        "deg",
        # counts and dimensionless
        "ratio",
        "pcs",
        "part",
        "ply",
        "hole",
        "debulk",
        "batch",
        "cycle",
        "person",
        # scaled
        "1e-6",
    }
)


def unit_tokens(unit: str) -> list[str]:
    return [t.strip() for t in unit.split("/")]


def is_known_unit(unit: str, currencies: frozenset[str] = frozenset()) -> bool:
    """True if every ``/``-separated token is a known base unit or currency code."""
    tokens = unit_tokens(unit)
    return all(t in BASE_UNITS or t in currencies for t in tokens) and all(tokens)
