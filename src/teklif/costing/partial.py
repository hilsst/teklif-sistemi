"""Partial price: a library copy where TODO values are replaced by neutral numbers.

A normal quote has no price while any factor is still TODO. For a rough idea Kemal wants to see
the price the entered factors give. The result is never a quote: the caller must label it
partial and list the filled-in values.
"""

from __future__ import annotations

import copy

from teklif.library import Library
from teklif.library.loader import iter_values

# Neutral value when a TODO is a divisor / multiplier; everything else counts as 0.
_NEUTRAL_ONE = (
    "load_efficiency",
    "efficiency",
    "operators",
    "learning_curve_ratio",
    "learning_reference_unit",
    "rates.",
    "nest_efficiency",
)


# Autoclave capability limits: unknown means "does not block the quote".
_NO_LIMIT = (
    "inner_diameter",
    "usable_length",
    "usable_width",
    "max_temp",
    "max_pressure",
    "vacuum_ports",
    "thermocouples",
)


def fill_missing(lib: Library) -> tuple[Library, list[str]]:
    """Return (library copy with TODO values filled, list of ``file:path`` that were TODO)."""
    out = copy.deepcopy(lib)
    filled: list[str] = []
    for fname, model in out.files().items():
        for path, v in iter_values(model):
            if v.value is None and path.rsplit(".", 1)[-1] != "nesting_scrap_ratio":
                neutral = any(k in path for k in _NEUTRAL_ONE)
                last = path.rsplit(".", 1)[-1]
                unlimited = (fname == "autoclaves.yaml" and last in _NO_LIMIT) or (
                    last == "debulk_interval_plies"
                )
                v.value = (
                    1e6
                    if unlimited
                    else (1000.0 if last == "roll_width" else (1.0 if neutral else 0.0))
                )
                note = " (sınırsız)" if unlimited else (" (nötr 1)" if neutral else " (0)")
                filled.append(f"{fname}: {path}{note}")
    return out, filled


__all__ = ["fill_missing"]
