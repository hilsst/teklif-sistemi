"""Pure cost formulas. No I/O, no library access, no defaults: numbers in, number out.

The engine wraps each call in a ``CostLine`` so that its inputs, their sources and
the formula text are stored with the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def learning_exponent(lc: float) -> float:
    """b = log2(LC); LC = 0.85 means each doubling of quantity takes 85 % of the time."""
    if not 0 < lc <= 1:
        raise ValueError(f"learning curve ratio must be in (0, 1], got {lc}")
    return math.log2(lc)


def learning_factor(lc: float, reference_unit: float, first_unit: float, quantity: float) -> float:
    """Average of (n / ref)^b over units n = first .. first + quantity - 1.

    Multiplying a standard time valid at unit ``ref`` by this factor gives the average
    time per unit over the order (T_n = T_ref × (n/ref)^b, the user's T_1 formula
    when ref = 1).
    """
    if reference_unit < 1 or first_unit < 1 or quantity < 1:
        raise ValueError("reference unit, first unit and quantity must be >= 1")
    b = learning_exponent(lc)
    first, qty = int(first_unit), int(quantity)
    return math.fsum((n / reference_unit) ** b for n in range(first, first + qty)) / qty


def operation_time(
    setup_per_batch: float,
    batch_size: float,
    run_terms: Sequence[tuple[float, float]],
    complexity_factor: float,
    learning: float,
    efficiency: float,
) -> float:
    """Paid elapsed minutes per part.

    t = (setup / batch + Σ(rate × driver) × complexity × learning) / efficiency
    """
    if batch_size <= 0:
        raise ValueError("batch size must be > 0")
    if not 0 < efficiency <= 1.5:
        raise ValueError(f"efficiency must be in (0, 1.5], got {efficiency}")
    run = math.fsum(rate * driver for rate, driver in run_terms)
    return (setup_per_batch / batch_size + run * complexity_factor * learning) / efficiency


def labor_cost(time_min: float, operators: float, hourly_rate: float, fx: float) -> float:
    """time / 60 × operators × labor rate × fx"""
    return time_min / 60.0 * operators * hourly_rate * fx


def machine_cost(time_min: float, hourly_rate: float, fx: float) -> float:
    """time / 60 × machine rate × fx"""
    return time_min / 60.0 * hourly_rate * fx


def ply_material_cost(
    area_m2: float, count: float, scrap_ratio: float, price: float, fx: float
) -> float:
    """area × count × (1 + scrap) × price × fx"""
    return area_m2 * count * (1.0 + scrap_ratio) * price * fx


def consumable_cost(
    driver: float, qty_per_driver: float, waste_ratio: float, price: float, fx: float
) -> float:
    """driver × qty × (1 + waste) × price × fx"""
    return driver * qty_per_driver * (1.0 + waste_ratio) * price * fx


def fits_in_autoclave(
    x_mm: float, y_mm: float, spacing_mm: float, length_mm: float, width_mm: float
) -> bool:
    """True if the tool plus spacing fits the usable floor in either orientation."""
    a, b = x_mm + 2 * spacing_mm, y_mm + 2 * spacing_mm
    return (a <= length_mm and b <= width_mm) or (b <= length_mm and a <= width_mm)


def autoclave_share(
    x_mm: float,
    y_mm: float,
    spacing_mm: float,
    length_mm: float,
    width_mm: float,
    load_efficiency: float,
) -> float:
    """Share of one cure cycle taken by one tool.

    (X + 2s)(Y + 2s) / (L × W × load efficiency)
    """
    if load_efficiency <= 0:
        raise ValueError("load efficiency must be > 0")
    footprint = (x_mm + 2 * spacing_mm) * (y_mm + 2 * spacing_mm)
    return footprint / (length_mm * width_mm * load_efficiency)


def autoclave_cost(cycle_time_h: float, hourly_rate: float, fx: float, share: float) -> float:
    """cycle time × hourly rate × fx × share"""
    return cycle_time_h * hourly_rate * fx * share


def overhead(basis_amount: float, ratio: float) -> float:
    """basis × overhead ratio"""
    return basis_amount * ratio


def price_from_cost(cost: float, ratio: float, basis: str) -> float:
    """markup: cost × (1 + r); margin: cost / (1 - r)."""
    if basis == "markup":
        return cost * (1.0 + ratio)
    if basis == "margin":
        if ratio >= 1:
            raise ValueError("margin ratio must be < 1")
        return cost / (1.0 - ratio)
    raise ValueError(f"unknown margin basis '{basis}'")
