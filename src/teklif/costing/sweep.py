"""Quantity sweep: how unit cost, unit price and order total move with order quantity.

Each row is a full ``compute_quote`` run with the quantity replaced, so the learning curve,
setup splitting and NRE all follow the normal rules. The batch size is capped at the
quantity (a batch cannot be larger than the order).

Overrides: those on order-level totals (and the unit price when NRE is amortized) are fixed
amounts that would be wrong at another quantity, so they are dropped and listed per row in
``dropped_overrides``. Other overrides are kept and listed in ``kept_overrides``; they are held
constant across quantities, which the caller should show.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from teklif.costing.engine import QuoteResult, compute_quote
from teklif.costing.inputs import QuoteInput
from teklif.library.loader import Library


@dataclass(frozen=True)
class SweepRow:
    quantity: int
    batch_size: int
    unit_cost: float | None
    unit_price: float | None
    order_total: float | None
    manual_review: bool
    dropped_overrides: tuple[str, ...] = ()
    kept_overrides: tuple[str, ...] = ()

    @property
    def effective_unit_price(self) -> float | None:
        """Order total (NRE and minimum order included) divided by quantity."""
        if self.order_total is None:
            return None
        return self.order_total / self.quantity


def quantity_sweep(
    inp: QuoteInput, lib: Library, quantities: Sequence[int] | None = None
) -> list[SweepRow]:
    order_level = {"price.order_total", "price.order_raw", "price.min_order_topup"}
    if inp.amortize_nre:
        order_level.add("price.unit_final")
    dropped = tuple(sorted(k for k in inp.overrides if k in order_level))
    kept = tuple(sorted(k for k in inp.overrides if k not in order_level))
    base_overrides = {k: v for k, v in inp.overrides.items() if k not in order_level}
    qtys = sorted(
        set(quantities if quantities is not None else lib.thresholds.costing.sweep_quantities)
    )
    rows: list[SweepRow] = []
    for qty in qtys:
        if qty < 1:
            raise ValueError(f"Quantity must be >= 1, got {qty}")
        batch = min(inp.batch_size, qty)
        result: QuoteResult = compute_quote(
            inp.model_copy(
                update={"quantity": qty, "batch_size": batch, "overrides": base_overrides}
            ),
            lib,
        )
        rows.append(
            SweepRow(
                quantity=qty,
                batch_size=batch,
                unit_cost=result.value("cost.unit"),
                unit_price=result.value("price.unit_final"),
                order_total=result.value("price.order_total"),
                manual_review=result.has_open_review,
                dropped_overrides=dropped,
                kept_overrides=kept,
            )
        )
    return rows
