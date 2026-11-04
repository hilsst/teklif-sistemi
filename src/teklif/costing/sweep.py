"""Quantity sweep: how unit cost, unit price and order total move with order quantity.

Each row is a full ``compute_quote`` run with the quantity replaced, so the learning curve,
setup splitting and NRE all follow the normal rules. The batch size is capped at the
quantity (a batch cannot be larger than the order). Overrides stay as entered.
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

    @property
    def effective_unit_price(self) -> float | None:
        """Order total (NRE and minimum order included) divided by quantity."""
        if self.order_total is None:
            return None
        return self.order_total / self.quantity


def quantity_sweep(
    inp: QuoteInput, lib: Library, quantities: Sequence[int] | None = None
) -> list[SweepRow]:
    qtys = sorted(
        set(quantities if quantities is not None else lib.thresholds.costing.sweep_quantities)
    )
    rows: list[SweepRow] = []
    for qty in qtys:
        if qty < 1:
            raise ValueError(f"Quantity must be >= 1, got {qty}")
        batch = min(inp.batch_size, qty)
        result: QuoteResult = compute_quote(
            inp.model_copy(update={"quantity": qty, "batch_size": batch}), lib
        )
        rows.append(
            SweepRow(
                quantity=qty,
                batch_size=batch,
                unit_cost=result.value("cost.unit"),
                unit_price=result.value("price.unit_final"),
                order_total=result.value("price.order_total"),
                manual_review=result.has_open_review,
            )
        )
    return rows
