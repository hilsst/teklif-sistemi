"""Multi-item quote: several parts plus one-off (NRE) items under one quote number.

A part item is an ordinary ``QuoteInput`` priced by ``compute_quote``. An NRE item is a
one-off amount entered by the user (tool, master model, fixture). The set adds the part
order totals and the NRE items, then applies the library's minimum order value once, on the
quote total. Every set-level number is a ``CostLine`` with formula, inputs and sources.

Tool and master model costs are still entered by hand; they become computed in Phase 3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from teklif.common.models import (
    CostInput,
    CostLine,
    Override,
    Price,
    ReviewFlag,
    Source,
    derived,
)
from teklif.costing.engine import (
    QuoteInputError,
    QuoteResult,
    _line,
    _Lines,
    compute_quote,
    fx_param,
    min_order_topup_line,
)
from teklif.costing.inputs import QuoteInput, _Strict
from teklif.library.loader import Library
from teklif.library.schema import NreType


class NreItem(_Strict):
    id: str
    name: str
    nre_type: NreType
    amount: Price
    note: str = ""


class QuoteSetInput(_Strict):
    quote_no: str
    customer: str = ""
    quote_currency: str | None = None
    blueprint: str | None = Field(None, description="Blueprint id the quote was started from")
    parts: list[QuoteInput] = Field(default_factory=list)
    nre_items: list[NreItem] = Field(default_factory=list)
    overrides: dict[str, Override] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parts_inherit_quote_no(cls, data: Any) -> Any:
        """Parts in a set share the set's quote number, so the file need not repeat it."""
        if isinstance(data, dict) and data.get("quote_no"):
            parts = [
                {"quote_no": data["quote_no"], **p} if isinstance(p, dict) else p
                for p in data.get("parts") or []
            ]
            return {**data, "parts": parts}
        return data

    @model_validator(mode="after")
    def _check(self) -> QuoteSetInput:
        if not self.parts and not self.nre_items:
            raise ValueError("A quote needs at least one part or NRE item")
        ids = [p.part_no for p in self.parts] + [n.id for n in self.nre_items]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"Duplicate item ids in the quote: {dup}")
        return self


class QuoteSetResult(BaseModel):
    input: QuoteSetInput
    currency: str
    parts: list[QuoteResult]
    lines: list[CostLine]

    def get(self, key: str) -> CostLine:
        for line in self.lines:
            if line.key == key:
                return line
        raise KeyError(key)

    def value(self, key: str) -> float | None:
        return self.get(key).final

    @property
    def review_items(self) -> list[tuple[str, ReviewFlag]]:
        """Every flag with its line; part flags are prefixed with the part number."""
        out = [(ln.key, fl) for ln in self.lines for fl in ln.flags]
        for part in self.parts:
            out += [(f"{part.input.part_no}/{k}", fl) for k, fl in part.review_items]
        return out

    @property
    def manual_review(self) -> bool:
        return bool(self.review_items)

    @property
    def open_review_items(self) -> list[tuple[str, ReviewFlag]]:
        out = [(ln.key, fl) for ln in self.lines for fl in ln.open_flags]
        for part in self.parts:
            out += [(f"{part.input.part_no}/{k}", fl) for k, fl in part.open_review_items]
        return out

    @property
    def has_open_review(self) -> bool:
        return bool(self.open_review_items)


def _named_sum(
    key: str, description: str, items: list[tuple[str, float | None]], unit: str
) -> CostLine:
    """Sum of named amounts; None when any is missing (reason stays on the source line)."""
    inputs = [CostInput(name=n, value=v, unit=unit, source=Source.DERIVED) for n, v in items]
    missing = [i.name for i in inputs if i.value is None]
    result: float | None = None
    flags: list[ReviewFlag] = []
    if missing:
        flags.append(ReviewFlag(code="missing_upstream", reason="Missing: " + ", ".join(missing)))
    else:
        result = sum(i.value for i in inputs if i.value is not None)
    return CostLine(
        key=key,
        description=description,
        formula="Σ " + (", ".join(n for n, _ in items) if items else "∅"),
        inputs=inputs,
        result=result,
        unit=unit,
        category="total",
        flags=flags,
    )


def compute_quote_set(inp: QuoteSetInput, lib: Library) -> QuoteSetResult:
    ccy = inp.quote_currency or lib.currency.quote_currency
    if ccy not in lib.currency.rates:
        raise QuoteInputError(f"Unknown quote currency '{ccy}'")
    out = _Lines(inp.overrides)

    def fx(from_ccy: str) -> tuple[Any, str | None]:
        return fx_param(lib, from_ccy, ccy)

    parts = [
        compute_quote(
            p.model_copy(
                update={
                    "quote_no": inp.quote_no,
                    "customer": p.customer or inp.customer,
                    "quote_currency": ccy,
                }
            ),
            lib,
            apply_min_order=False,
        )
        for p in inp.parts
    ]

    nre_lines: list[CostLine] = []
    for item in inp.nre_items:
        amount = item.amount.model_copy(update={"source": Source.USER})
        nre_lines.append(
            out.add(
                _line(
                    f"nre.{item.id}",
                    f"{item.name} ({item.nre_type})",
                    "amount × fx",
                    {"amount": amount, "fx": fx(amount.currency)},
                    lambda v: v["amount"] * v["fx"],
                    ccy,
                    "nre",
                )
            )
        )

    parts_total = out.add(
        _named_sum(
            "set.parts_total",
            "Parça kalemleri toplamı",
            [(f"{r.input.part_no}:price.order_total", r.value("price.order_total")) for r in parts],
            ccy,
        )
    )
    nre_total = out.add(
        _named_sum(
            "set.nre_total",
            "NRE kalemleri toplamı",
            [(ln.key, ln.final) for ln in nre_lines],
            ccy,
        )
    )
    minimum = lib.thresholds.costing.min_order_value.value
    use_min = minimum is None or minimum > 0  # null = TODO, never silently "off"
    raw = out.add(
        _line(
            "set.order_raw" if use_min else "set.order_total",
            "Teklif toplamı (minimum öncesi)" if use_min else "Teklif toplamı",
            "set.parts_total + set.nre_total",
            {"parts": derived(parts_total.final, ccy), "nre": derived(nre_total.final, ccy)},
            lambda v: v["parts"] + v["nre"],
            ccy,
            "price",
        )
    )
    if use_min:
        topup = out.add(min_order_topup_line("set.min_order_topup", raw, lib, fx))
        out.add(
            _line(
                "set.order_total",
                "Teklif toplamı",
                "set.order_raw + set.min_order_topup",
                {"raw": derived(raw.final, ccy), "topup": derived(topup.final, ccy)},
                lambda v: v["raw"] + v["topup"],
                ccy,
                "price",
            )
        )
    unused = out.unused_overrides()
    if unused:
        raise QuoteInputError(f"Overrides for unknown lines: {unused}")
    return QuoteSetResult(input=inp, currency=ccy, parts=parts, lines=out.lines)


def blueprint_skeleton(lib: Library, blueprint_id: str, quote_no: str) -> dict[str, Any]:
    """A quote-set input dict for the blueprint, ready to be completed by the user.

    Part items carry only structural fields (id, name, ``tooling_exists``); everything that
    needs real data (plies, geometry, quantity, amounts) is left for the user. The result
    is a starting point and does not validate until it is completed.
    """
    bps = {b.id: b for b in lib.blueprints.blueprints}
    if blueprint_id not in bps:
        raise QuoteInputError(f"Unknown blueprint '{blueprint_id}'")
    bp = bps[blueprint_id]
    parts: list[dict[str, Any]] = []
    nres: list[dict[str, Any]] = []
    for item in bp.items:
        if item.type == "part":
            parts.append(
                {
                    "part_no": item.id,
                    "part_name": item.name,
                    "process": next(iter(lib.processes)),
                    "material": None,
                    "cure_cycle": None,
                    "complexity": None,
                    "plies": [],
                    "geometry": None,
                    "quantity": None,
                    "batch_size": None,
                    "tooling_exists": item.tooling_exists,
                }
            )
        else:
            nres.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "nre_type": item.nre_type,
                    "amount": {
                        "value": None,
                        "currency": lib.currency.quote_currency,
                        "per": "pcs",
                    },
                }
            )
    return {
        "quote_no": quote_no,
        "customer": "",
        "blueprint": bp.id,
        "parts": parts,
        "nre_items": nres,
    }


__all__ = [
    "NreItem",
    "QuoteSetInput",
    "QuoteSetResult",
    "blueprint_skeleton",
    "compute_quote_set",
]
