"""CSV export of quote breakdowns and quantity sweeps.

Defaults suit Turkish Excel: ``;`` separator, decimal comma, UTF-8 with BOM. Pass
``decimal="."`` for machine consumption.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from teklif.common.models import CostLine
from teklif.costing.engine import QuoteResult
from teklif.costing.quote_set import QuoteSetResult
from teklif.costing.sweep import SweepRow
from teklif.reporting.quote_report import flags_text, inputs_text

BOM = "﻿"


def _num(value: float | None, decimal: str) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", decimal) or "0"


def _safe(text: str) -> str:
    """Neutralise spreadsheet formulas in free text (a leading = + - @ would be executed)."""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _write(rows: Sequence[Sequence[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, delimiter=";", lineterminator="\r\n").writerows(rows)
    return BOM + buf.getvalue()


BREAKDOWN_HEADER = [
    "kalem",
    "kategori",
    "açıklama",
    "formül",
    "girdiler [kaynak]",
    "hesaplanan",
    "ezilen",
    "ezme gerekçesi",
    "sonuç",
    "birim",
    "inceleme",
]


def _line_row(prefix: str, ln: CostLine, decimal: str) -> list[str]:
    return [
        prefix + ln.key,
        ln.category,
        _safe(ln.description),
        _safe(ln.formula),
        _safe(inputs_text(ln)),
        _num(ln.result, decimal),
        _num(ln.override.value, decimal) if ln.override else "",
        _safe(ln.override.reason) if ln.override else "",
        _num(ln.final, decimal),
        ln.unit,
        _safe(flags_text(ln)),
    ]


def breakdown_csv(result: QuoteResult, decimal: str = ",") -> str:
    rows = [BREAKDOWN_HEADER] + [_line_row("", ln, decimal) for ln in result.lines]
    return _write(rows)


def set_breakdown_csv(result: QuoteSetResult, decimal: str = ",") -> str:
    rows = [BREAKDOWN_HEADER] + [_line_row("", ln, decimal) for ln in result.lines]
    for part in result.parts:
        rows += [_line_row(f"{part.input.part_no}/", ln, decimal) for ln in part.lines]
    return _write(rows)


def _override_note(r: SweepRow) -> str:
    notes = []
    if r.dropped_overrides:
        notes.append("atlanan ezme: " + ", ".join(r.dropped_overrides))
    if r.kept_overrides:
        notes.append("sabit tutulan ezme: " + ", ".join(r.kept_overrides))
    return "; ".join(notes)


def sweep_csv(rows: Sequence[SweepRow], currency: str, decimal: str = ",") -> str:
    out: list[list[str]] = [
        [
            "adet",
            "parti",
            f"birim maliyet [{currency}]",
            f"birim fiyat [{currency}]",
            f"sipariş toplamı [{currency}]",
            f"etkin birim fiyat [{currency}]",
            "inceleme",
            "ezme notu",
        ]
    ]
    for r in rows:
        out.append(
            [
                str(r.quantity),
                str(r.batch_size),
                _num(r.unit_cost, decimal),
                _num(r.unit_price, decimal),
                _num(r.order_total, decimal),
                _num(r.effective_unit_price, decimal),
                "manual_review" if r.manual_review else "",
                _override_note(r),
            ]
        )
    return _write(out)
