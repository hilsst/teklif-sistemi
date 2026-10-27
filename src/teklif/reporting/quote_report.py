"""Quote output: Excel breakdown (openpyxl) and PDF (WeasyPrint, optional extra).

Both show every line with its formula, inputs (value, unit, source), computed result,
override and reason, and manual_review flags.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from teklif.common.models import CostLine
from teklif.costing.engine import QuoteResult

CATEGORY_LABELS: dict[str, str] = {
    "time": "Süre",
    "labor": "İşçilik",
    "machine": "Makine",
    "material": "Malzeme",
    "consumable": "Sarf",
    "autoclave": "Autoclave",
    "total": "Ara toplam",
    "price": "Fiyat",
    "nre": "NRE",
}

SUMMARY_KEYS: list[tuple[str, str]] = [
    ("material.total", "Malzeme"),
    ("consumable.total", "Sarf"),
    ("labor.total", "İşçilik"),
    ("machine.total", "Makine"),
    ("autoclave.total", "Autoclave"),
    ("cost.direct", "Direkt maliyet"),
    ("cost.overhead", "Genel gider"),
    ("cost.unit", "Birim maliyet"),
    ("price.unit", "Birim fiyat"),
    ("nre.tooling", "Kalıp + master model (NRE)"),
    ("price.unit_final", "Teklif birim fiyatı"),
    ("price.order_total", "Sipariş toplamı"),
]

_REVIEW_FILL = PatternFill("solid", fgColor="FCE4D6")
_OVERRIDE_FILL = PatternFill("solid", fgColor="FFF2CC")


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def inputs_text(line: CostLine) -> str:
    return "; ".join(f"{i.name}={fmt(i.value, 4)} {i.unit} [{i.source.value}]" for i in line.inputs)


def flags_text(line: CostLine) -> str:
    return "; ".join(f"{fl.code}: {fl.reason}" for fl in line.flags)


@dataclass(frozen=True)
class SummaryRow:
    label: str
    value: float | None
    unit: str
    review: bool


def summary_rows(result: QuoteResult) -> list[SummaryRow]:
    keys = {ln.key: ln for ln in result.lines}
    return [
        SummaryRow(label, keys[k].final, keys[k].unit, keys[k].manual_review)
        for k, label in SUMMARY_KEYS
        if k in keys
    ]


def write_excel(result: QuoteResult, path: Path | str) -> Path:
    inp = result.input
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Ozet"
    ws.append([f"Teklif {inp.quote_no}"])
    ws["A1"].font = Font(bold=True, size=14)
    for label, val in [
        ("Parça", f"{inp.part_no} {inp.part_name}".strip()),
        ("Müşteri", inp.customer),
        ("Proses", inp.process),
        ("Adet", inp.quantity),
        ("Parti", inp.batch_size),
        ("Para birimi", result.currency),
        ("Manuel inceleme", "EVET" if result.manual_review else "hayır"),
    ]:
        ws.append([label, val])
    ws.append([])
    ws.append(["Kalem", "Tutar", "Birim", "İnceleme"])
    for c in ws[ws.max_row]:
        c.font = Font(bold=True)
    for row in summary_rows(result):
        ws.append([row.label, row.value, row.unit, "manual_review" if row.review else ""])
        ws.cell(ws.max_row, 2).number_format = "#,##0.00"
        if row.review:
            for c in ws[ws.max_row]:
                c.fill = _REVIEW_FILL
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["D"].width = 16

    br = wb.create_sheet("Kirilim")
    headers = [
        "Kalem",
        "Kategori",
        "Açıklama",
        "Formül",
        "Girdiler [kaynak]",
        "Hesaplanan",
        "Ezilen",
        "Ezme gerekçesi",
        "Sonuç",
        "Birim",
        "Manuel inceleme",
    ]
    br.append(headers)
    for c in br[1]:
        c.font = Font(bold=True)
    for ln in result.lines:
        br.append(
            [
                ln.key,
                CATEGORY_LABELS.get(ln.category, ln.category),
                ln.description,
                ln.formula,
                inputs_text(ln),
                ln.result,
                ln.override.value if ln.override else None,
                ln.override.reason if ln.override else "",
                ln.final,
                ln.unit,
                flags_text(ln),
            ]
        )
        r = br.max_row
        for col in (6, 7, 9):
            br.cell(r, col).number_format = "#,##0.00"
        fill = _REVIEW_FILL if ln.flags else _OVERRIDE_FILL if ln.override else None
        if fill is not None:
            for c in br[r]:
                c.fill = fill
    for letter, width in zip(
        "ABCDEFGHIJK", (30, 12, 34, 40, 70, 13, 11, 28, 13, 7, 50), strict=True
    ):
        br.column_dimensions[letter].width = width
    for row in br.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    br.freeze_panes = "B2"

    out = Path(path)
    wb.save(out)
    return out


def render_html(result: QuoteResult) -> str:
    inp = result.input
    e = html.escape
    summary = "".join(
        f"<tr class='{'review' if r.review else ''}'><td>{e(r.label)}</td>"
        f"<td class='num'>{fmt(r.value)}</td><td>{e(r.unit)}</td></tr>"
        for r in summary_rows(result)
    )
    lines = []
    for ln in result.lines:
        cls = "review" if ln.flags else "override" if ln.override else ""
        over = (
            f"<br><b>Ezme:</b> {fmt(ln.override.value)} ({e(ln.override.reason)})"
            if ln.override
            else ""
        )
        flags = f"<br><b>İnceleme:</b> {e(flags_text(ln))}" if ln.flags else ""
        lines.append(
            f"<tr class='{cls}'><td>{e(ln.key)}<br><small>{e(ln.description)}</small></td>"
            f"<td><code>{e(ln.formula)}</code><br><small>{e(inputs_text(ln))}</small>"
            f"{over}{flags}</td>"
            f"<td class='num'>{fmt(ln.final)}</td><td>{e(ln.unit)}</td></tr>"
        )
    review_note = (
        "<p class='warn'>Bu teklifte manuel inceleme gerektiren kalemler var.</p>"
        if result.manual_review
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 14mm; }}
body {{ font-family: "DejaVu Sans", Arial, sans-serif; font-size: 9pt; }}
h1 {{ font-size: 15pt; margin: 0 0 4mm; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 6mm; }}
td, th {{ border: 0.5pt solid #999; padding: 2pt 4pt; vertical-align: top; }}
th {{ background: #ddd; text-align: left; }}
tr {{ page-break-inside: avoid; }}
.num {{ text-align: right; white-space: nowrap; }}
.review {{ background: #fce4d6; }} .override {{ background: #fff2cc; }}
.warn {{ color: #a00; font-weight: bold; }} small {{ color: #444; }}
</style></head><body>
<h1>Teklif {e(inp.quote_no)}</h1>
<p>Parça: {e(inp.part_no)} {e(inp.part_name)} · Müşteri: {e(inp.customer)} ·
Proses: {e(inp.process)} · Adet: {inp.quantity} · Parti: {inp.batch_size} ·
Para birimi: {e(result.currency)}</p>
{review_note}
<table><tr><th>Kalem</th><th>Tutar</th><th>Birim</th></tr>{summary}</table>
<h2>Kalem kalem kırılım</h2>
<table><tr><th>Kalem</th><th>Formül ve girdiler [kaynak]</th><th>Sonuç</th><th>Birim</th></tr>
{"".join(lines)}</table>
</body></html>"""


def write_pdf(result: QuoteResult, path: Path | str) -> Path:
    try:
        from weasyprint import HTML  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "PDF output needs the 'pdf' extra: uv sync --extra pdf "
            "(on Windows WeasyPrint also needs the GTK runtime)."
        ) from exc
    out = Path(path)
    HTML(string=render_html(result)).write_pdf(out)
    return out
