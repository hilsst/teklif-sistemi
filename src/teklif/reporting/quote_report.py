"""Quote output: Excel breakdown (openpyxl) and PDF (fpdf2, bundled DejaVu font).

Both show every line with its formula, inputs (value, unit, source), computed result,
override and reason, and manual_review flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF, FontFace
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


FONT_DIR = Path(__file__).resolve().parent / "fonts"
_GREY = (221, 221, 221)
_REVIEW_RGB = (252, 228, 214)
_OVERRIDE_RGB = (255, 242, 204)


def _pdf_doc() -> FPDF:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    return pdf


def pdf_bytes(result: QuoteResult) -> bytes:
    """Render the quote as PDF with fpdf2 (pure Python, works on Windows as is)."""
    inp = result.input
    pdf = _pdf_doc()
    pdf.set_font("DejaVu", "B", 15)
    pdf.cell(0, 9, f"Teklif {inp.quote_no}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 8.5)
    pdf.multi_cell(
        0,
        4.5,
        f"Parça: {inp.part_no} {inp.part_name} · Müşteri: {inp.customer} · "
        f"Proses: {inp.process} · Adet: {inp.quantity} · Parti: {inp.batch_size} · "
        f"Para birimi: {result.currency}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if result.manual_review:
        pdf.set_text_color(170, 0, 0)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(
            0,
            6,
            "Bu teklifte manuel inceleme gerektiren kalemler var.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("DejaVu", "", 8.5)
    with pdf.table(
        col_widths=(110, 50, 26),
        text_align=("LEFT", "RIGHT", "LEFT"),
        headings_style=FontFace(emphasis="BOLD", fill_color=_GREY),
        line_height=5,
    ) as table:
        table.row(["Kalem", "Tutar", "Birim"])
        for r in summary_rows(result):
            style = FontFace(fill_color=_REVIEW_RGB) if r.review else None
            table.row([r.label, fmt(r.value), r.unit], style=style)

    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Kalem kalem kırılım", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 7)
    with pdf.table(
        col_widths=(44, 106, 22, 14),
        text_align=("LEFT", "LEFT", "RIGHT", "LEFT"),
        headings_style=FontFace(emphasis="BOLD", fill_color=_GREY),
        line_height=3.4,
    ) as table:
        table.row(["Kalem", "Formül ve girdiler [kaynak]", "Sonuç", "Birim"])
        for ln in result.lines:
            detail = f"{ln.formula}\n{inputs_text(ln)}"
            if ln.override:
                detail += f"\nEzme: {fmt(ln.override.value)} ({ln.override.reason})"
            if ln.flags:
                detail += f"\nİnceleme: {flags_text(ln)}"
            fill = _REVIEW_RGB if ln.flags else _OVERRIDE_RGB if ln.override else None
            style = FontFace(fill_color=fill) if fill else None
            table.row([f"{ln.key}\n{ln.description}", detail, fmt(ln.final), ln.unit], style=style)
    return bytes(pdf.output())


def write_pdf(result: QuoteResult, path: Path | str) -> Path:
    out = Path(path)
    out.write_bytes(pdf_bytes(result))
    return out
