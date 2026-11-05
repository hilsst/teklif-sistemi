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
from teklif.costing.quote_set import QuoteSetResult

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
    ("nre.master", "Master model (NRE)"),
    ("nre.total", "NRE toplamı"),
    ("price.nre_priced", "Kalıp + model satış bedeli (kârlı)"),
    ("price.part_cost", "Ürün başı: parça maliyeti"),
    ("price.share_master", "Ürün başı: model payı"),
    ("price.share_tooling", "Ürün başı: kalıp payı"),
    ("price.profit", "Ürün başı: kâr"),
    ("price.unit_final", "Teklif birim fiyatı"),
    ("price.order_raw", "Sipariş toplamı (minimum öncesi)"),
    ("price.min_order_topup", "Minimum sipariş farkı"),
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
    labels = dict(SUMMARY_KEYS)
    if "nre.master" in keys:
        labels["nre.tooling"] = "Kalıp (NRE)"
    return [
        SummaryRow(labels[k], keys[k].final, keys[k].unit, bool(keys[k].open_flags))
        for k, _ in SUMMARY_KEYS
        if k in keys
    ]


def _breakdown_sheet(wb: Workbook, title: str, lines: list[CostLine]) -> None:
    br = wb.create_sheet(title)
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
    for ln in lines:
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
        ("Manuel inceleme", "EVET" if result.has_open_review else "hayır"),
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

    _breakdown_sheet(wb, "Kirilim", result.lines)

    out = Path(path)
    wb.save(out)
    return out


FONT_DIR = Path(__file__).resolve().parent / "fonts"
_GREY = (221, 221, 221)
_REVIEW_RGB = (252, 228, 214)
_OVERRIDE_RGB = (255, 242, 204)


class _QuotePdf(FPDF):
    """FPDF with an optional diagonal TASLAK (draft) watermark on every page."""

    draft = False

    def header(self) -> None:
        if not self.draft:
            return
        self.set_font("DejaVu", "B", 80)
        self.set_text_color(235, 200, 200)
        with self.rotation(45, x=105, y=148):
            self.text(45, 160, "TASLAK")
        self.set_text_color(0, 0, 0)


def _pdf_doc(draft: bool = False) -> FPDF:
    pdf = _QuotePdf(orientation="P", unit="mm", format="A4")
    pdf.draft = draft
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    return pdf


DRAFT_NOTE = "TASLAK: manuel inceleme bayrakları açık; bu belge müşteriye gönderilemez."


def _pdf_breakdown(pdf: FPDF, lines: list[CostLine]) -> None:
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
        for ln in lines:
            detail = f"{ln.formula}\n{inputs_text(ln)}"
            if ln.override:
                detail += f"\nEzme: {fmt(ln.override.value)} ({ln.override.reason})"
            if ln.flags:
                detail += f"\nİnceleme: {flags_text(ln)}"
            fill = _REVIEW_RGB if ln.flags else _OVERRIDE_RGB if ln.override else None
            style = FontFace(fill_color=fill) if fill else None
            table.row([f"{ln.key}\n{ln.description}", detail, fmt(ln.final), ln.unit], style=style)


def pdf_bytes(result: QuoteResult) -> bytes:
    """Render the quote as PDF with fpdf2 (pure Python, works on Windows as is)."""
    inp = result.input
    pdf = _pdf_doc(draft=result.has_open_review)
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
    if result.has_open_review:
        pdf.set_text_color(170, 0, 0)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(
            0,
            6,
            DRAFT_NOTE,
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

    _pdf_breakdown(pdf, result.lines)
    return bytes(pdf.output())


def write_pdf(result: QuoteResult, path: Path | str) -> Path:
    out = Path(path)
    out.write_bytes(pdf_bytes(result))
    return out


# --- multi-item quote (quote set) ----------------------------------------------------

SET_SUMMARY_KEYS: list[tuple[str, str]] = [
    ("set.parts_total", "Parça kalemleri toplamı"),
    ("set.nre_total", "NRE kalemleri toplamı"),
    ("set.order_raw", "Teklif toplamı (minimum öncesi)"),
    ("set.min_order_topup", "Minimum sipariş farkı"),
    ("set.order_total", "Teklif toplamı"),
]


def set_item_rows(result: QuoteSetResult) -> list[SummaryRow]:
    """One row per item: parts at their order total, NRE items at their amount."""
    rows = [
        SummaryRow(
            f"{r.input.part_no} {r.input.part_name} (adet {r.input.quantity})".strip(),
            r.value("price.order_total"),
            r.currency,
            r.has_open_review,
        )
        for r in result.parts
    ]
    rows += [
        SummaryRow(ln.description, ln.final, ln.unit, bool(ln.open_flags))
        for ln in result.lines
        if ln.key.startswith("nre.")
    ]
    return rows


def set_total_rows(result: QuoteSetResult) -> list[SummaryRow]:
    keys = {ln.key: ln for ln in result.lines}
    return [
        SummaryRow(label, keys[k].final, keys[k].unit, bool(keys[k].open_flags))
        for k, label in SET_SUMMARY_KEYS
        if k in keys
    ]


_BAD_SHEET_CHARS = set("/\\?*[]:")


def _sheet_title(part_no: str, index: int, used: set[str]) -> str:
    """Excel-safe, unique sheet title (<= 31 chars, no / \\ ? * [ ] :)."""
    clean = "".join("-" if c in _BAD_SHEET_CHARS else c for c in part_no).strip("' ")
    title = f"K{index:02d} {clean}"[:31]
    while title in used:  # cannot happen with the index prefix, kept as a guard
        title = title[:-1] + "_"
    used.add(title)
    return title


def write_set_excel(result: QuoteSetResult, path: Path | str) -> Path:
    inp = result.input
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Ozet"
    ws.append([f"Teklif {inp.quote_no}"])
    ws["A1"].font = Font(bold=True, size=14)
    for label, val in [
        ("Müşteri", inp.customer),
        ("Şablon", inp.blueprint or ""),
        ("Para birimi", result.currency),
        ("Manuel inceleme", "EVET (TASLAK)" if result.has_open_review else "hayır"),
    ]:
        ws.append([label, val])
    ws.append([])
    for heading, rows in (("Kalemler", set_item_rows(result)), ("Toplam", set_total_rows(result))):
        ws.append([heading, "Tutar", "Birim", "İnceleme"])
        for c in ws[ws.max_row]:
            c.font = Font(bold=True)
        for row in rows:
            ws.append([row.label, row.value, row.unit, "manual_review" if row.review else ""])
            ws.cell(ws.max_row, 2).number_format = "#,##0.00"
            if row.review:
                for c in ws[ws.max_row]:
                    c.fill = _REVIEW_FILL
        ws.append([])
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["D"].width = 16
    _breakdown_sheet(wb, "Kirilim (teklif)", result.lines)
    used = {"Ozet", "Kirilim (teklif)"}
    for i, r in enumerate(result.parts, start=1):
        _breakdown_sheet(wb, _sheet_title(r.input.part_no, i, used), r.lines)
    out = Path(path)
    wb.save(out)
    return out


def set_pdf_bytes(result: QuoteSetResult) -> bytes:
    inp = result.input
    pdf = _pdf_doc(draft=result.has_open_review)
    pdf.set_font("DejaVu", "B", 15)
    pdf.cell(0, 9, f"Teklif {inp.quote_no}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 8.5)
    pdf.multi_cell(
        0,
        4.5,
        f"Müşteri: {inp.customer} · Para birimi: {result.currency} · "
        f"Kalem sayısı: {len(inp.parts) + len(inp.nre_items)}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if result.has_open_review:
        pdf.set_text_color(170, 0, 0)
        pdf.set_font("DejaVu", "B", 9)
        pdf.cell(0, 6, DRAFT_NOTE, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_font("DejaVu", "", 8.5)
    for heading, rows in (("Kalemler", set_item_rows(result)), ("Toplam", set_total_rows(result))):
        with pdf.table(
            col_widths=(110, 50, 26),
            text_align=("LEFT", "RIGHT", "LEFT"),
            headings_style=FontFace(emphasis="BOLD", fill_color=_GREY),
            line_height=5,
        ) as table:
            table.row([heading, "Tutar", "Birim"])
            for r in rows:
                style = FontFace(fill_color=_REVIEW_RGB) if r.review else None
                table.row([r.label, fmt(r.value), r.unit], style=style)
        pdf.ln(3)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Teklif düzeyi kırılım", new_x="LMARGIN", new_y="NEXT")
    _pdf_breakdown(pdf, result.lines)
    for part in result.parts:
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 13)
        pdf.cell(
            0,
            8,
            f"Parça {part.input.part_no} {part.input.part_name}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("DejaVu", "", 8.5)
        with pdf.table(
            col_widths=(110, 50, 26),
            text_align=("LEFT", "RIGHT", "LEFT"),
            headings_style=FontFace(emphasis="BOLD", fill_color=_GREY),
            line_height=5,
        ) as table:
            table.row(["Kalem", "Tutar", "Birim"])
            for row in summary_rows(part):
                style = FontFace(fill_color=_REVIEW_RGB) if row.review else None
                table.row([row.label, fmt(row.value), row.unit], style=style)
        _pdf_breakdown(pdf, part.lines)
    return bytes(pdf.output())


def write_set_pdf(result: QuoteSetResult, path: Path | str) -> Path:
    out = Path(path)
    out.write_bytes(set_pdf_bytes(result))
    return out
