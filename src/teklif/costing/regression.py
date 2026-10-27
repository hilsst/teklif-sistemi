"""Reference-set regression: recompute past jobs and compare with actual cost.

Phase 1 acceptance: computed unit cost (NRE excluded) within ±tolerance of the
actual unit cost (``thresholds.yaml: acceptance.cost_tolerance``). The old quoted
price is compared for information only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from teklif.common.models import Source
from teklif.costing.engine import QuoteInputError, compute_quote, fx_param
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.library.loader import Library
from teklif.storage.history_import import ImportIssue, Value, read_workbook
from teklif.storage.history_schema import REFERENCE_QUOTES

REQUIRED_FOR_REGRESSION = (
    "proses",
    "malzeme",
    "kur_sicakligi",
    "siparis_adedi",
    "parti_adedi",
    "ply_sayisi",
    "toplam_ply_alani",
    "parca_alani",
    "trim_cevresi",
    "kalip_alani",
    "kalip_cevresi",
    "kalip_zarf_x",
    "kalip_zarf_y",
    "karmasiklik",
    "fiili_birim_maliyet",
)


@dataclass(frozen=True)
class RegressionRow:
    quote_no: str
    part_no: str
    status: str  # "ok", "out_of_tolerance", "manual_review", "skipped"
    computed: float | None = None
    actual: float | None = None
    deviation: float | None = None
    quoted_price: float | None = None
    computed_price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class RegressionReport:
    tolerance: float
    rows: list[RegressionRow]
    import_issues: list[ImportIssue]

    @property
    def compared(self) -> list[RegressionRow]:
        return [r for r in self.rows if r.deviation is not None]

    @property
    def within(self) -> list[RegressionRow]:
        return [r for r in self.rows if r.status == "ok"]


def _num(row: dict[str, Value], key: str) -> float:
    v = row[key]
    assert isinstance(v, int | float)
    return float(v)


def row_to_input(row: dict[str, Value], lib: Library) -> QuoteInput:
    """Build a QuoteInput from a reference row (single ply group, tool already exists)."""
    missing = [k for k in REQUIRED_FOR_REGRESSION if row.get(k) is None]
    if missing:
        raise QuoteInputError("eksik alan: " + ", ".join(missing))
    process = str(row["proses"])
    template = lib.processes.get(process)
    if template is None:
        raise QuoteInputError(f"proses şablonu yok: {process}")
    temp = _num(row, "kur_sicakligi")
    cycles = [c.id for c in template.cure_cycles if c.cure_temp.value == temp]
    if not cycles:
        raise QuoteInputError(f"{temp:g} °C için kür çevrimi yok")
    ply_count = int(_num(row, "ply_sayisi"))
    if ply_count < 1:
        raise QuoteInputError("ply_sayisi >= 1 olmalı")
    return QuoteInput(
        quote_no=str(row["teklif_no"]),
        part_no=str(row["parca_no"]),
        process=process,
        material=str(row["malzeme"]),
        cure_cycle=cycles[0],
        complexity=str(row["karmasiklik"]),
        plies=[PlyGroup(area_m2=_num(row, "toplam_ply_alani") / ply_count, count=ply_count)],
        geometry=PartGeometry(
            part_area_m2=_num(row, "parca_alani"),
            part_perimeter_m=_num(row, "trim_cevresi"),
            hole_count=int(row["delik_sayisi"] or 0),  # type: ignore[arg-type]
            tool_area_m2=_num(row, "kalip_alani"),
            tool_perimeter_m=_num(row, "kalip_cevresi"),
            tool_footprint_x_mm=_num(row, "kalip_zarf_x"),
            tool_footprint_y_mm=_num(row, "kalip_zarf_y"),
            source=Source.USER,
        ),
        quantity=int(_num(row, "siparis_adedi")),
        batch_size=int(_num(row, "parti_adedi")),
        first_unit_no=int(row.get("ilk_parca_sira_no") or 1),  # type: ignore[arg-type]
        tooling_exists=True,
    )


def run_regression(path: Path | str, lib: Library) -> RegressionReport:
    imported = read_workbook(path, REFERENCE_QUOTES, lib)
    tol = lib.thresholds.acceptance.cost_tolerance.value
    if tol is None:
        raise ValueError("acceptance.cost_tolerance is TODO")
    ccy = lib.currency.quote_currency
    rows: list[RegressionRow] = []
    for rec in imported.rows:
        qno, pno = str(rec.get("teklif_no") or ""), str(rec.get("parca_no") or "")
        try:
            result = compute_quote(row_to_input(rec, lib), lib)
        except (QuoteInputError, ValueError) as exc:
            rows.append(RegressionRow(qno, pno, "skipped", reason=str(exc)))
            continue
        computed = result.value("cost.unit")
        computed_price = result.value("price.unit")
        row_ccy = str(rec.get("para_birimi") or ccy)
        fx = fx_param(lib, row_ccy, ccy)[0].value
        actual_raw = rec["fiili_birim_maliyet"]
        quoted_raw = rec.get("teklif_birim_fiyat")
        if fx is None:
            rows.append(RegressionRow(qno, pno, "skipped", reason=f"{row_ccy} kuru TODO"))
            continue
        actual = float(actual_raw) * fx  # type: ignore[arg-type]
        quoted = float(quoted_raw) * fx if isinstance(quoted_raw, int | float) else None
        if computed is None:
            codes = sorted({fl.code for _, fl in result.review_items})
            rows.append(
                RegressionRow(
                    qno,
                    pno,
                    "manual_review",
                    actual=actual,
                    quoted_price=quoted,
                    reason="hesaplanamadı: " + ", ".join(codes),
                )
            )
            continue
        dev = (computed - actual) / actual if actual else None
        status = "ok" if dev is not None and abs(dev) <= tol else "out_of_tolerance"
        rows.append(
            RegressionRow(
                qno,
                pno,
                status,
                computed,
                actual,
                dev,
                quoted,
                computed_price,
                "manual_review işaretli" if result.manual_review else "",
            )
        )
    return RegressionReport(tolerance=tol, rows=rows, import_issues=imported.issues)
