"""Learning data: save one real case (part + labor hours + tool/model costs) and refit estimators.

A case is a part STEP with the company's real numbers: the hours each operation took, and the
cost of the tool and master model made for it (with their STEP files when they exist). Saved
cases feed the rough tool-cost formula (cost = base + per_m2 x part area), the master-model
cost range and labor-hour fits. Nothing is changed silently: the refit is shown against the
current formula and written only on request, with a backup of the old file.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from teklif.geometry.plate_features import extract_plate_features, plate_step_values
from teklif.ingest.step_props import (
    EXTRACTOR_VERSION,
    StepUnavailableError,
    derive_part_values,
    extract_step_props,
)
from teklif.storage.catalog import (
    Catalog,
    CatalogError,
    MasterModel,
    Part,
    PartOperation,
    Tool,
    ToolPart,
)


@dataclass
class OpEntry:
    """One operation of a case. ``hours`` are person-hours for the whole part."""

    operation: str
    hours: float
    work_center: str = ""
    actual: bool = True


@dataclass
class CaseInput:
    part_no: str
    part_step: tuple[bytes, str] | None = None
    customer: str = ""
    process: str = ""
    ops: list[OpEntry] = field(default_factory=list)
    tool_cost: float | None = None
    model_cost: float | None = None
    currency: str = "EUR"
    tool_step: tuple[bytes, str] | None = None
    model_step: tuple[bytes, str] | None = None
    includes_master: bool = False
    done_on: date | None = None


@dataclass
class CaseResult:
    part_id: int
    notes: list[str] = field(default_factory=list)


def _measure_part(cat: Catalog, cad_id: int, path: Path, notes: list[str]) -> list:  # type: ignore[type-arg]
    try:
        props = extract_step_props(path)
    except StepUnavailableError as exc:
        notes.append(f"{path.name}: ölçülemedi ({exc}); alan elle girilmeli")
        return []
    cat.record_values("cad_file", cad_id, props.values)
    try:
        values = props.values + plate_step_values(extract_plate_features(path))
    except StepUnavailableError as exc:
        notes.append(f"{path.name}: levha özellikleri çıkarılamadı ({exc}); yaklaşık değer")
        values = props.values + derive_part_values(props)
    cat.record_values("cad_file", cad_id, values)
    cat.mark_measured(cad_id, EXTRACTOR_VERSION)
    return values


def _add_cad(
    cat: Catalog, step: tuple[bytes, str], kind: str, notes: list[str]
) -> tuple[int, list]:  # type: ignore[type-arg]
    row, _ = cat.add_cad_file(step[0], step[1], kind)
    values = []
    if kind == "part":
        values = _measure_part(cat, row.id, Path(row.stored_path), notes)
    return row.id, values


def save_case(cat: Catalog, case: CaseInput) -> CaseResult:
    """Store the part, its operations, and the tool/model with costs. Never overwrites."""
    part_no = case.part_no.strip()
    if not part_no:
        raise CatalogError("part number is required")
    with Session(cat.engine) as s:
        if s.scalar(select(Part).where(Part.part_no == part_no, Part.rev == "A")):
            raise CatalogError(f"part {part_no} (rev A) already exists in the catalog")
    notes: list[str] = []
    cad_id: int | None = None
    values: list = []  # type: ignore[type-arg]
    if case.part_step is not None:
        cad_id, values = _add_cad(cat, case.part_step, "part", notes)
    part = cat.add(
        Part(
            part_no=part_no,
            rev="A",
            name=part_no,
            customer=case.customer,
            process=case.process,
            cad_file_id=cad_id,
        )
    )
    if values:
        cat.record_values("part", part.id, values)
    else:
        notes.append("parça alanı kayıtlı değil: kalıp maliyeti formülü bu parçayı kullanamaz")
    for seq, op in enumerate(case.ops, start=1):
        if op.hours <= 0:
            continue
        cat.add_operation(
            part.id,
            "actual" if op.actual else "estimate",
            seq=seq,
            operation=op.operation,
            work_center=op.work_center,
            persons=1.0,
            setup_min=0.0,
            run_min_per_part=op.hours * 60.0,
            batch_size=1,
            done_on=case.done_on,
            note="person-hours",
        )
    model = None
    if case.model_cost is not None or case.model_step is not None:
        model_cad = (
            _add_cad(cat, case.model_step, "master_model", notes)[0] if case.model_step else None
        )
        model = cat.add(MasterModel(model_no=f"{part_no} MODEL", cad_file_id=model_cad))
        if case.model_cost is not None:
            cat.add_cost(
                "master_model", model.id, case.model_cost, case.currency, cost_date=case.done_on
            )
    if case.tool_cost is not None or case.tool_step is not None:
        tool_cad = _add_cad(cat, case.tool_step, "tool", notes)[0] if case.tool_step else None
        tool = cat.add(
            Tool(
                tool_no=f"{part_no} KALIP",
                cad_file_id=tool_cad,
                master_model_id=model.id if model else None,
            )
        )
        if case.tool_cost is not None:
            cat.add_cost(
                "tool",
                tool.id,
                case.tool_cost,
                case.currency,
                cost_date=case.done_on,
                includes_master=case.includes_master,
            )
        cat.link_tool_part(tool.id, part.id)
    return CaseResult(part.id, notes)


# --- collecting samples ---------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    part_no: str
    area_m2: float
    cost: float  # EUR


def _area(cat: Catalog, part_id: int) -> float | None:
    p = cat.params("part", part_id).get("part_area_m2")
    return p.value if p is not None and p.value else None


def collect_tool_samples(
    cat: Catalog, to_eur: Callable[[float, str], float | None]
) -> tuple[list[Sample], list[str]]:
    """(part, tool cost) pairs: tools linked to a part with a measured area and a cost."""
    out: list[Sample] = []
    notes: list[str] = []
    with Session(cat.engine) as s:
        links = list(s.scalars(select(ToolPart)))
        tools = {t.id: t for t in s.scalars(select(Tool))}
        parts = {p.id: p for p in s.scalars(select(Part))}
    seen: set[int] = set()
    for link in links:
        tool, part = tools.get(link.tool_id), parts.get(link.part_id)
        if tool is None or part is None or part.id in seen:
            continue
        costs = cat.costs("tool", tool.id)
        if not costs:
            continue
        rec = costs[-1]
        area = _area(cat, part.id)
        if area is None:
            notes.append(f"{part.part_no}: parça alanı yok, formüle girmedi")
            continue
        if rec.includes_master:
            notes.append(f"{part.part_no}: kalıp maliyeti model dahil, formüle girmedi")
            continue
        eur = to_eur(rec.total, rec.currency)
        if eur is None:
            notes.append(f"{part.part_no}: {rec.currency} kuru yok, formüle girmedi")
            continue
        seen.add(part.id)
        out.append(Sample(part.part_no, area, eur))
    return out, notes


def collect_model_costs(cat: Catalog, to_eur: Callable[[float, str], float | None]) -> list[float]:
    out: list[float] = []
    with Session(cat.engine) as s:
        models = list(s.scalars(select(MasterModel)))
    for m in models:
        costs = cat.costs("master_model", m.id)
        if costs:
            eur = to_eur(costs[-1].total, costs[-1].currency)
            if eur is not None:
                out.append(eur)
    return out


# --- fits -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFit:
    base: float
    slope: float
    n: int
    r_squared: float
    loo_max_error: float  # leave-one-out, relative to the cost
    max_residual_ratio: float
    x_min: float
    x_max: float


def fit_linear(xs: list[float], ys: list[float]) -> LinearFit | None:
    """y = base + slope x. Needs three points with different x; None otherwise."""
    if len(xs) < 3 or len(set(round(x, 6) for x in xs)) < 2:
        return None
    x, y = np.array(xs, dtype=float), np.array(ys, dtype=float)
    slope, base = np.polyfit(x, y, 1)
    pred = base + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    loo = 0.0
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        if len(set(np.round(x[keep], 6))) < 2:
            return None
        s_i, b_i = np.polyfit(x[keep], y[keep], 1)
        loo = max(loo, abs(b_i + s_i * x[i] - y[i]) / abs(y[i]))
    return LinearFit(
        base=float(base),
        slope=float(slope),
        n=len(x),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        loo_max_error=loo,
        max_residual_ratio=float(np.max(np.abs(y - pred) / np.abs(y))),
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


@dataclass(frozen=True)
class ModelCostFit:
    mean: float
    min: float
    max: float
    n: int


def fit_model_cost(costs: list[float]) -> ModelCostFit | None:
    if not costs:
        return None
    return ModelCostFit(float(np.mean(costs)), min(costs), max(costs), len(costs))


def write_cost_models(
    path: Path,
    tool: LinearFit,
    model: ModelCostFit | None,
    valid_for: str,
    r2_area_model: float = 0.0,
) -> Path | None:
    """Rewrite tool_cost_model.yaml from a fit; returns the backup path (None if no old file)."""
    if tool.slope < 1.0:  # under 1 EUR per m2 is no relation to area
        raise ValueError("fit has no positive cost per m2; formula not updated")
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{date.today().isoformat()}")
        shutil.copyfile(path, backup)
    lines = [
        "# Kaba kalıp maliyeti tahmini: maliyet = base + per_m2 x parça tek yüz alanı.",
        "# 'Öğrenme verisi' ekranından, kayıtlı gerçek kalıp maliyetleriyle yeniden hesaplandı.",
        "# Yalnız örneklerin alan aralığında geçerlidir; dışında uyarı verilir.",
        "tool_cost_model:",
        "  basis: part_one_side_area_m2",
        "  currency: EUR",
        f"  base: {tool.base:.2f}",
        f"  per_m2: {tool.slope:.2f}",
        f"  valid_area_m2: {{min: {tool.x_min:.3f}, max: {tool.x_max:.3f}}}",
        f'  valid_for: "{valid_for}"',
        f"  samples: {tool.n}",
        f"  r_squared: {tool.r_squared:.3f}",
        f"  max_residual_ratio: {tool.max_residual_ratio:.3f}",
        f"  leave_one_out_max_error: {tool.loo_max_error:.3f}",
        "  source: derived",
    ]
    if model is not None:
        lines += [
            "",
            "# Master model maliyeti: alanla ilişkisi zayıf; geçmiş modellerin ortalaması/aralığı",
            "model_cost_model:",
            "  currency: EUR",
            f"  mean: {model.mean:.2f}",
            f"  min: {model.min:.2f}",
            f"  max: {model.max:.2f}",
            f"  samples: {model.n}",
            f"  r_squared_area: {r2_area_model:.2f}",
            f'  valid_for: "{valid_for}"',
            "  source: derived",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup


# --- labor ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LaborRow:
    part_no: str
    area_m2: float | None
    hours: dict[str, float]  # operation -> person-hours


def labor_rows(cat: Catalog) -> list[LaborRow]:
    """Per part: area and person-hours per operation (actual entries only)."""
    rows: list[LaborRow] = []
    with Session(cat.engine) as s:
        parts = list(s.scalars(select(Part)))
        ops = list(s.scalars(select(PartOperation).where(PartOperation.kind == "actual")))
    by_part: dict[int, dict[str, float]] = {}
    for o in ops:
        hours = o.persons * o.run_min_per_part / 60.0
        d = by_part.setdefault(o.part_id, {})
        d[o.operation] = d.get(o.operation, 0.0) + hours
    for p in parts:
        if p.id in by_part:
            rows.append(LaborRow(p.part_no, _area(cat, p.id), by_part[p.id]))
    return rows


def fit_labor(rows: list[LaborRow], operation: str) -> LinearFit | None:
    xs, ys = [], []
    for r in rows:
        if r.area_m2 is not None and r.hours.get(operation, 0.0) > 0:
            xs.append(r.area_m2)
            ys.append(r.hours[operation])
    return fit_linear(xs, ys)
