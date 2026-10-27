"""Operation list and driver quantities for a part, from a YAML process template."""

from __future__ import annotations

import math
from dataclasses import dataclass

from teklif.common.models import Param, Source
from teklif.costing.inputs import QuoteInput
from teklif.library.schema import DRIVER_UNITS, Operation, ProcessTemplate


class RouteError(ValueError):
    """The quote input does not match the process template."""


@dataclass(frozen=True)
class DriverValue:
    param: Param
    path: str | None = None


def active_operations(template: ProcessTemplate, inp: QuoteInput) -> list[Operation]:
    known = {op.id for op in template.operations}
    unknown = (set(inp.include_optional) | set(inp.exclude_operations)) - known
    if unknown:
        raise RouteError(f"Unknown operations for '{template.id}': {sorted(unknown)}")
    return [
        op
        for op in template.operations
        if (not op.optional or op.id in inp.include_optional)
        and op.id not in inp.exclude_operations
    ]


def compute_drivers(template: ProcessTemplate, inp: QuoteInput) -> dict[str, DriverValue]:
    """Driver quantities per part, each with its unit and source."""
    g = inp.geometry
    src = g.source

    def val(name: str, value: float | None, source: Source = src, desc: str = "") -> Param:
        return Param(value=value, unit=DRIVER_UNITS[name], source=source, desc=desc or None)

    ply_count = sum(p.count for p in inp.plies)
    ply_area = math.fsum(p.area_m2 * p.count for p in inp.plies)

    interval = template.parameters.get("debulk_interval_plies")
    debulk: float | None = None
    debulk_path: str | None = None
    if interval is not None:
        debulk_path = f"processes/{template.id}.yaml:parameters.debulk_interval_plies"
        if interval.value is not None and interval.value > 0:
            debulk = float(math.ceil(ply_count / interval.value))

    return {
        "per_part": DriverValue(val("per_part", 1.0, Source.DERIVED)),
        "ply_area_total_m2": DriverValue(
            val("ply_area_total_m2", ply_area, desc="Σ ply area × count")
        ),
        "ply_count": DriverValue(val("ply_count", float(ply_count))),
        "part_area_m2": DriverValue(val("part_area_m2", g.part_area_m2)),
        "part_perimeter_m": DriverValue(val("part_perimeter_m", g.part_perimeter_m)),
        "hole_count": DriverValue(val("hole_count", float(g.hole_count))),
        "tool_area_m2": DriverValue(val("tool_area_m2", g.tool_area_m2)),
        "tool_perimeter_m": DriverValue(val("tool_perimeter_m", g.tool_perimeter_m)),
        "debulk_count": DriverValue(
            val("debulk_count", debulk, Source.DERIVED, "ceil(ply_count / debulk_interval)"),
            debulk_path,
        ),
    }
