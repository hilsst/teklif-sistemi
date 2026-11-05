"""Which library values does a quote for a given process actually need?

A probe quote is computed on the library as it is; every manual-review flag that points to a
library value tells us that this value is used (and missing). Values no quote of that process
touches stay hidden in the default library view, so the user is not asked for them.
"""

from __future__ import annotations

import itertools

from teklif.common.models import Source
from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.library import Library
from teklif.library.editor import EditableValue

# Needed by the automatic ply count / fabric price whatever the flags say.
_ALWAYS_SUFFIXES = (".areal_weight", ".cured_ply_thickness", ".roll_width")


def needed_paths(lib: Library, process: str) -> set[str]:
    """``file:path`` of every library value that a quote for ``process`` reads, filled or not
    (all material, autoclave, cure-cycle and complexity choices, all optional operations)."""
    tpl = lib.processes[process]
    optional = [op.id for op in tpl.operations if op.optional]
    geom = PartGeometry(
        part_area_m2=1.0,
        part_perimeter_m=1.0,
        hole_count=1,
        tool_area_m2=2.0,
        tool_perimeter_m=3.0,
        tool_footprint_x_mm=1000.0,
        tool_footprint_y_mm=1000.0,
        source=Source.DERIVED,
    )
    out: set[str] = set()
    for mat in lib.materials.prepregs:
        for ac in lib.autoclaves.autoclaves:
            for cure, cx in itertools.product(tpl.cure_cycles, lib.thresholds.complexity_factors):
                inp = QuoteInput(
                    quote_no="probe",
                    part_no="probe",
                    customer="",
                    process=process,
                    material=mat.id,
                    cure_cycle=cure.id,
                    autoclave=ac.id,
                    complexity=cx,
                    plies=[PlyGroup(name="probe", area_m2=1.0, count=1)],
                    geometry=geom,
                    quantity=1,
                    batch_size=1,
                    include_optional=optional,
                )
                try:
                    res = compute_quote(inp, lib)
                except (QuoteInputError, ArithmeticError, ValueError):
                    continue
                out |= {i.path for ln in res.lines for i in ln.inputs if i.path}
    return out


def is_needed(v: EditableValue, needed: set[str]) -> bool:
    if ".consumables[" in v.path and v.path.endswith(".qty"):
        return False  # proposed from geometry and approved on the quote screen
    return f"{v.file}:{v.path}" in needed or (
        v.file == "materials.yaml" and v.path.endswith(_ALWAYS_SUFFIXES)
    )
