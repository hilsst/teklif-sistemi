"""Basic STEP properties (envelope, volume, area) with OCP; no draft analysis (Phase 2).

OCP (``cadquery-ocp``) is an optional dependency (``pip install teklif[cad]``). Without it
``extract_step_props`` raises ``StepUnavailableError`` and every value stays manual.

Values are measured in the file's model space after OCCT converts to millimetres. The
envelope is the *axis-aligned* bounding box in the file's coordinate system; if the part was
modelled at an angle it is larger than the oriented envelope, so it carries a review flag.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from kalip_analiz.step import StepUnavailableError as StepUnavailableError
from kalip_analiz.step import load_step_shape as load_step_shape
from teklif.common.models import Source

EXTRACTOR_VERSION = "step_props-2"
MM2_PER_M2 = 1_000_000.0
MM3_PER_M3 = 1_000_000_000.0


class StepValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: float | None
    unit: str
    source: Source
    review_reason: str | None = None


class StepProps(BaseModel):
    model_config = ConfigDict(frozen=True)

    solid_count: int
    file_length_units: list[str]
    values: list[StepValue]

    def get(self, key: str) -> StepValue:
        return next(v for v in self.values if v.key == key)


def ocp_available() -> bool:
    try:
        import OCP  # noqa: F401
    except ImportError:
        return False
    return True


def solid_list(shape: Any) -> list[Any]:
    """The solids of a shape in traversal order (the index used everywhere for assemblies)."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    out = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def pick_solids(shape: Any, solids: Collection[int]) -> Any:
    """Compound of the solids with these indices."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    all_solids = solid_list(shape)
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for i in sorted(set(solids)):
        if 0 <= i < len(all_solids):
            builder.Add(comp, all_solids[i])
    return comp


def extract_step_props(path: Path | str, solids_only: Collection[int] | None = None) -> StepProps:
    """Measure envelope, volume and surface area of a STEP file.

    ``solids_only``: for an assembly, measure only these solid indices.
    """
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc

    shape, units = load_step_shape(path)
    if solids_only:
        shape = pick_solids(shape, solids_only)

    solids = 0
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids += 1
        explorer.Next()

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    xmin, ymin, zmin = box.CornerMin().X(), box.CornerMin().Y(), box.CornerMin().Z()
    xmax, ymax, zmax = box.CornerMax().X(), box.CornerMax().Y(), box.CornerMax().Z()
    area_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area_props)

    values = [
        StepValue(key="envelope_x_mm", value=xmax - xmin, unit="mm", source=Source.CAD),
        StepValue(key="envelope_y_mm", value=ymax - ymin, unit="mm", source=Source.CAD),
        StepValue(key="envelope_z_mm", value=zmax - zmin, unit="mm", source=Source.CAD),
        StepValue(
            key="surface_area_m2",
            value=area_props.Mass() / MM2_PER_M2,
            unit="m2",
            source=Source.CAD,
        ),
    ]
    if solids:
        vol_props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vol_props)
        values.append(
            StepValue(
                key="volume_m3", value=vol_props.Mass() / MM3_PER_M3, unit="m3", source=Source.CAD
            )
        )
    else:
        values.append(
            StepValue(
                key="volume_m3",
                value=None,
                unit="m3",
                source=Source.CAD,
                review_reason="surface model: no solid, enter thickness manually",
            )
        )
    if solids > 1:
        values = [
            v.model_copy(update={"review_reason": f"{solids} solids in file: values are summed"})
            for v in values
        ]
    if units and units[0].upper() not in ("MILLIMETRE", "MILLIMETER", "MM"):
        values = [
            v.model_copy(
                update={
                    "review_reason": v.review_reason
                    or f"file length unit is {units[0]}, converted to mm"
                }
            )
            for v in values
        ]
    return StepProps(solid_count=solids, file_length_units=units, values=values)


def derive_part_values(props: StepProps) -> list[StepValue]:
    """Part-card values derived from a thin solid. Always flagged for review.

    Hole count and trim perimeter need face classification and arrive with Phase 2.
    """
    area = props.get("surface_area_m2").value
    volume = props.get("volume_m3").value
    if area is None or volume is None or area <= 0:
        return []
    one_side = area / 2
    reason = "total area / 2: edge faces are included, so slightly high for thick parts"
    return [
        StepValue(
            key="part_area_m2",
            value=one_side,
            unit="m2",
            source=Source.DERIVED,
            review_reason=reason,
        ),
        StepValue(
            key="mean_thickness_mm",
            value=volume * MM3_PER_M3 / (one_side * MM2_PER_M2),
            unit="mm",
            source=Source.DERIVED,
            review_reason="volume / one-side area; compare with ply count x CPT",
        ),
    ]
