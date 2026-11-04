"""Basic STEP properties (envelope, volume, area) with OCP; no draft analysis (Phase 2).

OCP (``cadquery-ocp``) is an optional dependency (``pip install teklif[cad]``). Without it
``extract_step_props`` raises ``StepUnavailableError`` and every value stays manual.

Values are measured in the file's model space after OCCT converts to millimetres. The
envelope is the *axis-aligned* bounding box in the file's coordinate system; if the part was
modelled at an angle it is larger than the oriented envelope, so it carries a review flag.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from teklif.common.models import Source

EXTRACTOR_VERSION = "step_props-1"
MM2_PER_M2 = 1_000_000.0
MM3_PER_M3 = 1_000_000_000.0


class StepUnavailableError(RuntimeError):
    """OCP is not installed or the file cannot be read."""


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


def extract_step_props(path: Path | str) -> StepProps:
    """Measure envelope, volume and surface area of a STEP file."""
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.collections import Sequence_TCollection_AsciiString
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise StepUnavailableError(f"cannot read STEP file {path}")
    length, angle, solid_angle = (Sequence_TCollection_AsciiString() for _ in range(3))
    reader.FileUnits(length, angle, solid_angle)
    units = [length.Value(i + 1).ToCString() for i in range(length.Length())]
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise StepUnavailableError("STEP file contains no shape")

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
