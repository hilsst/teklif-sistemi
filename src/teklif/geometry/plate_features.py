"""Faz 2a: shell-part features from a STEP solid (one-side area, thickness, holes, trim).

All values are *estimates* from face classification, always derived + review-flagged.
Heuristic: faces whose width (2 x area / perimeter) is at most ``EDGE_WIDTH_FACTOR`` x the
mean thickness are treated as edge (rim / hole wall) faces; the rest are the two skins.
Thickness is iterated once: t0 = V / (A_total / 2), then t = V / A_oneside.
Closed sections and complexity scores are not decided here (metrics only).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from teklif.common.models import Source
from teklif.ingest.step_props import (
    MM2_PER_M2,
    StepUnavailableError,
    StepValue,
    load_step_shape,
)

EDGE_WIDTH_FACTOR = 1.5
MM3_PER_M3 = 1_000_000_000.0
EXTRACTOR_VERSION = "plate_features-1"


class Hole(BaseModel):
    model_config = ConfigDict(frozen=True)

    diameter_mm: float
    count: int


class PlateFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    one_side_area_m2: float
    mean_thickness_mm: float
    trim_perimeter_mm: float
    hole_count: int
    holes: list[Hole]
    curved_area_fraction: float
    edge_area_m2: float
    notes: list[str]
    extractor_version: str = EXTRACTOR_VERSION


class _Face(BaseModel):
    area: float
    perimeter: float
    planar: bool
    hole_key: tuple[float, float, float, float, float, float, float] | None = None
    hole_angle: float = 0.0


def _as_face(shape: Any) -> Any:
    from OCP.TopoDS import TopoDS

    cast = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    return cast(shape)


def _faces(shape: Any) -> list[_Face]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer

    out: list[_Face] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = _as_face(exp.Current())
        exp.Next()
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        e = TopExp_Explorer(face, TopAbs_EDGE)
        perim = 0.0
        while e.More():
            lp = GProp_GProps()
            BRepGProp.LinearProperties_s(e.Current(), lp)
            perim += lp.Mass()
            e.Next()
        ad = BRepAdaptor_Surface(face)
        kind = ad.GetType()
        key = None
        angle = 0.0
        if kind == GeomAbs_Cylinder:
            cyl = ad.Cylinder()
            ax = cyl.Axis()
            d = ax.Direction()
            loc = ax.Location()
            u0, u1 = ad.FirstUParameter(), ad.LastUParameter()
            um, vm = (u0 + u1) / 2, (ad.FirstVParameter() + ad.LastVParameter()) / 2
            p = ad.Value(um, vm)
            du, dv = ad.DN(um, vm, 1, 0), ad.DN(um, vm, 0, 1)
            n = du.Crossed(dv)
            if face.Orientation() == TopAbs_REVERSED:
                n.Reverse()
            radial = (p.X() - loc.X(), p.Y() - loc.Y(), p.Z() - loc.Z())
            t = radial[0] * d.X() + radial[1] * d.Y() + radial[2] * d.Z()
            rad = (radial[0] - t * d.X(), radial[1] - t * d.Y(), radial[2] - t * d.Z())
            if n.X() * rad[0] + n.Y() * rad[1] + n.Z() * rad[2] < 0:  # normal toward axis
                dx, dy, dz = d.X(), d.Y(), d.Z()
                if (dx, dy, dz) < (0.0, 0.0, 0.0):
                    dx, dy, dz = -dx, -dy, -dz
                # axis line identified by its closest point to the origin
                s = loc.X() * dx + loc.Y() * dy + loc.Z() * dz
                cx, cy, cz = loc.X() - s * dx, loc.Y() - s * dy, loc.Z() - s * dz
                key = tuple(
                    round(v, 1) for v in (cyl.Radius(), dx * 10, dy * 10, dz * 10, cx, cy, cz)
                )
                angle = abs(u1 - u0)
        out.append(
            _Face(
                area=area,
                perimeter=perim,
                planar=kind == GeomAbs_Plane,
                hole_key=key,
                hole_angle=angle,
            )
        )
    return out


def extract_plate_features(path: Path | str) -> PlateFeatures:
    """Estimate shell-part features from a solid STEP file."""
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc
    shape, _units = load_step_shape(path)
    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vp)
    volume = vp.Mass()
    if volume <= 0:
        raise StepUnavailableError("no solid volume: plate features need a solid model")
    faces = _faces(shape)
    total = sum(f.area for f in faces)
    notes: list[str] = []
    t = volume / (total / 2)
    one_side = total / 2
    edge: list[_Face] = []
    for _ in range(2):
        edge = [
            f
            for f in faces
            if f.perimeter > 0 and 2 * f.area / f.perimeter <= EDGE_WIDTH_FACTOR * t
        ]
        edge_area = sum(f.area for f in edge)
        one_side = (total - edge_area) / 2
        if one_side <= 0:
            raise StepUnavailableError("could not separate skins from edges: not a thin part?")
        t = volume / one_side
    edge_area = sum(f.area for f in edge)
    # holes: cluster concave cylinders by axis and radius, sum angular extent
    clusters: dict[tuple[float, ...], float] = {}
    hole_area = 0.0
    for f in faces:
        if f.hole_key is not None:
            clusters[f.hole_key] = clusters.get(f.hole_key, 0.0) + f.hole_angle
            hole_area += f.area
    holes: dict[float, int] = {}
    for key, ang in clusters.items():
        if ang >= 2 * math.pi * 0.95:
            diameter = round(2 * key[0], 1)
            holes[diameter] = holes.get(diameter, 0) + round(ang / (2 * math.pi))
    hole_count = sum(holes.values())
    outer_edge = max(edge_area - hole_area, 0.0)
    trim = outer_edge / t
    skin = [f for f in faces if f not in edge]
    curved = sum(f.area for f in skin if not f.planar)
    skin_area = sum(f.area for f in skin) or 1.0
    notes.append("estimated from face classification; trim perimeter includes non-circular cutouts")
    if t > 0 and max((f.area for f in faces), default=0) < 5 * t * t:
        notes.append("part may not be a thin shell: check thickness")
    return PlateFeatures(
        one_side_area_m2=one_side / MM2_PER_M2,
        mean_thickness_mm=t,
        trim_perimeter_mm=trim,
        hole_count=hole_count,
        holes=[Hole(diameter_mm=d, count=c) for d, c in sorted(holes.items())],
        curved_area_fraction=curved / skin_area,
        edge_area_m2=edge_area / MM2_PER_M2,
        notes=notes,
    )


def plate_step_values(f: PlateFeatures) -> list[StepValue]:
    """Part-card values from the features; all derived and flagged for review."""
    reason = "; ".join(f.notes)

    def v(key: str, value: float, unit: str, why: str) -> StepValue:
        return StepValue(
            key=key, value=value, unit=unit, source=Source.DERIVED, review_reason=f"{why}. {reason}"
        )

    return [
        v("part_area_m2", f.one_side_area_m2, "m2", "total area minus edge faces, halved"),
        v("mean_thickness_mm", f.mean_thickness_mm, "mm", "volume / one-side area"),
        v("part_perimeter_m", f.trim_perimeter_mm / 1000, "m", "edge-face area / thickness"),
        v("hole_count", float(f.hole_count), "adet", "full concave cylinders on the part"),
    ]
