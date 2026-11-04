from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from teklif.geometry.plate_features import extract_plate_features


def _write(shape: object, path: Path) -> Path:
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(str(path))
    return path


def _plate_with_holes(path: Path, holes: int) -> Path:
    shape = BRepPrimAPI_MakeBox(400, 300, 3).Shape()
    for i in range(holes):
        ax = gp_Ax2(gp_Pnt(50 + i * 60, 100, -1), gp_Dir(0, 0, 1))
        shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(ax, 10, 5).Shape()).Shape()
    return _write(shape, path)


def test_plain_plate(tmp_path: Path) -> None:
    f = extract_plate_features(_plate_with_holes(tmp_path / "p.step", 0))
    assert f.one_side_area_m2 == pytest.approx(0.4 * 0.3, rel=1e-3)
    assert f.mean_thickness_mm == pytest.approx(3, rel=1e-2)
    assert f.trim_perimeter_mm == pytest.approx(1400, rel=0.02)
    assert f.hole_count == 0
    assert f.curved_area_fraction == pytest.approx(0)


def test_plate_with_holes(tmp_path: Path) -> None:
    f = extract_plate_features(_plate_with_holes(tmp_path / "h.step", 3))
    assert f.hole_count == 3
    assert f.holes[0].diameter_mm == pytest.approx(20)
    assert f.holes[0].count == 3
    assert f.trim_perimeter_mm == pytest.approx(1400, rel=0.03)
    assert f.one_side_area_m2 == pytest.approx(0.12 - 3 * 3.14159 * 1e-4, rel=1e-2)


def test_curved_shell_has_curved_fraction(tmp_path: Path) -> None:
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(ax, 100, 200, 1.5).Shape()  # 90 deg segment
    inner = BRepPrimAPI_MakeCylinder(ax, 97, 200, 1.5).Shape()
    f = extract_plate_features(_write(BRepAlgoAPI_Cut(outer, inner).Shape(), tmp_path / "c.step"))
    assert f.curved_area_fraction > 0.95
    assert f.mean_thickness_mm == pytest.approx(3, rel=0.05)


def test_step_values_are_derived_and_flagged(tmp_path: Path) -> None:
    from teklif.geometry.plate_features import plate_step_values

    vals = plate_step_values(extract_plate_features(_plate_with_holes(tmp_path / "v.step", 2)))
    assert {v.key for v in vals} == {
        "part_area_m2",
        "mean_thickness_mm",
        "hole_count",
    }
    assert all(v.source.value == "derived" and v.review_reason for v in vals)
