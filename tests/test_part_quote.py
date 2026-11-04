from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from teklif.app.part_quote import FLANGE_MM, measure, tool_footprint_mm


def _plate(tmp: Path) -> bytes:
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeBox(1200, 800, 4).Shape(), STEPControl_AsIs)
    path = tmp / "p.step"
    w.Write(str(path))
    return path.read_bytes()


def test_measure_uploaded_bytes(tmp_path: Path) -> None:
    m = measure(_plate(tmp_path), "p.step")
    assert m.features.one_side_area_m2 == pytest.approx(0.96, rel=1e-2)
    assert m.features.mean_thickness_mm == pytest.approx(4, rel=1e-2)


def test_footprint_adds_flange_on_both_sides(tmp_path: Path) -> None:
    m = measure(_plate(tmp_path), "p.step")
    assert tool_footprint_mm(m.props) == pytest.approx((1200 + 2 * FLANGE_MM, 800 + 2 * FLANGE_MM))
