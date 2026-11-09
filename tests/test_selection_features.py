"""Area, holes and thickness of the selected faces of a (multi-part) STEP file."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Builder, TopoDS_Compound

from teklif.geometry.selection_features import selection_features
from teklif.geometry.surface_select import load_face_mesh


def _plate_with_holes() -> object:
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 200, 100, 4).Shape()
    for x in (50, 150):
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, 50, -1), gp_Dir(0, 0, 1)), 8, 10).Shape()
        box = BRepAlgoAPI_Cut(box, cyl).Shape()
    return box


def _write(path: Path, *shapes: object) -> None:
    comp = TopoDS_Compound()
    b = TopoDS_Builder()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    w = STEPControl_Writer()
    w.Transfer(comp, STEPControl_AsIs)
    assert w.Write(str(path)) == IFSelect_RetDone


def test_selection_gets_its_own_area_holes_and_thickness(tmp_path: Path) -> None:
    thick = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, -40), 200, 100, 25).Shape()  # another part below
    path = tmp_path / "asm.step"
    _write(path, _plate_with_holes(), thick)
    mesh = load_face_mesh(path, with_edges=True)
    top = [f.id for f in mesh.faces if f.mean_normal[2] > 0.99 and f.vertices[:, 2].mean() > 1]
    assert top
    sf = selection_features(mesh, top)
    assert sf.area_m2 == pytest.approx((200 * 100 - 2 * 3.14159 * 64) / 1e6, rel=0.01)
    assert sf.hole_count == 2
    assert sf.mean_thickness_mm == pytest.approx(4.0, rel=0.05)
    assert len(sf.solids) == 1

    # the thick part below gets its own values
    big = [f.id for f in mesh.faces if f.mean_normal[2] > 0.99 and f.vertices[:, 2].mean() < -10]
    sb = selection_features(mesh, big)
    assert sb.hole_count == 0
    assert sb.mean_thickness_mm == pytest.approx(25.0, rel=0.05)
    assert sb.area_m2 == pytest.approx(0.02, rel=0.01)


def test_pocket_is_not_a_hole(tmp_path: Path) -> None:
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 200, 100, 10).Shape()
    pocket = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(100, 50, 6), gp_Dir(0, 0, 1)), 10, 10).Shape()
    path = tmp_path / "pocket.step"
    _write(path, BRepAlgoAPI_Cut(box, pocket).Shape())
    mesh = load_face_mesh(path, with_edges=True)
    top = [f.id for f in mesh.faces if f.mean_normal[2] > 0.99 and f.vertices[:, 2].mean() > 9]
    sf = selection_features(mesh, top)
    assert sf.hole_count == 0 and sf.n_inner_loops >= 1
    assert any("delik sayılmadı" in n for n in sf.notes)
