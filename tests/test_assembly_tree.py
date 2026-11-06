"""Product tree of an assembly STEP: names, counts, solid indices, standard-part grouping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from teklif.geometry.assembly_tree import is_standard


def test_standard_part_detection() -> None:
    assert is_standard("ST0003216", "ST0003216")
    assert is_standard("TP1", "Cıvata M8x20")
    assert is_standard("TP1", "Hex bolt")
    assert not is_standard("TP0007905", "Plaka")
    assert not is_standard("TP0103677", "Braket")


def _write_assembly(path: Path) -> None:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def named(label: Any, name: str) -> None:
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))

    def at(x: float) -> Any:
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(x, 0, 0))
        return TopLoc_Location(t)

    plate = tool.AddShape(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100, 100, 4).Shape(), False)
    named(plate, "PLATE1")
    bolt = tool.AddShape(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 5, 5, 20).Shape(), False)
    named(bolt, "ST0001")
    top = tool.NewShape()
    named(top, "TOP")
    named(tool.AddComponent(top, plate, at(0)), "PLATE1.1")
    for i in range(3):
        named(tool.AddComponent(top, bolt, at(200 + 20 * i)), f"ST0001.{i + 1}")
    tool.UpdateAssemblies()
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    assert w.Transfer(doc)
    assert w.Write(str(path)) == IFSelect_RetDone


def test_read_assembly_groups_parts_and_matches_solids(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from teklif.geometry.assembly_tree import read_assembly
    from teklif.ingest.step_props import extract_step_props

    path = tmp_path / "asm.step"
    _write_assembly(path)
    tree = read_assembly(path)
    assert tree.is_assembly and tree.n_solids == 4 and tree.unmatched == 0
    by = {g.product: g for g in tree.parts}
    assert set(by) == {"PLATE1", "ST0001"}
    assert by["PLATE1"].count == 1 and not by["PLATE1"].standard
    assert by["ST0001"].count == 3 and by["ST0001"].standard
    assert sorted(i for inst in by["ST0001"].instances for i in inst.solids) == sorted(
        set(range(4)) - set(by["PLATE1"].solids)
    )
    assert tree.parts[0].product == "PLATE1"  # special parts first
    whole = extract_step_props(path)
    one = extract_step_props(path, by["PLATE1"].solids)
    vol_one = {v.key: v.value or 0.0 for v in one.values}["volume_m3"]
    vol_all = {v.key: v.value or 0.0 for v in whole.values}["volume_m3"]
    assert vol_one == pytest.approx(100 * 100 * 4 / 1e9)
    assert vol_all > vol_one
