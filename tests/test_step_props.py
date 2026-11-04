from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from teklif.ingest.step_props import StepUnavailableError, extract_step_props


def _box_step(path: Path, x: float, y: float, z: float) -> Path:
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(x, y, z).Shape(), STEPControl_AsIs)
    writer.Write(str(path))
    return path


def test_box_properties(tmp_path: Path) -> None:
    props = extract_step_props(_box_step(tmp_path / "box.step", 100, 50, 10))
    assert props.solid_count == 1
    assert props.get("envelope_x_mm").value == pytest.approx(100)
    assert props.get("envelope_z_mm").value == pytest.approx(10)
    assert props.get("surface_area_m2").value == pytest.approx(0.013)
    assert props.get("volume_m3").value == pytest.approx(5e-5)
    assert all(v.source.value == "cad" for v in props.values)


def test_unreadable_file_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.step"
    bad.write_text("not a step file")
    with pytest.raises(StepUnavailableError):
        extract_step_props(bad)
