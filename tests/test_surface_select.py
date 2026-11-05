from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from teklif.geometry.surface_select import (
    draft_angles_deg,
    load_face_mesh,
    suggest_pull_directions,
)


def _write(shape: object, path: Path) -> Path:
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(str(path))
    return path


def _plate(tmp: Path, holes: int = 0) -> Path:
    shape = BRepPrimAPI_MakeBox(400, 300, 4).Shape()
    for i in range(holes):
        ax = gp_Ax2(gp_Pnt(60 + i * 80, 150, -1), gp_Dir(0, 0, 1))
        shape = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeCylinder(ax, 10, 6).Shape()).Shape()
    return _write(shape, tmp / "p.step")


def _bottom_face(mesh) -> int:  # type: ignore[no-untyped-def]
    # the largest face whose normal points to -Z (tool side of a plate lying on the tool)
    best = max((f for f in mesh.faces if f.mean_normal[2] < -0.9), key=lambda f: f.area_mm2)
    return int(best.id)


def test_plate_bottom_face_area_and_boundary(tmp_path: Path) -> None:
    mesh = load_face_mesh(_plate(tmp_path))
    fid = _bottom_face(mesh)
    assert mesh.area_m2([fid]) == pytest.approx(0.12, rel=1e-3)
    assert mesh.boundary_length_mm([fid]) == pytest.approx(1400, rel=1e-3)


def test_boundary_includes_holes(tmp_path: Path) -> None:
    mesh = load_face_mesh(_plate(tmp_path, holes=3))
    fid = _bottom_face(mesh)
    expected = 1400 + 3 * np.pi * 20
    assert mesh.boundary_length_mm([fid]) == pytest.approx(expected, rel=0.01)


def test_flat_tool_face_releases_in_plus_z_and_blocks_in_minus_z(tmp_path: Path) -> None:
    mesh = load_face_mesh(_plate(tmp_path))
    fid = _bottom_face(mesh)
    up = draft_angles_deg(mesh, [fid], np.array([0, 0, 1.0]))[fid]
    down = draft_angles_deg(mesh, [fid], np.array([0, 0, -1.0]))[fid]
    assert np.allclose(up, 90)
    assert np.allclose(down, -90)
    best = suggest_pull_directions(mesh, [fid], warn_min_deg=0.0, min_ok_deg=2.0)[0]
    assert best.direction == pytest.approx((0, 0, 1))
    assert best.undercut_area_ratio == 0.0


def test_tilted_wall_draft_is_the_tilt_angle() -> None:
    # synthetic: a wall tilted 5 deg from vertical, normal pointing out of the part
    from teklif.geometry.surface_select import FaceMesh, MeshFace

    ang = np.radians(5)
    n = np.array([np.cos(ang), 0.0, -np.sin(ang)])  # faces mostly +X, slightly down (tool side)
    face = MeshFace(
        id=0,
        area_mm2=1.0,
        planar=True,
        vertices=np.zeros((3, 3)),
        triangles=np.array([[0, 1, 2]]),
        normals=n[None, :],
        tri_areas=np.array([0.5]),
    )
    deg = draft_angles_deg(FaceMesh([face]), [0], np.array([0, 0, 1.0]))[0]
    assert deg[0] == pytest.approx(5.0)


def test_panel_helpers(tmp_path: Path) -> None:
    import numpy as np

    from teklif.app.surface_panel import GREEN, RED, build_figure, triangle_colors
    from teklif.geometry.surface_select import faces_facing

    mesh = load_face_mesh(_plate(tmp_path))
    bottom = faces_facing(mesh, np.array([0.0, 0.0, -1.0]), 10)
    assert bottom
    up = triangle_colors(mesh, bottom, (0.0, 0.0, 1.0), 2.0, 0.0)
    down = triangle_colors(mesh, bottom, (0.0, 0.0, -1.0), 2.0, 0.0)
    assert GREEN in up and RED in down
    assert len(build_figure(mesh, up).data) >= 2
