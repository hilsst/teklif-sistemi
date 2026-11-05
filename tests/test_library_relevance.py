from pathlib import Path

import pytest

from teklif.app.library_relevance import is_needed, needed_paths
from teklif.app.surface_panel import mold_submesh
from teklif.library import load_library
from teklif.library.editor import editable_values

LIB = Path(__file__).resolve().parents[1] / "library"


def test_needed_is_a_strict_subset_of_todo() -> None:
    lib = load_library(LIB)
    needed = needed_paths(lib, "sandvic_toplu")
    todo = [v for v in editable_values(lib) if v.value is None]
    shown = [v for v in todo if is_needed(v, needed)]
    assert shown
    assert len(shown) < len(todo)
    assert any(v.path.endswith("hourly_rate") for v in shown)


def test_mold_submesh_renumbers() -> None:
    import numpy as np

    from teklif.geometry.surface_select import FaceMesh, MeshFace

    def face(i: int) -> MeshFace:
        return MeshFace(
            i,
            1.0,
            True,
            np.zeros((3, 3)),
            np.array([[0, 1, 2]]),
            np.array([[0, 0, 1.0]]),
            np.array([0.5]),
        )

    sub = mold_submesh(FaceMesh([face(0), face(1), face(2)]), [2])
    assert [f.id for f in sub.faces] == [0]


def test_outline_has_eop_loops_for_holes_and_flange_only_on_outer_rim() -> None:
    import numpy as np

    from teklif.geometry.surface_select import FaceMesh, MeshFace

    def square(x0: float, y0: float, w: float, h: float) -> list[np.ndarray]:
        c = np.array([[x0, y0, 0], [x0 + w, y0, 0], [x0 + w, y0 + h, 0], [x0, y0 + h, 0]], float)
        return [np.array([c[i], c[(i + 1) % 4]]) for i in range(4)]

    polys = {k: p for k, p in enumerate(square(0, 0, 100, 50) + square(40, 20, 10, 10))}
    v = np.array([[0, 0, 0], [100, 0, 0], [100, 50, 0], [0, 50, 0]], dtype=float)
    f = MeshFace(
        0,
        5000.0,
        True,
        v,
        np.array([[0, 1, 2], [0, 2, 3]]),
        np.tile([0, 0, 1.0], (2, 1)),
        np.array([2500.0, 2500.0]),
        edges=[(k, 10.0) for k in polys],
        edge_polys=polys,
    )
    ol = FaceMesh([f]).outline([0], np.array([0.0, 0.0, 1.0]))
    assert ol is not None
    assert len(ol.loops) == 2 and len(ol.loops[ol.outer]) == 4  # rim + the hole
    assert ol.ring is not None and ol.tangent is not None
    # flat sheet: the surface runs on outward in the plane
    assert np.allclose(np.abs(ol.tangent[:, 2]), 0.0, atol=1e-6)
    from teklif.app.surface_panel import ToolProfile, profile_rings

    r = profile_rings(ol, np.array([0.0, 0.0, 1.0]), ToolProfile(20.0, 40.0, 150.0))
    assert r is not None
    # MEOP = EOP + 20 mm along the tangent, turn = +40 mm more, flange perpendicular to pull
    assert np.allclose(np.linalg.norm(r["meop"] - r["eop"], axis=1), 20.0)
    assert np.allclose(np.linalg.norm(r["turn"] - r["meop"], axis=1), 40.0)
    flange_vec = r["edge"] - r["turn"]
    assert np.allclose(flange_vec @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-6)  # flange ⟂ pull
    assert np.allclose(np.linalg.norm(flange_vec, axis=1), 150.0)


def test_triangulate_concave_polygon_covers_its_area() -> None:
    import numpy as np

    from teklif.geometry.surface_select import triangulate_polygon

    # L shape, area 3
    xy = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    tris = triangulate_polygon(xy)
    area = sum(
        abs(
            (xy[b][0] - xy[a][0]) * (xy[c][1] - xy[a][1])
            - (xy[b][1] - xy[a][1]) * (xy[c][0] - xy[a][0])
        )
        / 2
        for a, b, c in tris
    )
    assert len(tris) == 4 and np.isclose(area, 3.0)


def test_chain_loops_by_vertex_keys_and_open_chains() -> None:
    import numpy as np

    from teklif.geometry.surface_select import _chain_loops

    def seg(a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
        return np.array([[*a, 0.0], [*b, 0.0]])

    # a triangle given with one reversed edge (vertex keys 1,2,3) + a loose open path (7-8-9)
    segs = [
        seg((0, 0), (1, 0)),
        seg((0, 1), (1, 0)),
        seg((0, 1), (0, 0)),
        seg((5, 0), (6, 0)),
        seg((6, 0), (7, 0)),
    ]
    ends: list[tuple[object, object]] = [(1, 2), (3, 2), (3, 1), (7, 8), (8, 9)]
    closed, opened = _chain_loops(segs, ends)
    assert len(closed) == 1 and len(closed[0]) == 3
    assert len(opened) == 1 and len(opened[0]) == 4


def test_tube_profile_continues_the_wall_towards_the_opening(tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("OCP")
    import numpy as np
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

    # open-ended wall parallel to the pull: radius 100, x from 0 to 80
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from teklif.app.surface_panel import ToolProfile, profile_rings
    from teklif.geometry.surface_select import load_face_mesh

    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), 100.0, 80.0).Shape()
    w = STEPControl_Writer()
    w.Transfer(cyl, STEPControl_AsIs)
    path = tmp_path / "tube.step"
    w.Write(str(path))
    m = load_face_mesh(path, 5.0, 10.0, True)
    lateral = max(m.faces, key=lambda f: f.area_mm2).id
    pull = np.array(
        [1.0, 0.0, 0.0]
    )  # walls: no draft; the opening is at x = 0 (smallest along pull)
    ol = m.outline([lateral], pull)
    assert ol is not None and ol.ring is not None and ol.tangent is not None
    assert np.allclose(ol.ring[:, 0], 0.0, atol=1e-3)  # EOP at the opening rim
    assert np.allclose(ol.tangent @ pull, -1.0, atol=0.02)  # the wall goes on past the rim
    r = profile_rings(ol, pull, ToolProfile(20.0, 40.0, 150.0))
    assert r is not None
    assert np.allclose(r["meop"][:, 0], -20.0, atol=0.5) and np.allclose(
        r["turn"][:, 0], -60.0, atol=1
    )
    radial = np.hypot(r["edge"][:, 1], r["edge"][:, 2]) - np.hypot(r["turn"][:, 1], r["turn"][:, 2])
    assert np.allclose(radial, 150.0, atol=2.0)  # the flange turns 90° and runs out radially


def test_cylinder_wall_draft_is_exactly_zero(tmp_path: Path) -> None:
    """A coarsely tessellated cylinder wall must read 0 deg, not +-1 deg from facet tilt."""
    pytest.importorskip("OCP")
    import numpy as np
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from teklif.geometry.surface_select import draft_angles_deg, load_face_mesh

    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), 100.0, 80.0).Shape()
    w = STEPControl_Writer()
    w.Transfer(cyl, STEPControl_AsIs)
    path = tmp_path / "cyl.step"
    w.Write(str(path))
    mesh = load_face_mesh(path, 5.0, 10.0)
    side = max(mesh.faces, key=lambda f: f.area_mm2).id
    ang = draft_angles_deg(mesh, [side], np.array([1.0, 0.0, 0.0]))[side]
    assert abs(ang).max() < 1e-3


def test_trim_length_uses_the_longer_edge_of_the_wall(tmp_path: Path) -> None:
    """Selecting the inner face of a tube: the trim runs along the longer outer edge."""
    pytest.importorskip("OCP")
    import numpy as np
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from teklif.geometry.surface_select import load_face_mesh

    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    outer = BRepPrimAPI_MakeCylinder(ax, 100.0, 80.0).Shape()
    inner = BRepPrimAPI_MakeCylinder(ax, 95.0, 80.0).Shape()
    w = STEPControl_Writer()
    w.Transfer(BRepAlgoAPI_Cut(outer, inner).Shape(), STEPControl_AsIs)
    path = tmp_path / "tube.step"
    w.Write(str(path))
    m = load_face_mesh(path, 1.0, 5.0, True)
    curved = sorted((f for f in m.faces if not f.planar), key=lambda f: f.area_mm2)
    inner_face = curved[0].id  # smaller radius
    boundary, outer_len = m.trim_lengths_mm([inner_face], np.array([1.0, 0.0, 0.0]))
    assert abs(outer_len - 2 * np.pi * 100) < 4.0  # outer circle, not the 2*pi*95 inner one
    assert boundary > 2 * (2 * np.pi * 95)
