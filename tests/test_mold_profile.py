"""Mold profile (EOP -> MEOP -> flange turn -> flange) at awkward edges: concave cut-outs, a narrow
notch, a rim that jumps between the top plane and a skirt, and a top plane meeting a wall."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teklif.geometry.mold_profile import plane_frame, polyline_distance, untangle
from teklif.geometry.surface_select import FaceMesh, MeshFace, Outline

PULL = np.array([0.0, 0.0, -1.0])


# --- independent checks --------------------------------------------------------------------


def _dedup(xy: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = [xy[0]]
    for p in xy[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.array(keep)


def _self_crossings(xy: np.ndarray) -> list[tuple[int, int]]:
    """Proper crossings between non-adjacent segments of a closed 2D polyline (brute force)."""
    u = _dedup(xy)
    m = len(u)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    out = []
    for i in range(m):
        a, b = u[i], u[(i + 1) % m]
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            c, d = u[j], u[(j + 1) % m]
            scale = max(np.linalg.norm(b - a), np.linalg.norm(d - c), 1.0)
            eps = 1e-7 * scale * scale
            o1, o2 = orient(a, b, c), orient(a, b, d)
            o3, o4 = orient(c, d, a), orient(c, d, b)
            crossing = o1 * o2 < 0 and o3 * o4 < 0
            if crossing and min(abs(o1), abs(o2), abs(o3), abs(o4)) > eps:
                out.append((i, j))
    return out


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _plan(p: np.ndarray) -> np.ndarray:
    _, a, b = plane_frame(PULL)
    out: np.ndarray = np.c_[p @ a, p @ b]
    return out


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def _opposed_neighbour_quads(a: np.ndarray, b: np.ndarray) -> list[int]:
    """Strip quads (a[i], a[i+1], b[i+1], b[i]) whose normal points against the next quad's:
    the strip folds back over itself there."""
    n = len(a)
    normals: list[np.ndarray | None] = []
    for i in range(n):
        j = (i + 1) % n
        q = np.cross(b[j] - a[i], b[i] - a[j])
        ln = float(np.linalg.norm(q))
        normals.append(q / ln if ln > 1e-6 else None)
    bad = []
    for i in range(n):
        x, y = normals[i], normals[(i + 1) % n]
        if x is not None and y is not None and float(x @ y) < -0.2:
            bad.append(i)
    return bad


def _check_profile(ol: Outline, r: dict[str, np.ndarray], flat_part: bool) -> None:
    assert ol.ring is not None and ol.tangent is not None and ol.out_dir is not None
    eop, meop, turn, edge = r["eop"], r["meop"], r["turn"], r["edge"]
    assert len(eop) == len(meop) == len(turn) == len(edge)
    assert np.isfinite(np.vstack([meop, turn, edge])).all()
    # straight on (unit tangent, not trimmed at a corner): exactly 20 mm, then 40 mm more
    unit = np.isclose(np.linalg.norm(ol.tangent, axis=1), 1.0)
    plain_m = unit & np.isclose(meop, eop + 20.0 * ol.tangent).all(axis=1)
    plain_t = unit & np.isclose(turn, eop + 60.0 * ol.tangent).all(axis=1)
    assert plain_m.mean() > 0.6 and plain_t.mean() > 0.5
    assert np.allclose(np.linalg.norm(meop - eop, axis=1)[plain_m], 20.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(turn - meop, axis=1)[plain_m & plain_t], 40.0, atol=1e-6)
    # no line comes closer to the EOP than its distance (mitred and trimmed corners included)
    assert (polyline_distance(meop, eop) >= 20.0 - 1e-3).all()
    assert (polyline_distance(turn, eop) >= 60.0 - 1e-3).all()
    # flange: perpendicular to the pull everywhere, exactly 150 mm where it was not trimmed
    fl = edge - turn
    assert np.allclose(fl @ PULL, 0.0, atol=1e-6)
    od_unit = np.isclose(np.linalg.norm(ol.out_dir, axis=1), 1.0)
    plain_f = od_unit & np.isclose(edge, turn + 150.0 * ol.out_dir).all(axis=1)
    assert plain_f.mean() > 0.5
    assert np.allclose(np.linalg.norm(fl, axis=1)[plain_f], 150.0, atol=1e-6)
    # the flange edge never crosses itself, runs the same way round as the EOP, and encloses it
    e2, p2 = _plan(edge), _plan(eop)
    assert _self_crossings(e2) == []
    assert np.sign(_area(_dedup(e2))) == np.sign(_area(p2))
    outer = _dedup(e2)
    assert all(_inside(p, outer) for p in p2)
    assert (polyline_distance(e2, _plan(turn)) >= 150.0 - 1e-3).all()
    # no strip folds back over itself
    for a, b in ((eop, meop), (meop, turn), (turn, edge)):
        assert _opposed_neighbour_quads(a, b) == []
    if flat_part:  # everything in one plane: every strip quad turns the same way in plan
        s = np.sign(_area(p2))
        for a, b in ((eop, meop), (meop, turn), (turn, edge)):
            a2, b2 = _plan(a), _plan(b)
            for i in range(len(a2)):
                j = (i + 1) % len(a2)
                q = np.array([a2[i], a2[j], b2[j], b2[i]])
                assert s * _area(q) <= 1e-6, (i, _area(q))


# --- synthetic parts -----------------------------------------------------------------------


def _shapes() -> Any:
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    class S:
        @staticmethod
        def box(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> Any:
            return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()

        @staticmethod
        def cyl(x: float, y: float, z: float, r: float, h: float) -> Any:
            return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)), r, h).Shape()

        @staticmethod
        def cut(a: Any, b: Any) -> Any:
            return BRepAlgoAPI_Cut(a, b).Shape()

    return S


def _mesh(shape: Any, path: Path) -> FaceMesh:
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    from teklif.geometry.surface_select import load_face_mesh

    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(str(path))
    return load_face_mesh(path, 5.0, 10.0, True)


def _profile(mesh: FaceMesh, ids: list[int]) -> tuple[Outline, dict[str, np.ndarray]]:
    from teklif.app.surface_panel import ToolProfile, profile_rings

    ol = mesh.outline(ids, PULL)
    assert ol is not None
    r = profile_rings(ol, PULL, ToolProfile(20.0, 40.0, 150.0))
    assert r is not None
    return ol, r


def test_flat_plate_with_tight_arc_cutout_and_narrow_notch(tmp_path: Path) -> None:
    """Concave arc (r 120 < 210) and a 60 mm notch: the offsets would fold; they are trimmed."""
    s = _shapes()
    shape = s.cut(s.box(0, 0, 0, 600, 400, 5), s.cyl(300, 470, -1, 120, 60))
    shape = s.cut(shape, s.box(-1, 150, -1, 101, 60, 10))
    m = _mesh(shape, tmp_path / "flat.step")
    top = [f.id for f in m.faces if float(f.vertices[:, 2].min()) > 4.9]
    ol, r = _profile(m, top)
    assert ol.tangent is not None
    assert np.allclose(ol.tangent[:, 2], 0.0, atol=1e-6)  # a flat sheet goes on in its plane
    _check_profile(ol, r, flat_part=True)
    # the flange over the cut-out does not reach back across it: no edge point lies over the part
    e = r["edge"]
    assert not ((e[:, 0] > 1) & (e[:, 0] < 599) & (e[:, 1] > 1) & (e[:, 1] < 399)).any()


def test_tray_rim_switching_between_top_plane_and_skirt(tmp_path: Path) -> None:
    """Skirts with a gap (the rim climbs from the skirt bottom to the plate edge and back) and a
    concave arc cut through plate and skirt: no strip twists or folds."""
    s = _shapes()
    shape = s.cut(s.box(0, 0, 0, 600, 400, 40), s.box(5, 5, -1, 590, 390, 36))
    shape = s.cut(shape, s.box(-1, 100, -1, 7, 200, 36))
    shape = s.cut(shape, s.cyl(300, 470, -1, 120, 60))
    m = _mesh(shape, tmp_path / "tray.step")
    skin = [f.id for f in m.faces if float(f.vertices[:, 2].max()) > 39.0]
    ol, r = _profile(m, skin)
    _check_profile(ol, r, flat_part=False)
    assert ol.ring is not None and ol.tangent is not None
    # skirt bottoms (z = 0, away from the gap and the cut) go on straight down the wall
    eop = r["eop"]
    bottom = np.isclose(eop[:, 2], 0.0) & (np.isclose(eop[:, 1], 0.0) | np.isclose(eop[:, 0], 600))
    assert bottom.any()
    on_wall = bottom & np.isclose(np.linalg.norm(ol.tangent, axis=1), 1.0)
    assert np.allclose(ol.tangent[on_wall] @ PULL, 1.0, atol=1e-6)
    # at the gap the plate edge (z = 35) goes on down too; the flange there sits at 35 - 60
    gap = np.isclose(eop[:, 0], 0.0) & (eop[:, 1] > 110) & (eop[:, 1] < 290)
    assert np.allclose(r["turn"][gap, 2], -25.0, atol=1e-6)


def test_top_plane_meeting_a_wall(tmp_path: Path) -> None:
    """Top plane on three sides, a skirt on the fourth: the top goes on sideways, the skirt down,
    its end edges sideways; the corner is rounded, nothing crosses."""
    s = _shapes()
    shape = s.cut(s.box(0, 0, 0, 600, 400, 40), s.box(-5, 5, -1, 600, 390, 36))
    m = _mesh(shape, tmp_path / "topwall.step")
    ids = [
        f.id
        for f in m.faces
        if float(f.vertices[:, 2].min()) > 39.9 or float(f.vertices[:, 0].min()) > 599.9
    ]
    ol, r = _profile(m, ids)
    _check_profile(ol, r, flat_part=False)
    assert ol.tangent is not None
    eop, t = r["eop"], ol.tangent
    top_edge = np.isclose(eop[:, 2], 40.0) & (eop[:, 0] < 590)
    assert top_edge.any() and np.allclose(t[top_edge, 2], 0.0, atol=1e-6)
    # skirt bottom (a straight run keeps only its end points, each with its rounded corner):
    # the copies that go on along the wall point straight down the pull
    bottom = np.isclose(eop[:, 0], 600.0) & np.isclose(eop[:, 2], 0.0)
    skirt = bottom & np.isclose(t @ PULL, 1.0, atol=1e-6)
    assert skirt.sum() >= 2
    end = np.isclose(eop[:, 0], 600.0) & (eop[:, 2] > 1) & (eop[:, 2] < 39)
    assert np.allclose(t[end, 2], 0.0, atol=1e-6)  # the skirt's end edge: sideways, level
    # flange heights: at the top plane on the top sides, below the skirt on the skirt side
    assert np.allclose(r["edge"][top_edge, 2], 40.0) and np.allclose(r["edge"][skirt, 2], -60.0)


def test_untangle_cuts_off_loops_and_keeps_the_point_count() -> None:
    # a square whose top side folds back into a small loop (offset of a tight concave bite)
    ring = np.array(
        [
            [0, 0],
            [100, 0],
            [100, 100],
            [60, 100],
            [40, 120],
            [40, 80],
            [60, 80],
            [50, 100],
            [0, 100],
        ],
        dtype=float,
    )
    assert _self_crossings(ring) != []
    out = untangle(ring, 1.0)
    assert out.shape == ring.shape
    assert _self_crossings(out) == []
    assert np.isclose(abs(_area(_dedup(out))), 100 * 100, rtol=0.05)


def test_outline_profile_on_a_large_mesh_is_quick() -> None:
    """~100k triangles (a gently domed sheet): outline + rings in about a second."""
    n = 224
    u = np.linspace(0.0, 1000.0, n)
    x, y = np.meshgrid(u, u, indexing="ij")
    z = 50.0 * np.sin(np.pi * x / 1000) * np.sin(np.pi * y / 1000)
    v = np.c_[x.ravel(), y.ravel(), z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = (
        idx[:-1, :-1].ravel(),
        idx[1:, :-1].ravel(),
        idx[1:, 1:].ravel(),
        idx[:-1, 1:].ravel(),
    )
    tris = np.vstack([np.c_[a, b, c], np.c_[a, c, d]]).astype(np.int64)
    p0, p1, p2 = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(nrm, axis=1)
    nrm = nrm / area2[:, None]
    border = [idx[:, 0], idx[-1, :], idx[::-1, -1], idx[0, ::-1]]
    polys = {k: v[side] for k, side in enumerate(border)}
    face = MeshFace(
        0,
        float(area2.sum() / 2),
        False,
        v,
        tris,
        nrm,
        area2 / 2,
        edges=[(k, 1.0) for k in polys],
        edge_polys=polys,
    )
    assert len(tris) > 95_000
    from teklif.app.surface_panel import ToolProfile, profile_rings

    t0 = time.perf_counter()
    ol = FaceMesh([face]).outline([0], np.array([0.0, 0.0, 1.0]))
    assert ol is not None
    r = profile_rings(ol, np.array([0.0, 0.0, 1.0]), ToolProfile(20.0, 40.0, 150.0))
    assert r is not None
    assert time.perf_counter() - t0 < 4.0
