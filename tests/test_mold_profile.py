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


def _runs(p: np.ndarray, tol: float = 1e-6) -> list[list[int]]:
    """Runs (index lists, around the closed ring) of consecutive points at one place."""
    n = len(p)
    same = [float(np.linalg.norm(p[(i + 1) % n] - p[i])) <= tol for i in range(n)]
    if all(same):
        return [list(range(n))]
    start = (same.index(False) + 1) % n
    out: list[list[int]] = [[start]]
    for k in range(1, n):
        i = (start + k) % n
        if same[(i - 1) % n]:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _check_envelope(ol: Outline, r: dict[str, np.ndarray]) -> None:
    """Flat part: every MEOP / turn point lies ON the offset envelope of the EOP (20 / 60 mm from
    the EOP polyline: never closer, and never further, i.e. not collapsed onto a far-away point),
    the flange edge 150 mm beyond the turn line (at most the small mitre of a gentle plan corner
    further), and the continuation points outward of the part."""
    assert ol.ring is not None and ol.tangent is not None
    eop, meop, turn, edge = r["eop"], r["meop"], r["turn"], r["edge"]
    for line, dist in ((meop, 20.0), (turn, 60.0)):
        d = polyline_distance(line, eop)
        assert d.min() >= dist - 1e-3, (dist, int(np.argmin(d)), d.min())
        assert d.max() <= dist + 0.05, (dist, int(np.argmax(d)), d.max())
    fl = polyline_distance(_plan(edge), _plan(turn))
    assert fl.min() >= 150.0 - 1e-3 and fl.max() <= 150.0 + 1.0, (fl.min(), fl.max())
    # a run of one turn point is either a rounded corner's fan copies (same EOP point, the turn on
    # the 60 mm arc about it) or a sharp corner of the turn line itself (a concave corner, a step
    # shorter than the offset): then the point is a corner of the envelope, >= 60 mm from every
    # EOP point of the run and exactly 60 mm from the EOP polyline (checked above)
    for run in _runs(turn):
        if len(run) <= 3:
            continue
        gap = np.linalg.norm(eop[run] - turn[run[0]], axis=1)
        assert gap.min() >= 60.0 - 1e-3, (run, gap.min())
        if np.allclose(eop[run], eop[run[0]]):
            assert np.allclose(gap, 60.0, atol=1e-3), (run, gap)
    # the surface goes on outward: in plan the continuation never points into the part
    p2 = _plan(eop)
    outer = _dedup(p2)
    probe = p2 + 0.5 * _plan(ol.tangent) / np.maximum(
        np.linalg.norm(_plan(ol.tangent), axis=1, keepdims=True), 1e-12
    )
    assert not any(_inside(q, outer) for q in probe)


# --- synthetic parts -----------------------------------------------------------------------


def _shapes() -> Any:
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    class S:
        @staticmethod
        def plate(pts: list[tuple[float, float]], h: float = 4.0) -> Any:
            """A flat plate (0 <= z <= h) with the given polygon outline."""
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
            from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
            from OCP.gp import gp_Vec

            poly = BRepBuilderAPI_MakePolygon()
            for x, y in pts:
                poly.Add(gp_Pnt(x, y, 0.0))
            poly.Close()
            face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
            return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, h)).Shape()

        @staticmethod
        def fillet_vertical(
            shape: Any, r: float, radii: dict[tuple[float, float], float] | None = None
        ) -> Any:
            """Every edge running along z rounded with radius r, or radii[(x, y)] for the one at
            x, y (a sheet's rounded plan corners; their trim edge is sampled far coarser than
            the face mesh along it)."""
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
            from OCP.TopAbs import TopAbs_EDGE
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import TopoDS

            cast = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
            mk = BRepFilletAPI_MakeFillet(shape)
            ex = TopExp_Explorer(shape, TopAbs_EDGE)
            seen: set[tuple[float, float]] = set()
            while ex.More():
                e = cast(ex.Current())
                ex.Next()
                c = BRepAdaptor_Curve(e)
                p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
                key = (round(p0.X(), 3), round(p0.Y(), 3))
                vertical = abs(p0.X() - p1.X()) < 1e-6 and abs(p0.Y() - p1.Y()) < 1e-6
                if vertical and key not in seen:
                    seen.add(key)
                    mk.Add((radii or {}).get(key, r), e)
            mk.Build()
            return mk.Shape()

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


def _top(mesh: FaceMesh, z: float) -> list[int]:
    return [f.id for f in mesh.faces if float(f.vertices[:, 2].min()) > z - 0.1]


def test_short_edge_between_two_rounded_convex_corners(tmp_path: Path) -> None:
    """A corner cut off by a ~20 mm edge, all plan corners rounded (r 8): two convex corners
    closer than the offsets. The trim edge of each round is sampled as a coarse chord while the
    face mesh follows the arc finely; the continuation used to flip into the part on such chords,
    and the turn line collapsed onto one point ~76 mm out (Kemal's TP0007903)."""
    s = _shapes()
    shape = s.fillet_vertical(s.plate([(0, 0), (586, 0), (600, 14), (600, 400), (0, 400)]), 8.0)
    m = _mesh(shape, tmp_path / "cut_corner.step")
    ol, r = _profile(m, _top(m, 4.0))
    _check_profile(ol, r, flat_part=True)
    _check_envelope(ol, r)
    # the turn line runs round the cut corner on the 60 mm envelope, well spread (no collapse)
    turn = r["turn"]
    corner = (turn[:, 0] > 600) & (turn[:, 1] < 0)
    assert len(np.unique(turn[corner].round(3), axis=0)) >= 8


@pytest.mark.parametrize(
    "outline",
    [
        # a 10 mm step in the top side: a convex and a concave corner 10 mm apart
        [(0, 0), (600, 0), (600, 400), (320, 400), (320, 390), (0, 390)],
        # a notch 12 mm wide, 8 mm deep: far narrower than the offsets
        [(0, 0), (600, 0), (600, 400), (306, 400), (306, 392), (294, 392), (294, 400), (0, 400)],
    ],
    ids=["step", "notch"],
)
def test_small_concave_step_and_notch(tmp_path: Path, outline: list[tuple[float, float]]) -> None:
    """Small steps tighter than the offsets: the turn line keeps the envelope of the corner arcs
    and the far edge (a sharp corner of its own), never a point off the envelope."""
    s = _shapes()
    shape = s.fillet_vertical(s.plate(outline), 3.0)
    m = _mesh(shape, tmp_path / "step.step")
    ol, r = _profile(m, _top(m, 4.0))
    _check_profile(ol, r, flat_part=True)
    _check_envelope(ol, r)


def test_narrow_bent_strip(tmp_path: Path) -> None:
    """An L strip 3 mm wide bent round concentric corners (r 6.5 outside, 3.5 inside): the chord
    of the outer round lies nearer to the inner side's mesh edges than to its own arc, so the
    owning mesh edge is looked up near the chord's ends, not at its middle."""
    s = _shapes()
    strip = s.plate([(0, 0), (3, 0), (3, 97), (100, 97), (100, 100), (0, 100)])
    shape = s.fillet_vertical(strip, 1.0, {(0.0, 100.0): 6.5, (3.0, 97.0): 3.5})
    m = _mesh(shape, tmp_path / "strip.step")
    ol, r = _profile(m, _top(m, 4.0))
    _check_profile(ol, r, flat_part=True)
    _check_envelope(ol, r)


KEMAL_TP0007903 = Path(
    "/root/.claude/uploads/c6d37376-1199-5972-b87c-d02a4cb92560/b29c037f-TP0007903.stp"
)


@pytest.fixture(scope="module")
def kemal_mesh() -> FaceMesh:
    if not KEMAL_TP0007903.exists():
        pytest.skip("Kemal's TP0007903 STEP not present")
    pytest.importorskip("OCP")
    from teklif.geometry.surface_select import load_face_mesh

    return load_face_mesh(KEMAL_TP0007903, 5.0, 10.0, True)


@pytest.mark.parametrize(
    ("solid", "facing"),
    [(3, 1.0), (3, -1.0), (2, 1.0), (2, -1.0)],
    ids=["solid3-top", "solid3-bottom", "solid2-top", "solid2-bottom"],
)
def test_kemal_tp0007903(kemal_mesh: FaceMesh, solid: int, facing: float) -> None:
    """Kemal's real part (18 plates, pull -Z): solid 3 is the plate of his screenshot (rounded
    corners ~21 mm apart, a jog), solid 2 a 3 mm wide bent strip on a not-quite-flat plane."""
    from teklif.app.surface_panel import ToolProfile, profile_rings
    from teklif.geometry.surface_select import faces_facing

    m = kemal_mesh
    up = set(faces_facing(m, np.array([0.0, 0.0, facing]), 30.0))
    ids = [i for i in up if m.faces[i].solid == solid]
    t0 = time.perf_counter()
    ol = m.outline(ids, PULL)
    assert ol is not None
    r = profile_rings(ol, PULL, ToolProfile(20.0, 40.0, 150.0))
    assert r is not None
    assert time.perf_counter() - t0 < 1.0
    _check_profile(ol, r, flat_part=True)
    _check_envelope(ol, r)


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


def test_nearest_segment_is_exact_with_mixed_segment_lengths() -> None:
    """Long segments have very small lower bounds; the true nearest short one must still win."""
    from teklif.geometry import mold_profile as mp

    rng = np.random.default_rng(3)
    n = 200
    a = rng.uniform(-500, 500, (n, 3))
    lens = rng.choice([1.0, 5.0, 300.0], n)
    d = rng.normal(size=(n, 3))
    b = a + d / np.linalg.norm(d, axis=1, keepdims=True) * lens[:, None]
    q = rng.uniform(-500, 500, (60, 3))
    idx, dist = mp.nearest_segment(q, a, b)
    ab = b - a
    for i, p in enumerate(q):
        t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        brute = np.sqrt(((p - a - t[:, None] * ab) ** 2).sum(1))
        assert abs(dist[i] - brute.min()) < 1e-9
        assert abs(brute[idx[i]] - brute.min()) < 1e-9


def test_serrated_rim_ring_stays_small(tmp_path: Path) -> None:
    """A rim with hundreds of small teeth is thinned so the offset clean-up stays fast."""
    import time

    pytest.importorskip("OCP")
    from teklif.geometry import mold_profile as mp

    t = np.linspace(0, 2 * np.pi, 4001)[:-1]
    r = 1000 + 3 * (np.arange(len(t)) % 2)
    loop = np.c_[r * np.cos(t), r * np.sin(t), np.zeros(len(t))]
    start = time.time()
    rim = mp.decimate_loop(loop, 60, 160.0)
    assert len(rim) <= 200
    assert time.time() - start < 2.0
