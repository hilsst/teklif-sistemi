"""Pull-direction search and the single / core / split / split+core decision (no OCP needed)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from teklif.draft.pull_search import (
    RegionKind,
    Settings,
    Solution,
    TriSet,
    assess,
    best_direction,
    components,
    pull_angles_deg,
    search_directions,
    settings_from_thresholds,
)

Pt = Sequence[float]
Quad = tuple[Pt, Pt, Pt, Pt, Pt]


def _soup(quads: list[Quad], rot_deg: float = 0.0) -> TriSet:
    """Triangles of quads (a, b, c, d, outward normal), split into an n x n grid so regions have
    real adjacency; optionally the whole part rotated about X."""
    verts: list[np.ndarray] = []
    tris: list[list[int]] = []
    norms: list[np.ndarray] = []
    for a, b, c, d, nrm in quads:
        pa, pb, pc, pd = (np.array(p, dtype=float) for p in (a, b, c, d))
        n = 4
        grid = [
            [
                pa + (pb - pa) * i / n + (pd - pa) * j / n + (pc - pb - pd + pa) * i * j / n**2
                for i in range(n + 1)
            ]
            for j in range(n + 1)
        ]
        off = len(verts)
        verts += [p for row in grid for p in row]
        for j in range(n):
            for i in range(n):
                v0 = off + j * (n + 1) + i
                for t in ([v0, v0 + 1, v0 + n + 2], [v0, v0 + n + 2, v0 + n + 1]):
                    tris.append(t)
                    norms.append(np.array(nrm, dtype=float) / np.linalg.norm(nrm))
    v = np.array(verts)
    nn = np.array(norms)
    if rot_deg:
        r = math.radians(rot_deg)
        rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        v, nn = v @ rx.T, nn @ rx.T
    return TriSet.from_triangles(v, np.array(tris), nn)


def _tray(flare_deg: float = 5.0) -> list[Quad]:
    """Outer (tool-side) surface of an open tray lying in a female tool: base 200 x 200, walls
    50 high flaring outwards (positive draft for +Z)."""
    t = 50 * math.tan(math.radians(flare_deg))
    s, c = math.sin(math.radians(flare_deg)), math.cos(math.radians(flare_deg))
    q: list[Quad] = [([0, 0, 0], [200, 0, 0], [200, 200, 0], [0, 200, 0], [0, 0, -1])]
    q.append(([0, 0, 0], [0, 200, 0], [-t, 200, 50], [-t, 0, 50], [-c, 0, -s]))
    q.append(([200, 0, 0], [200, 200, 0], [200 + t, 200, 50], [200 + t, 0, 50], [c, 0, -s]))
    q.append(([0, 0, 0], [200, 0, 0], [200, -t, 50], [0, -t, 50], [0, -c, -s]))
    q.append(([0, 200, 0], [200, 200, 0], [200, 200 + t, 50], [0, 200 + t, 50], [0, c, -s]))
    return q


def _hook(x: float, side: float, size: float = 20.0) -> Quad:
    """A small separate patch leaning back over the part (negative draft for +Z)."""
    a = math.radians(10)
    return (
        [x, 80, 10],
        [x, 80 + size, 10],
        [x, 80 + size, 10 + size],
        [x, 80, 10 + size],
        [side * math.cos(a), 0, math.sin(a)],
    )


def _dovetail(lean_deg: float = 10.0) -> list[Quad]:
    """U channel whose two walls lean inwards: undercut on both sides (needs a split)."""
    t = 100 * math.tan(math.radians(lean_deg))
    s, c = math.sin(math.radians(lean_deg)), math.cos(math.radians(lean_deg))
    e = 100 * math.tan(math.radians(5))  # closed ends flaring out 5 deg (no axial pull)
    es, ec = math.sin(math.radians(5)), math.cos(math.radians(5))
    return [
        ([-50, 0, 0], [50, 0, 0], [50, 300, 0], [-50, 300, 0], [0, 0, -1]),
        ([-50, 0, 0], [-50, 300, 0], [-50 + t, 300, 100], [-50 + t, 0, 100], [-c, 0, s]),
        ([50, 0, 0], [50, 300, 0], [50 - t, 300, 100], [50 - t, 0, 100], [c, 0, s]),
        ([-50, 0, 0], [50, 0, 0], [50 - t, -e, 100], [-50 + t, -e, 100], [0, -ec, -es]),
        (
            [-50, 300, 0],
            [50, 300, 0],
            [50 - t, 300 + e, 100],
            [-50 + t, 300 + e, 100],
            [0, ec, -es],
        ),
    ]


S = Settings()


def test_adjacency_groups_connected_triangles() -> None:
    ts = _soup(_tray())
    lab = components(ts, np.ones(len(ts.areas), dtype=bool))
    assert len(np.unique(lab)) == 1  # shared edges join base and walls


def test_tray_best_pull_is_axis_and_single_piece() -> None:
    ts = _soup(_tray())
    best = best_direction(ts.normals, ts.areas, S)
    assert best.undercut_ratio == 0
    assert np.dot(best.direction, [0, 0, 1]) > math.cos(math.radians(0.3))
    assert best.min_draft_deg == pytest.approx(5.0, abs=0.3)
    a = assess(ts, np.array(best.direction), S)
    assert a.solution is Solution.SINGLE
    assert not a.flags
    assert any(t.key == "tooling.decision" for t in a.trace)


def test_search_recovers_a_tilted_part() -> None:
    ts = _soup(_tray(), rot_deg=20.0)
    best = best_direction(ts.normals, ts.areas, S)
    want = np.array([0, -math.sin(math.radians(20)), math.cos(math.radians(20))])
    assert np.degrees(np.arccos(np.clip(np.dot(best.direction, want), -1, 1))) < 0.5
    assert pull_angles_deg(best.direction)["Z"] == pytest.approx(20.0, abs=0.5)


def test_search_finds_a_direction_that_avoids_the_undercut() -> None:
    # 2 deg back-lean on one wall: +Z locks it, a tilted pull releases everything
    t = 50 * math.tan(math.radians(2))
    s, c = math.sin(math.radians(2)), math.cos(math.radians(2))
    q = _tray(5.0)
    q[1] = ([0, 0, 0], [0, 200, 0], [t, 200, 50], [t, 0, 50], [-c, 0, s])
    ts = _soup(q)
    assert assess(ts, np.array([0, 0, 1.0]), S).solution is not Solution.SINGLE
    best = best_direction(ts.normals, ts.areas, S)
    assert best.undercut_ratio == 0
    assert assess(ts, np.array(best.direction), S).solution is Solution.SINGLE


def test_small_local_undercut_is_a_core() -> None:
    ts = _soup([*_tray(), _hook(-30, -1)])
    best = best_direction(ts.normals, ts.areas, S)
    a = assess(ts, np.array(best.direction), S)
    assert a.solution is Solution.CORE
    assert a.n_cores == 1
    assert [r.kind for r in a.regions] == [RegionKind.CORE]


def test_two_sided_undercut_is_a_two_piece_split() -> None:
    ts = _soup(_dovetail())
    a = assess(ts, np.array([0, 0, 1.0]), S)
    assert a.solution is Solution.SPLIT
    assert a.plan is not None and len(a.plan.pieces) == 2
    assert any(f.code == "split_parting_line" for f in a.flags)
    for p in a.plan.pieces:
        assert p.pull.undercut_ratio == 0


def test_runners_up_show_a_side_pull_against_the_split() -> None:
    # the closed dovetail slides out sideways with one end wall as a core, but most of it at 0 deg:
    # the search offers both and flags the low-draft one
    ts = _soup(_dovetail())
    found = search_directions(ts.normals, ts.areas, S)
    assert len(found) >= 2
    sols = {assess(ts, np.array(f.direction), S).solution for f in found}
    assert len(sols) >= 2
    side = assess(ts, np.array(found[0].direction), S)
    if side.pull.warn_ratio >= S.warn_review_ratio:
        assert any(f.code == "low_draft" for f in side.flags)


def test_split_and_core_together() -> None:
    # a small lip facing up and to +Y: the closed ends keep every piece from releasing it. Only a
    # pull grazing base, ends and lip at exactly 0 deg would, so this test asks for 0.5 deg draft
    h = math.radians(30)
    up_y = [0, math.sin(h), math.cos(h)]
    lip: Quad = ([-10, 140, 30], [10, 140, 30], [10, 160, 30], [-10, 160, 30], up_y)
    ts = _soup([*_dovetail(), lip])
    a = assess(ts, np.array([0, 0, 1.0]), Settings(warn_min_deg=0.5))
    assert a.solution is Solution.SPLIT_CORE
    assert a.n_cores == 1
    assert a.plan is not None and len(a.plan.pieces) == 2
    assert "ikisi birlikte" in a.label


def test_small_regions_on_opposite_sides_add_up_to_a_split() -> None:
    q = [*_tray(), _hook(-30, -1, 80.0), _hook(230, 1, 80.0)]
    ts = _soup(q)
    a = assess(ts, np.array([0, 0, 1.0]), S)
    each = [r.area_ratio for r in a.regions]
    assert all(r < S.core_max_area_ratio for r in each)
    assert sum(each) >= S.core_max_area_ratio
    assert all(r.kind is RegionKind.SPLIT for r in a.regions)
    assert a.solution in (Solution.SPLIT, Solution.SPLIT_CORE, Solution.REVIEW)


def test_closed_undercut_needs_review() -> None:
    # a region facing straight back over the part on most of the surface cannot be released
    roof: Quad = ([0, 0, 40], [200, 0, 40], [200, 200, 40], [0, 200, 40], [0, 0, 1])
    q = [*_tray(), roof]
    ts = _soup(q)
    a = assess(ts, np.array([0, 0, 1.0]), Settings(max_split_pieces=1))
    assert a.solution is Solution.REVIEW
    assert a.manual_review


def test_tiny_shallow_patch_is_noise_and_flagged() -> None:
    s, c = math.sin(math.radians(0.2)), math.cos(math.radians(0.2))
    patch: Quad = ([-40, 90, 10], [-40, 95, 10], [-40, 95, 15], [-40, 90, 15], [-c, 0, s])
    ts = _soup([*_tray(), patch])
    a = assess(ts, np.array([0, 0, 1.0]), S)
    assert a.solution is Solution.SINGLE
    assert [r.kind for r in a.regions] == [RegionKind.NOISE]
    assert any(f.code == "undercut_noise" for f in a.flags)


def test_settings_come_from_the_library() -> None:
    from teklif.library import load_library

    lib = load_library()
    s = settings_from_thresholds(lib.thresholds)
    assert s.core_max_area_ratio == lib.thresholds.undercut.core_max_area_ratio.value
    assert s.max_split_pieces == 3
    assert s.samples == int(lib.thresholds.pull_search.samples.value or 0)
