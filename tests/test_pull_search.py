"""Pull-direction search and the single / core / split / split+core decision (no OCP needed)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

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
    parting_walls,
    pull_angles_deg,
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


def _dovetail(lean_deg: float = 10.0, length: float = 100.0) -> list[Quad]:
    """Hood over a part: top plate and two side walls leaning inwards (undercut on both sides for
    +Z), closed by two end walls that flare outwards 5 deg. The ends are big enough that pulling
    along the hood's axis leaves a region too large for a core: only a split releases it."""
    t = 100 * math.tan(math.radians(lean_deg))
    s, c = math.sin(math.radians(lean_deg)), math.cos(math.radians(lean_deg))
    e = 100 * math.tan(math.radians(5))
    es, ec = math.sin(math.radians(5)), math.cos(math.radians(5))
    ln = length
    return [
        ([-50, 0, 0], [50, 0, 0], [50, ln, 0], [-50, ln, 0], [0, 0, -1]),
        ([-50, 0, 0], [-50, ln, 0], [-50 + t, ln, 100], [-50 + t, 0, 100], [-c, 0, s]),
        ([50, 0, 0], [50, ln, 0], [50 - t, ln, 100], [50 - t, 0, 100], [c, 0, s]),
        ([-50, 0, 0], [50, 0, 0], [50 - t, -e, 100], [-50 + t, -e, 100], [0, -ec, -es]),
        (
            [-50, ln, 0],
            [50, ln, 0],
            [50 - t, ln + e, 100],
            [-50 + t, ln + e, 100],
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
    # 2 deg back-lean on one wall: straight +Z locks it, a tilted pull releases everything
    t = 50 * math.tan(math.radians(2))
    s, c = math.sin(math.radians(2)), math.cos(math.radians(2))
    q = _tray(5.0)
    q[1] = ([0, 0, 0], [0, 200, 0], [t, 200, 50], [t, 0, 50], [-c, 0, s])
    ts = _soup(q)
    a = assess(ts, np.array([0, 0, 1.0]), S)  # the user looks at straight +Z
    assert a.pull.undercut_ratio > 0
    assert a.solution is Solution.SINGLE  # but the best single piece is a tilted pull
    assert a.chosen is not None
    best = a.chosen.pieces[0].pull
    assert best.undercut_ratio == 0
    assert pull_angles_deg(best.direction)["Z"] > 0.5


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
    ch = a.chosen
    assert ch is not None and ch.kind == "plane" and ch.n_pieces == 2 and ch.n_cores == 0
    assert a.plan is not None and len(a.plan.pieces) == 2
    assert any(f.code == "split_parting_line" for f in a.flags)
    n = np.array(ch.planes[0].normal)
    assert abs(n[0]) > 0.95  # the plane cuts the hood left from right
    p0, p1 = ch.pieces[0].pull, ch.pieces[1].pull
    for p in (p0, p1):
        assert p.undercut_ratio == 0
    # the pieces slide apart: the part moves +n relative to the negative-side piece and -n for
    # the positive-side one
    assert np.dot(p0.direction, n) >= -1e-6
    assert np.dot(p1.direction, n) <= 1e-6
    walls = parting_walls(ts, ch.piece_of, 20.0)
    assert len(walls) > 0
    assert np.abs(walls[:, :, 0]).max() < 60  # the wall stands on the cut near x = 0


def test_methods_are_ranked_by_cost_and_listed() -> None:
    ts = _soup(_dovetail())
    a = assess(ts, np.array([0, 0, 1.0]), S)
    costs = [m.cost for m in a.methods]
    assert costs == sorted(costs)
    kinds = {m.kind for m in a.methods}
    assert {"single", "plane"} <= kinds
    single = next(m for m in a.methods if m.kind == "single")
    assert single.cost > a.methods[0].cost
    assert any(t.key.startswith("method.") for t in a.trace)


def test_piece_limit_is_respected() -> None:
    ts = _soup(_dovetail())
    one = assess(ts, np.array([0, 0, 1.0]), replace(S, max_split_pieces=1))
    assert all(m.n_pieces == 1 for m in one.methods)
    assert one.solution is Solution.REVIEW  # nothing releases it as one piece
    two = assess(ts, np.array([0, 0, 1.0]), replace(S, max_split_pieces=2))
    assert all(m.n_pieces <= 2 for m in two.methods)
    assert two.solution is Solution.SPLIT


def test_lip_is_released_by_a_split_with_at_most_one_core() -> None:
    # a separate small lip facing up and to +Y next to the hood: still a split, never unresolved
    h = math.radians(30)
    up_y = [0, math.sin(h), math.cos(h)]
    lip: Quad = ([-10, 40, 30], [10, 40, 30], [10, 60, 30], [-10, 60, 30], up_y)
    ts = _soup([*_dovetail(), lip])
    a = assess(ts, np.array([0, 0, 1.0]), Settings(warn_min_deg=0.5, max_split_pieces=3))
    assert a.solution in (Solution.SPLIT, Solution.SPLIT_CORE)
    assert a.chosen is not None and a.chosen.n_unresolved == 0 and a.n_cores <= 1


def test_split_and_core_together_is_its_own_solution() -> None:
    ts = _soup(_dovetail())
    a = assess(ts, np.array([0, 0, 1.0]), S)
    ch = a.chosen
    assert ch is not None
    both = replace(ch, pieces=ch.pieces[:2], n_cores=1)
    assert both.solution is Solution.SPLIT_CORE
    assert replace(ch, n_cores=0).solution is Solution.SPLIT


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
    assert a.manual_review  # low draft everywhere or unresolved: the engineer must look


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


def test_view_describes_the_method_the_user_picked() -> None:
    ts = _soup(_dovetail())
    a = assess(ts, np.array([0, 0, 1.0]), S)
    assert a.recommended is a.methods[0] and a.shown == 0
    k = next(i for i, m in enumerate(a.methods) if m.kind == "single")
    v = a.view(k)
    assert v.chosen is a.methods[k] and v.recommended is a.methods[0]
    assert v.solution is a.methods[k].solution and v.solution is not a.solution
    assert v.n_cores == a.methods[k].n_cores and v.plan is None
    assert v.reasons[0].startswith("Bu yöntemi tablodan siz seçtiniz")
    assert not any(f.code == "split_parting_line" for f in v.flags)
    assert not any(f.code == "method_close" for f in v.flags)
    assert all(not al.startswith(a.methods[k].label + ":") for al in v.alternatives)
    assert v.methods is a.methods  # the comparison itself is unchanged
    assert a.view(0) is a and a.view(99) is a
    back = v.view(0)
    assert back.chosen is a.methods[0] and back.solution is a.solution
    assert any(f.code == "split_parting_line" for f in back.flags)


def test_tiny_shallow_undercuts_are_noise_not_cores() -> None:
    # a 10 x 10 mm patch leaning back by 3 deg on a 200 x 200 tray: far below the noise area
    # and shallower than noise_draft_deg, so it must not cost a core
    a = math.radians(3)
    sliver: Quad = (
        [90, 90, 10],
        [100, 90, 10],
        [100, 100, 20],
        [90, 100, 20],
        [math.cos(a), 0, math.sin(a)],
    )
    ts = _soup([*_tray(), sliver])
    out = assess(ts, np.array([0, 0, 1.0]), S)
    assert out.n_cores == 0
    assert [r.kind for r in out.regions] == [RegionKind.NOISE]
    assert any(f.code == "undercut_noise" for f in out.flags)


def test_big_cores_cost_more_than_small_ones() -> None:
    from teklif.draft.pull_search import method_cost

    small = method_cost(1, 2, 0, 0.0, S, core_share=0.01)
    big = method_cost(1, 2, 0, 0.0, S, core_share=0.09)
    assert big - small == pytest.approx(S.core_area_cost_weight * 0.08)
    assert method_cost(2, 0, 0, 0.0, S) < method_cost(1, 2, 0, 0.0, S, core_share=0.09)
