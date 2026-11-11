"""Best pull direction for a user-selected tool surface, undercut regions and the tooling decision.

Kemal (2026-11-09): after the STEP is loaded the system no longer builds a mold. It takes the tool
surface the user picked, searches the pull direction that releases it best (ideally one that needs
neither a split tool nor a loose core), and then says which solution the remaining undercuts need:
single-piece tool, loose core(s) (maça), split tool (1/2/3 pieces), or split tool AND cores.

Conventions (same as ``kalip_analiz.mesh``): triangle normals point out of the part,
i.e. into the tool; for a pull direction d the draft angle is ``asin(-n . d)``; positive releases,
below ``draft.warn_min_deg`` is an undercut.

Method (deterministic, every threshold from ``thresholds.yaml``):
1. Search: Fibonacci samples on the sphere inside a cone around the mean inward normal, plus the
   six axes; best few refined by a pattern search on the sphere. Order (lexicographic): smallest
   undercut area share, then largest share with draft >= ``min_ok_deg``, then largest robust
   minimum draft (so on a plateau of undercut-free directions the one with the most draft wins).
2. Undercut triangles are grouped into regions by edge adjacency.
3. Regions that are both tiny and shallow are tessellation noise (reported, not decided on).
   A region below ``undercut.core_max_area_ratio`` of the tool area is a loose core; larger
   regions, and small regions facing each other from opposite sides whose sum is not small, need
   a split tool.
4. Split plan: the parting line runs along the undercut boundaries; each piece gets its own best
   direction, adjacent pieces are merged while the merge does not add undercut. What a piece still
   cannot release is a core again (small) or unresolved (manual_review).

No ray casting: a lock always needs a negative-draft face, so regions are found from the draft
alone, but positive faces shadowed by an overhang are not added to the region area.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from kalip_analiz.settings import LIB, field_path
from kalip_analiz.settings import Settings as Settings
from kalip_analiz.settings import settings_from_thresholds as settings_from_thresholds
from kalip_analiz.trace import CostInput, CostLine, ReviewFlag, Source

if TYPE_CHECKING:
    from kalip_analiz.mesh import FaceMesh

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


# --- settings (thresholds live in kalip_analiz.settings / thresholds.yaml) ---------------


_UNITS = {
    "min_ok_deg": "deg",
    "warn_min_deg": "deg",
    "noise_draft_deg": "deg",
    "max_tilt_deg": "deg",
    "piece_max_tilt_deg": "deg",
    "sandwich_tol_deg": "deg",
    "refine_min_step_deg": "deg",
    "max_split_pieces": "count",
    "samples": "count",
    "plane_normals": "count",
    "exact_candidates": "count",
    "axis_splits": "count",
    "axis_mid_plane": "count",
}


def _inp(s: Settings, name: str) -> CostInput:
    return CostInput(
        name=name,
        value=float(getattr(s, name)),
        unit=_UNITS.get(name, "ratio"),
        source=s.source(name),
        path=f"{LIB}:{field_path(Settings, name)}",
    )


# --- triangle set ---------------------------------------------------------------------


@dataclass(frozen=True)
class TriSet:
    """The selected tool surface as a triangle soup with edge adjacency."""

    normals: F64  # (m, 3) unit, out of the part
    areas: F64  # (m,) mm2
    centroids: F64  # (m, 3)
    face: I64  # (m,) B-rep face id per triangle
    pairs: I64  # (k, 2) triangles that share an edge
    tri_vid: I64 | None = None  # (m, 3) merged vertex id per triangle corner
    vert_xyz: F64 | None = None  # (n, 3) coordinates of the merged vertices

    @property
    def total_area(self) -> float:
        return float(self.areas.sum())

    @classmethod
    def from_triangles(
        cls,
        vertices: F64,
        triangles: I64,
        normals: F64 | None = None,
        face: I64 | None = None,
        tol_mm: float = 1e-3,
    ) -> TriSet:
        v = np.asarray(vertices, dtype=float)
        t = np.asarray(triangles, dtype=np.int64)
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        cross = np.cross(b - a, c - a)
        lens = np.linalg.norm(cross, axis=1)
        if normals is None:
            normals = cross / np.where(lens > 0, lens, 1.0)[:, None]
        # vertices at the same place (faces carry their own copies of shared edge nodes)
        _, first, inv = np.unique(
            np.round(v / tol_mm), axis=0, return_index=True, return_inverse=True
        )
        vid = inv.ravel()[t]
        vert_xyz = v[first]
        e = np.sort(np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]], axis=1), axis=2)
        e = e.reshape(-1, 2)
        tri_of = np.repeat(np.arange(len(t)), 3)
        _, eid = np.unique(e, axis=0, return_inverse=True)
        eid = eid.ravel()
        order = np.argsort(eid, kind="stable")
        same = eid[order][1:] == eid[order][:-1]
        pairs = np.stack([tri_of[order][:-1][same], tri_of[order][1:][same]], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        return cls(
            normals=np.asarray(normals, dtype=float),
            areas=lens / 2,
            centroids=(a + b + c) / 3,
            face=np.zeros(len(t), dtype=np.int64) if face is None else np.asarray(face),
            pairs=pairs.astype(np.int64),
            tri_vid=vid.astype(np.int64),
            vert_xyz=vert_xyz,
        )


def triset_from_mesh(mesh: FaceMesh, ids: list[int]) -> TriSet:
    """Triangles of the selected faces of a tessellated STEP."""
    fs = [mesh.faces[i] for i in ids if len(mesh.faces[i].triangles)]
    if not fs:
        raise ValueError("no triangles in the selection")
    verts, tris, norms, owner, off = [], [], [], [], 0
    for f in fs:
        verts.append(f.vertices)
        tris.append(f.triangles + off)
        norms.append(f.normals)
        owner.append(np.full(len(f.triangles), f.id, dtype=np.int64))
        off += len(f.vertices)
    return TriSet.from_triangles(
        np.vstack(verts), np.vstack(tris), np.vstack(norms), np.concatenate(owner)
    )


def components(ts: TriSet, mask: NDArray[np.bool_]) -> I64:
    """Connected-component label per triangle (``-1`` outside ``mask``)."""
    lab = np.where(mask, np.arange(len(mask)), -1).astype(np.int64)
    p = ts.pairs[mask[ts.pairs[:, 0]] & mask[ts.pairs[:, 1]]]
    if len(p):
        while True:  # min-label propagation with pointer jumping
            old = lab.copy()
            m = np.minimum(lab[p[:, 0]], lab[p[:, 1]])
            np.minimum.at(lab, p[:, 0], m)
            np.minimum.at(lab, p[:, 1], m)
            idx = np.flatnonzero(mask)
            lab[idx] = lab[lab[idx]]
            if np.array_equal(lab, old):
                break
    return lab


# --- draft and scoring ----------------------------------------------------------------


def draft_deg(normals: F64, d: F64) -> F64:
    """Draft angle per triangle; rounded so a true 0 deg wall is not -1e-14 (red)."""
    return np.round(np.degrees(np.arcsin(np.clip(-(normals @ d), -1.0, 1.0))), 6) + 0.0


@dataclass(frozen=True)
class DirectionScore:
    direction: tuple[float, float, float]
    undercut_ratio: float  # area share with draft < warn_min_deg
    warn_ratio: float  # area share with warn_min_deg <= draft < min_ok_deg
    ok_ratio: float  # area share with draft >= min_ok_deg
    min_draft_deg: float  # robust minimum (robust_min_quantile of the triangles left out)

    def key(self) -> tuple[float, float, float]:
        return (round(self.undercut_ratio, 7), -round(self.ok_ratio, 7), -self.min_draft_deg)


def score(normals: F64, areas: F64, d: F64, s: Settings) -> DirectionScore:
    d = d / np.linalg.norm(d)
    dr = draft_deg(normals, d)
    tot = float(areas.sum()) or 1.0
    bad = float(areas[dr < s.warn_min_deg].sum()) / tot
    ok = float(areas[dr >= s.min_ok_deg].sum()) / tot
    k = min(len(dr) - 1, int(s.robust_min_quantile * len(dr)))
    lo = float(np.partition(dr, k)[k]) if len(dr) else 0.0
    warn = max(0.0, 1.0 - bad - ok)
    warn = 0.0 if warn < 1e-9 else warn
    return DirectionScore((float(d[0]), float(d[1]), float(d[2])), bad, warn, ok, lo)


def _fibonacci(n: int) -> F64:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5**0.5) * i
    out: F64 = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    return out


def _basis(d: F64) -> tuple[F64, F64]:
    a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(d, a)


def mean_inward(normals: F64, areas: F64) -> F64:
    m = -(normals * areas[:, None]).sum(axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


Constraint = tuple[F64, float, float]  # lo <= d . axis <= hi


def _allowed(v: F64, ref: F64, cos_lim: float, cons: Sequence[Constraint]) -> bool:
    if float(v @ ref) < cos_lim:
        return False
    return all(lo - 1e-9 <= float(v @ a) <= hi + 1e-9 for a, lo, hi in cons)


def search_directions(
    normals: F64,
    areas: F64,
    s: Settings,
    seeds: tuple[F64, ...] = (),
    keep: int = 4,
    cons: Sequence[Constraint] = (),
    max_tilt_deg: float | None = None,
) -> list[DirectionScore]:
    """Distinct locally best pull directions, best first (see module docstring). The runners-up
    let the engineer compare, e.g. a side pull with one core against a split tool.

    ``cons``: only directions with ``lo <= d . axis <= hi`` for every (axis, lo, hi) are allowed
    (a mold piece must be able to move away from its neighbours). Empty when none is allowed.
    """
    if not len(normals):
        raise ValueError("empty surface")
    ref = mean_inward(normals, areas)
    cos_lim = math.cos(math.radians(s.max_tilt_deg if max_tilt_deg is None else max_tilt_deg))
    cos_lim -= 1e-9
    cands = [ref, *seeds, *np.vstack([np.eye(3), -np.eye(3)]), *_fibonacci(max(s.samples, 12))]
    c = np.array([v / np.linalg.norm(v) for v in cands])
    c = c[[_allowed(v, ref, cos_lim, cons) for v in c]]
    if not len(c):
        return []
    tot = float(areas.sum()) or 1.0
    bad = np.empty(len(c))
    ok = np.empty(len(c))
    for lo in range(0, len(c), 64):  # (m, 64) blocks keep the memory small on big meshes
        dr = np.degrees(np.arcsin(np.clip(-(normals @ c[lo : lo + 64].T), -1.0, 1.0)))
        dr = np.round(dr, 6)
        bad[lo : lo + 64] = (areas[:, None] * (dr < s.warn_min_deg)).sum(axis=0) / tot
        ok[lo : lo + 64] = (areas[:, None] * (dr >= s.min_ok_deg)).sum(axis=0) / tot
    order = np.lexsort((-ok, np.round(bad, 7)))
    starts: list[F64] = []
    far = math.cos(math.radians(15.0))
    for i in ([0] if _allowed(c[0], ref, cos_lim, cons) and np.allclose(c[0], ref) else []) + [
        int(j) for j in order
    ]:
        if len(starts) >= keep + 1:
            break
        if all(float(c[i] @ x) < far for x in starts):
            starts.append(c[i])
    step0 = math.sqrt(4 * math.pi / max(s.samples, 12))
    allow = lambda v: _allowed(v, ref, cos_lim, cons)  # noqa: E731
    found = sorted(
        (_refine(normals, areas, x, allow, step0, s) for x in starts), key=DirectionScore.key
    )
    near = math.cos(math.radians(10.0))  # runners-up must be a different way out
    out: list[DirectionScore] = []
    for r in found:
        if all(float(np.dot(r.direction, o.direction)) < near for o in out):
            out.append(r)
    return out


def best_direction(
    normals: F64, areas: F64, s: Settings, seeds: tuple[F64, ...] = ()
) -> DirectionScore:
    """The pull direction that releases this surface best."""
    return search_directions(normals, areas, s, seeds, keep=3)[0]


def _refine(
    normals: F64,
    areas: F64,
    d: F64,
    allow: Callable[[F64], bool],
    step: float,
    s: Settings,
) -> DirectionScore:
    cur = score(normals, areas, d, s)
    stop = math.radians(s.refine_min_step_deg)
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    while step >= stop:
        a, b = _basis(np.array(cur.direction))
        moved = False
        for t in angles:
            v = np.array(cur.direction) * math.cos(step) + math.sin(step) * (
                math.cos(t) * a + math.sin(t) * b
            )
            if not allow(v):
                continue
            sc = score(normals, areas, v, s)
            if sc.key() < cur.key():
                cur, moved = sc, True
        if not moved:
            step /= 2
    return cur


# --- regions and decision -------------------------------------------------------------


class RegionKind(StrEnum):
    NOISE = "gurultu"
    CORE = "maca"
    SPLIT = "bolunme"


@dataclass(frozen=True)
class UndercutRegion:
    tris: I64
    area_mm2: float
    area_ratio: float  # of the whole tool surface
    min_draft_deg: float
    lateral: tuple[float, float, float]  # mean normal projected off the pull (which side)
    faces: tuple[int, ...]
    kind: RegionKind
    why: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # area-weighted centre, mm
    outward: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mean normal (out of the part)


def _tup(v: F64) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def undercut_regions(ts: TriSet, sel: I64, d: F64, s: Settings) -> list[UndercutRegion]:
    """Undercut regions of the triangles ``sel`` for pull ``d``, classified (area share is always
    of the whole tool surface, so a piece's residual is judged like the original)."""
    d = np.asarray(d, dtype=float) / np.linalg.norm(d)
    mask = np.zeros(len(ts.areas), dtype=bool)
    dr = draft_deg(ts.normals[sel], d)
    mask[sel[dr < s.warn_min_deg]] = True
    lab = components(ts, mask)
    tot = ts.total_area or 1.0
    out: list[UndercutRegion] = []
    for root in np.unique(lab[mask]):
        idx = np.flatnonzero(lab == root)
        area = float(ts.areas[idx].sum())
        ratio = area / tot
        mind = float(draft_deg(ts.normals[idx], d).min())
        n = (ts.normals[idx] * ts.areas[idx, None]).sum(axis=0)
        lat = n - (n @ d) * d
        ln = float(np.linalg.norm(lat))
        lat = lat / ln if ln > 1e-12 else lat
        if ratio < s.noise_area_ratio and mind >= -s.noise_draft_deg:
            kind = RegionKind.NOISE
            why = (
                f"alan %{ratio * 100:.2f} < %{s.noise_area_ratio * 100:.2f} ve en derin "
                f"{mind:.2f}° ≥ -{s.noise_draft_deg:g}°: tessellation gürültüsü olabilir"
            )
        elif ratio < s.core_max_area_ratio:
            kind = RegionKind.CORE
            why = f"alan %{ratio * 100:.1f} < maça eşiği %{s.core_max_area_ratio * 100:.0f}"
        else:
            kind = RegionKind.SPLIT
            why = f"alan %{ratio * 100:.1f} ≥ maça eşiği %{s.core_max_area_ratio * 100:.0f}"
        out.append(
            UndercutRegion(
                idx,
                area,
                ratio,
                mind,
                (float(lat[0]), float(lat[1]), float(lat[2])),
                tuple(sorted({int(x) for x in ts.face[idx]})),
                kind,
                why,
                _tup(np.average(ts.centroids[idx], axis=0, weights=ts.areas[idx])),
                _tup(n / (float(np.linalg.norm(n)) or 1.0)),
            )
        )
    out.sort(key=lambda r: -r.area_mm2)
    return out


def _mark_opposite(regions: list[UndercutRegion], s: Settings) -> list[UndercutRegion]:
    """Small regions on opposite sides whose sum is not small: a two-sided undercut -> split."""
    cores = [r for r in regions if r.kind is RegionKind.CORE]
    hit: set[int] = set()
    for i, a in enumerate(cores):
        for b in cores[i + 1 :]:
            if (
                float(np.dot(a.lateral, b.lateral)) < s.opposite_cos
                and a.area_ratio + b.area_ratio >= s.core_max_area_ratio
            ):
                hit |= {id(a), id(b)}
    if not hit:
        return regions
    out = []
    for r in regions:
        if id(r) in hit:
            r = UndercutRegion(
                r.tris,
                r.area_mm2,
                r.area_ratio,
                r.min_draft_deg,
                r.lateral,
                r.faces,
                RegionKind.SPLIT,
                r.why + "; karşılıklı yandaki bölgeyle toplamı maça eşiğini aşıyor",
                r.center,
                r.outward,
            )
        out.append(r)
    return out


@dataclass(frozen=True)
class MoldPiece:
    tris: I64
    area_ratio: float
    pull: DirectionScore
    cores: list[UndercutRegion]
    unresolved: list[UndercutRegion]


@dataclass(frozen=True)
class SplitPlan:
    pieces: list[MoldPiece]

    @property
    def n_cores(self) -> int:
        return sum(len(p.cores) for p in self.pieces)

    @property
    def unresolved(self) -> list[UndercutRegion]:
        return [r for p in self.pieces for r in p.unresolved]


def _significant_undercut(ts: TriSet, sel: I64, d: F64, s: Settings) -> float:
    return sum(
        r.area_ratio for r in undercut_regions(ts, sel, d, s) if r.kind is not RegionKind.NOISE
    )


def split_plan(
    ts: TriSet, cut: list[UndercutRegion], pull: F64, s: Settings, max_pieces: int = 8
) -> SplitPlan:
    """Split the surface along the boundaries of the ``cut`` regions, give every piece its own
    best direction and merge adjacent pieces while that adds no undercut."""
    n = len(ts.areas)
    owner = np.full(n, -1, dtype=np.int64)
    for k, r in enumerate(cut):
        owner[r.tris] = k
    rest = components(ts, owner < 0)
    tot = ts.total_area or 1.0
    others = sorted(
        (np.flatnonzero(rest == x) for x in np.unique(rest[rest >= 0])),
        key=lambda g: -float(ts.areas[g].sum()),
    )
    if len(others) > 1:  # small islands (not touching anything) ride with the largest rest piece
        small = [g for g in others[1:] if float(ts.areas[g].sum()) / tot < s.core_max_area_ratio]
        if small:
            others = [np.concatenate([others[0], *small])] + [
                g for g in others[1:] if float(ts.areas[g].sum()) / tot >= s.core_max_area_ratio
            ]
    groups = [g for g in [*(r.tris for r in cut), *others] if len(g)]
    if len(groups) > max_pieces:  # too fragmented to merge sensibly: leave to the engineer
        groups = groups[:max_pieces]

    def solve(g: I64, seeds: tuple[F64, ...]) -> tuple[DirectionScore, float]:
        sc = best_direction(ts.normals[g], ts.areas[g], s, seeds)
        return sc, _significant_undercut(ts, g, np.array(sc.direction), s)

    sol = [solve(g, (pull,)) for g in groups]
    gid = np.full(n, -1, dtype=np.int64)
    for k, g in enumerate(groups):
        gid[g] = k

    def adjacent(i: int, j: int) -> bool:
        a, b = gid[ts.pairs[:, 0]], gid[ts.pairs[:, 1]]
        return bool((((a == i) & (b == j)) | ((a == j) & (b == i))).any())

    eps = 1e-6
    while len(groups) > 1:
        best: tuple[float, int, int, DirectionScore, float] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if not adjacent(i, j):
                    continue
                g = np.concatenate([groups[i], groups[j]])
                sc, res = solve(
                    g, (np.array(sol[i][0].direction), np.array(sol[j][0].direction), pull)
                )
                gain = sol[i][1] + sol[j][1] + eps - res
                if gain >= 0 and (best is None or res < best[0]):
                    best = (res, i, j, sc, res)
        if best is None:
            break
        _, i, j, sc, res = best
        groups[i] = np.concatenate([groups[i], groups[j]])
        sol[i] = (sc, res)
        del groups[j], sol[j]
        gid[:] = -1
        for k, g in enumerate(groups):
            gid[g] = k
    pieces = []
    for g, (sc, _) in zip(groups, sol, strict=True):
        regs = [
            r
            for r in undercut_regions(ts, g, np.array(sc.direction), s)
            if r.kind is not RegionKind.NOISE
        ]
        pieces.append(
            MoldPiece(
                g,
                float(ts.areas[g].sum()) / tot,
                sc,
                [r for r in regs if r.kind is RegionKind.CORE],
                [r for r in regs if r.kind is RegionKind.SPLIT],
            )
        )
    pieces.sort(key=lambda p: -p.area_ratio)
    return SplitPlan(pieces)


class Solution(StrEnum):
    SINGLE = "tek_parca"
    CORE = "maca"
    SPLIT = "bolunmus_kalip"
    SPLIT_CORE = "bolunmus_kalip_maca"
    REVIEW = "manual_review"


SOLUTION_LABEL = {
    Solution.SINGLE: "Tek parça kalıp (maça ve bölünme gerekmiyor)",
    Solution.CORE: "Tek parça kalıp + maça",
    Solution.SPLIT: "Bölünmüş kalıp",
    Solution.SPLIT_CORE: "Bölünmüş kalıp + maça (ikisi birlikte)",
    Solution.REVIEW: "Karar verilemedi: mühendis incelemesi (manual_review)",
}


# --- split methods --------------------------------------------------------------------
# Kemal 2026-11-09: before deciding, the surface is judged with several ways of splitting the
# tool (one piece, planes through the part, the undercut regions' own boundaries); each way gives
# every piece its own pull direction and leaves some cores. The cheapest way wins.

# Offsets of the parting planes as shares of the tool-surface area along the plane normal: the
# discretisation of the search, not a company value.
_Q2 = (0.35, 0.5, 0.65)
_Q3 = ((0.25, 0.5), (0.33, 0.67), (0.5, 0.75), (0.25, 0.75))
# plane normals tried, candidates evaluated exactly per piece count and axes tried for the
# visibility split are thresholds: ``pull_search.plane_normals``, ``exact_candidates``,
# ``axis_splits`` (``axis_mid_plane``: always evaluate the mid-part plane of every world axis)
_MERGE_COS = math.cos(math.radians(10.0))  # plane normals closer than this count as one


@dataclass(frozen=True)
class PartingPlane:
    normal: tuple[float, float, float]
    offset: float  # mm: the plane is n . x = offset


UNVERIFIED_KINDS = ("region", "direction")  # pieces not checked to slide apart from each other


@dataclass(frozen=True)
class Method:
    """One way to build the tool: pieces (each with a pull direction and its cores) and cost."""

    kind: str  # "single" | "plane" | "visibility" | "direction" | "region"
    label: str
    pieces: list[MoldPiece]
    planes: list[PartingPlane]
    piece_of: I64  # (m,) piece index per triangle
    n_cores: int
    n_unresolved: int
    cost: float
    warn_share: float = 0.0  # area share released with only warn_min_deg..min_ok_deg of draft
    core_share: float = 0.0  # share of the tool surface that the cores cover

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    @property
    def solution(self) -> Solution:
        if self.n_unresolved:
            return Solution.REVIEW
        if self.n_pieces > 1:
            return Solution.SPLIT_CORE if self.n_cores else Solution.SPLIT
        return Solution.CORE if self.n_cores else Solution.SINGLE


def method_cost(
    n_pieces: int,
    n_cores: int,
    n_unresolved: int,
    warn_share: float,
    s: Settings,
    core_share: float = 0.0,
) -> float:
    return (
        s.split_cost_weight * (n_pieces - 1)
        + s.core_cost_weight * n_cores
        + s.core_area_cost_weight * core_share
        + s.unresolved_cost_weight * n_unresolved
        + s.warn_cost_weight * warn_share
    )


def _make_method(
    kind: str,
    label: str,
    ts: TriSet,
    pieces: list[MoldPiece],
    planes: list[PartingPlane],
    s: Settings,
) -> Method:
    piece_of = np.full(len(ts.areas), -1, dtype=np.int64)
    for k, p in enumerate(pieces):
        piece_of[p.tris] = k
    n_cores = sum(len(p.cores) for p in pieces)
    n_unres = sum(len(p.unresolved) for p in pieces)
    warn = sum(p.pull.warn_ratio * p.area_ratio for p in pieces)
    core_share = sum(r.area_ratio for p in pieces for r in p.cores)
    cost = method_cost(len(pieces), n_cores, n_unres, warn, s, core_share)
    return Method(kind, label, pieces, planes, piece_of, n_cores, n_unres, cost, warn, core_share)


def _aggregate(normals: F64, areas: F64, cell: float = 0.01) -> tuple[F64, F64]:
    """Triangles with (nearly) the same normal merged: the quick search only needs the area per
    normal, and big planar or cylindrical faces collapse to a handful of entries."""
    key = np.round(normals / cell).astype(np.int32)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = np.bincount(inv, weights=areas)
    n = np.stack([np.bincount(inv, weights=normals[:, k] * areas) for k in range(3)], axis=1)
    ln = np.linalg.norm(n, axis=1)
    keep = (w > 0) & (ln > 1e-12)
    return n[keep] / ln[keep, None], w[keep]


def _plane_cons(k: int, m: int, n: F64, s: Settings) -> list[Constraint]:
    """Pull constraints of piece ``k`` of ``m`` pieces along plane normal ``n``: the piece on the
    negative side moves away from the others (the part moves +n relative to it), the piece on the
    positive side the other way, a piece between two planes only along them."""
    slack = math.sin(math.radians(s.sandwich_tol_deg))
    if k == 0:
        return [(n, 0.0, 1.0)]
    if k == m - 1:
        return [(n, -1.0, 0.0)]
    return [(n, -slack, slack)]


def _quick_settings(s: Settings) -> Settings:
    return replace(
        s,
        samples=max(60, s.samples // 5),
        refine_min_step_deg=max(s.refine_min_step_deg, 0.5),
    )


def _weighted_quantiles(pos: F64, areas: F64, qs: Sequence[float]) -> list[float]:
    order = np.argsort(pos)
    cum = np.cumsum(areas[order]) / (float(areas.sum()) or 1.0)
    idx = np.minimum(np.searchsorted(cum, qs), len(pos) - 1)
    return [float(pos[order][i]) for i in idx]


def _plane_normals(ts: TriSet, pull: F64, regions: list[UndercutRegion], s: Settings) -> list[F64]:
    """Candidate parting-plane normals, most informative first: the sides the undercuts face,
    the part's principal axes, the world axes, directions around the baseline pull."""
    cands: list[F64] = []
    for r in regions:
        v = np.asarray(r.lateral, dtype=float)
        if float(np.linalg.norm(v)) > 0.5 and r.kind is not RegionKind.NOISE:
            cands.append(v / np.linalg.norm(v))
    c = ts.centroids - np.average(ts.centroids, axis=0, weights=ts.areas)
    cov = (c * ts.areas[:, None]).T @ c
    _, vecs = np.linalg.eigh(cov)
    cands += [vecs[:, k] for k in (2, 1, 0)]
    cands += list(np.eye(3))
    a, b = _basis(pull)
    cands += [math.cos(t) * a + math.sin(t) * b for t in np.radians([0, 45, 90, 135])]
    out: list[F64] = []
    for v in cands:
        v = v / np.linalg.norm(v)
        if all(abs(float(v @ o)) < _MERGE_COS for o in out):  # +n and -n are the same plane
            out.append(v)
        if len(out) >= s.plane_normals:
            break
    return out


def _quick_plane(
    ts: TriSet, n: F64, offsets: Sequence[float], s: Settings, qs: Settings
) -> tuple[float, float] | None:
    """Undercut area share left by a plane split (aggregated search); None if a piece cannot
    move away from its neighbours at all. Returns (undercut share, low-draft share)."""
    lab = np.digitize(ts.centroids @ n, offsets)
    m = len(offsets) + 1
    left = warn = 0.0
    for k in range(m):
        g = np.flatnonzero(lab == k)
        if not len(g):
            return None
        nn, aa = _aggregate(ts.normals[g], ts.areas[g])
        found = search_directions(
            nn, aa, qs, keep=1, cons=_plane_cons(k, m, n, s), max_tilt_deg=s.piece_max_tilt_deg
        )
        if not found:
            return None
        left += found[0].undercut_ratio * float(ts.areas[g].sum())
        warn += found[0].warn_ratio * float(ts.areas[g].sum())
    tot = ts.total_area or 1.0
    return left / tot, warn / tot


def _exact_pieces(
    ts: TriSet, groups: list[I64], cons: list[list[Constraint]], s: Settings, tilt: float
) -> list[MoldPiece] | None:
    tot = ts.total_area or 1.0
    pieces: list[MoldPiece] = []
    for g, cn in zip(groups, cons, strict=True):
        found = search_directions(ts.normals[g], ts.areas[g], s, keep=3, cons=cn, max_tilt_deg=tilt)
        if not found:
            return None
        sc = found[0]
        regs = _mark_opposite(undercut_regions(ts, g, np.array(sc.direction), s), s)
        regs = [r for r in regs if r.kind is not RegionKind.NOISE]
        pieces.append(
            MoldPiece(
                g,
                float(ts.areas[g].sum()) / tot,
                sc,
                [r for r in regs if r.kind is RegionKind.CORE],
                [r for r in regs if r.kind is RegionKind.SPLIT],
            )
        )
    return pieces


def _plane_methods(
    ts: TriSet, pull: F64, regions: list[UndercutRegion], s: Settings
) -> list[Method]:
    """Plane splits (2 and, when allowed, 3 pieces): quick search of every normal/offset, exact
    evaluation of the best few."""
    qs = _quick_settings(s)
    quick: list[tuple[float, int, F64, list[float]]] = []  # (approximate cost, ...)
    for n in _plane_normals(ts, pull, regions, s):
        pos = ts.centroids @ n
        options: list[list[float]] = [_weighted_quantiles(pos, ts.areas, (q,)) for q in _Q2]
        if s.max_split_pieces >= 3:
            options += [_weighted_quantiles(pos, ts.areas, q) for q in _Q3]
        for offs in options:
            r = _quick_plane(ts, n, offs, s, qs)
            if r is not None:
                approx = s.unresolved_cost_weight * r[0] + s.warn_cost_weight * r[1]
                quick.append((approx, len(offs) + 1, n, offs))
    methods: list[Method] = []
    for m in (2, 3):
        ranked = sorted((q for q in quick if q[1] == m), key=lambda q: q[0])
        best = ranked[: s.exact_candidates]
        if s.axis_mid_plane and m == 2:
            # the mid-part plane of each world axis is the plane an engineer tries first; the
            # quick score must not drop it (a rank by approximate cost once did)
            for q in ranked[s.exact_candidates :]:
                n_q, offs_q = q[2], q[3]
                if float(np.abs(n_q).max()) > 0.98 and offs_q == _weighted_quantiles(
                    ts.centroids @ n_q, ts.areas, (_Q2[1],)
                ):
                    best.append(q)
        for _, _, n, offs in best:
            lab = np.digitize(ts.centroids @ n, offs)
            groups = [np.flatnonzero(lab == k) for k in range(m)]
            cons = [_plane_cons(k, m, n, s) for k in range(m)]
            pieces = _exact_pieces(ts, groups, cons, s, s.piece_max_tilt_deg)
            if pieces is None:
                continue
            nt = (float(n[0]), float(n[1]), float(n[2]))
            axis = "XYZ"[int(np.argmax(np.abs(n)))] if float(np.abs(n).max()) > 0.98 else None
            where = f"{axis} eksenine dik" if axis else f"normal {_fmt(nt)}"
            methods.append(
                _make_method(
                    "plane",
                    f"{m} parçalı bölünmüş kalıp ({where} ayırma düzlemi)",
                    ts,
                    pieces,
                    [PartingPlane(nt, o) for o in offs],
                    s,
                )
            )
    return methods


def _direction_methods(ts: TriSet, s: Settings) -> list[Method]:
    """Direction partitions (2 .. max pieces): choose k pull directions that together release as
    much of the surface as possible (greedy cover, then Lloyd iterations), give every triangle to
    the direction that releases it best (smoothed so the pieces are coherent patches), and count
    what is still locked. The parting lines run on the surface where the release switches, so a
    hood whose two sides lean in gets a left and a right piece without any core."""
    if s.max_split_pieces < 2 or not len(ts.areas):
        return []
    qs = _quick_settings(s)
    nn, aa = _aggregate(ts.normals, ts.areas)
    cand = _fibonacci(max(250, qs.samples * 2))
    dd = np.degrees(np.arcsin(np.clip(-(nn @ cand.T), -1.0, 1.0)))
    wgt = np.where(dd >= s.min_ok_deg, 1.0, np.where(dd >= s.warn_min_deg, 0.5, 0.0))
    out: list[Method] = []
    for k in range(2, s.max_split_pieces + 1):
        cov = np.zeros(len(nn))
        dirs: list[F64] = []
        for _ in range(k):
            gain = (aa[:, None] * np.maximum(wgt - cov[:, None], 0.0)).sum(axis=0)
            j = int(np.argmax(gain))
            dirs.append(cand[j])
            cov = np.maximum(cov, wgt[:, j])
        lab = _assign(ts, np.array(dirs), s)
        for _ in range(3):  # Lloyd: re-optimise every direction on its own triangles
            new = []
            for i in range(len(dirs)):
                g = np.flatnonzero(lab == i)
                if not len(g):
                    new.append(dirs[i])
                    continue
                nq, aq = _aggregate(ts.normals[g], ts.areas[g])
                f = search_directions(nq, aq, qs, (dirs[i],), keep=1, max_tilt_deg=180.0)
                new.append(np.array(f[0].direction) if f else dirs[i])
            relab = _assign(ts, np.array(new), s)
            same = bool((relab == lab).all())
            dirs, lab = new, relab
            if same:
                break
        groups = _connected_groups(ts, lab, len(dirs), s)
        if len(groups) < 2 or len(groups) > s.max_split_pieces:
            continue
        pieces = _exact_pieces(ts, groups, [[] for _ in groups], s, 180.0)
        if pieces is None:
            continue
        pieces.sort(key=lambda p: -p.area_ratio)
        out.append(
            _make_method(
                "direction",
                f"{len(pieces)} parçalı bölünmüş kalıp (yüzey yönlerine göre bölme; ayrılabilirlik "
                "doğrulanmadı)",
                ts,
                pieces,
                [],
                s,
            )
        )
    return out


def _connected_groups(ts: TriSet, lab: I64, n_labels: int, s: Settings) -> list[I64]:
    """Triangle groups of a labelling where every piece is one connected patch. A small patch
    (< core_max_area_ratio of the surface) cut off from its label's large patch is given to the
    label around it, so that piece's own undercut analysis counts it (as a core if nothing
    releases it); a label with no large patch dissolves into its surroundings. A label with
    several large patches becomes several pieces (interleaved strips would be a comb of pieces,
    not two halves)."""
    lab = lab.copy()
    total = float(ts.areas.sum()) or 1.0
    pa, pb = ts.pairs[:, 0], ts.pairs[:, 1]
    for _ in range(3):  # reassigning an island can leave another one behind: a few rounds
        changed = False
        for i in range(n_labels):
            mine = lab == i
            if not mine.any():
                continue
            comp = components(ts, mine)
            ids, inv = np.unique(comp[mine], return_inverse=True)
            share = np.bincount(inv, weights=ts.areas[mine]) / total
            for c, sh in zip(ids, share, strict=True):
                if sh >= s.core_max_area_ratio:  # a main patch of this label stays
                    continue
                isl = comp == c
                cross = isl[pa] != isl[pb]
                out = np.where(isl[pa[cross]], pb[cross], pa[cross])
                around = lab[out]
                around = around[around != i]
                if len(around):
                    lab[isl] = int(np.bincount(around).argmax())
                    changed = True
        if not changed:
            break
    groups: list[I64] = []
    for i in range(n_labels):
        mine = lab == i
        if not mine.any():
            continue
        comp = components(ts, mine)
        ids, inv = np.unique(comp[mine], return_inverse=True)
        share = np.bincount(inv, weights=ts.areas[mine]) / total
        tris_i = np.flatnonzero(mine)
        big = [int(c) for c, sh in zip(ids, share, strict=True) if sh >= s.core_max_area_ratio]
        if len(big) <= 1:
            groups.append(tris_i)
            continue
        keep = int(ids[int(np.argmax(share))])
        for c in big:
            g = tris_i[comp[tris_i] == c]
            if c == keep:
                g = tris_i[(comp[tris_i] == c) | ~np.isin(comp[tris_i], big)]
            groups.append(g)
    return groups


def _visibility_methods(
    ts: TriSet, pull: F64, regions: list[UndercutRegion], s: Settings
) -> list[Method]:
    """Two pieces that open straight away from each other along a candidate axis (the part moves
    +n relative to one piece and -n relative to the other): every triangle goes to the piece it
    is visible from, so the parting line runs on the surface where the release switches. A
    hood whose two sides lean in is released by both halves with no core."""
    if s.max_split_pieces < 2 or not len(ts.areas):
        return []
    out: list[Method] = []
    for n in _plane_normals(ts, pull, regions, s)[: s.axis_splits]:
        lab = _assign(ts, np.array([n, -n]), s)
        groups = _connected_groups(ts, lab, 2, s)
        if len(groups) != 2:
            continue
        # the piece that is released along +n is the first (negative side of the plane)
        groups.sort(key=lambda g: float(np.mean(ts.normals[g] @ n)))
        # the pieces open straight away from each other: pulls stay within sandwich_tol_deg of +-n
        tol = math.cos(math.radians(s.sandwich_tol_deg))
        cons: list[list[Constraint]] = [[(n, tol, 1.0)], [(n, -1.0, -tol)]]
        pieces = _exact_pieces(ts, groups, cons, s, 180.0)
        if pieces is None:
            continue
        axis = "XYZ"[int(np.argmax(np.abs(n)))] if float(np.abs(n).max()) > 0.98 else None
        where = (
            f"{axis} ekseni boyunca"
            if axis
            else f"yön {_fmt((float(n[0]), float(n[1]), float(n[2])))}"
        )
        out.append(
            _make_method(
                "visibility",
                f"2 parçalı bölünmüş kalıp ({where} ± çekme, ayırma hattı yüzeyde)",
                ts,
                pieces,
                [],
                s,
            )
        )
    return out


def _assign(ts: TriSet, dirs: NDArray[np.float64], s: Settings, passes: int = 6) -> I64:
    """Piece of every triangle for the pull ``dirs``: the direction that releases it best; where
    several release it (draft >= warn_min_deg) the one most neighbours already use."""
    dr = np.degrees(np.arcsin(np.clip(-(ts.normals @ dirs.T), -1.0, 1.0)))
    lab = np.argmax(dr, axis=1).astype(np.int64)
    elig = dr >= s.warn_min_deg
    multi = elig.sum(axis=1) >= 2
    if not multi.any() or not len(ts.pairs):
        return lab
    pa, pb = ts.pairs[:, 0], ts.pairs[:, 1]
    for _ in range(passes):
        nd = len(dirs)
        cnt = (
            (
                np.bincount(pa * nd + lab[pb], minlength=len(lab) * nd)
                + np.bincount(pb * nd + lab[pa], minlength=len(lab) * nd)
            )
            .reshape(len(lab), nd)
            .astype(float)
        )
        cnt = np.where(elig, cnt + 0.01 * (np.arange(len(dirs)) == lab[:, None]), -1.0)
        best = np.argmax(cnt, axis=1)
        new = np.where(multi, best, lab)
        if (new == lab).all():
            break
        lab = new
    return lab


def parting_walls(ts: TriSet, piece_of: I64, length_mm: float) -> NDArray[np.float64]:
    """Triangles of the parting walls: along every edge between two pieces, a strip standing
    perpendicular to the tool surface (along the mean normal, into the tool) ``length_mm`` deep."""
    if ts.tri_vid is None or ts.vert_xyz is None or not len(ts.pairs):
        return np.zeros((0, 3, 3))
    pa, pb = ts.pairs[:, 0], ts.pairs[:, 1]
    diff = piece_of[pa] != piece_of[pb]
    pa, pb = pa[diff], pb[diff]
    if not len(pa):
        return np.zeros((0, 3, 3))
    va, vb = ts.tri_vid[pa], ts.tri_vid[pb]
    in_b = (va[:, :, None] == vb[:, None, :]).any(axis=2)
    ok = in_b.sum(axis=1) == 2
    va, in_b, pa, pb = va[ok], in_b[ok], pa[ok], pb[ok]
    idx = np.argsort(~in_b, axis=1, kind="stable")[:, :2]
    e = np.take_along_axis(va, idx, axis=1)
    p0, p1 = ts.vert_xyz[e[:, 0]], ts.vert_xyz[e[:, 1]]
    nrm = ts.normals[pa] + ts.normals[pb]
    ln = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.where(ln > 1e-9, ln, 1.0)[:, None] * length_mm
    q0, q1 = p0 + nrm, p1 + nrm
    return np.concatenate([np.stack([p0, p1, q1], 1), np.stack([p0, q1, q0], 1)])


@dataclass(frozen=True)
class ToolingAssessment:
    pull: DirectionScore  # the baseline (single-piece) pull the user looked at
    regions: list[UndercutRegion]  # undercut regions of that baseline pull
    solution: Solution
    plan: SplitPlan | None  # chosen method's pieces when it is a split
    n_cores: int
    reasons: list[str]
    alternatives: list[str]
    flags: list[ReviewFlag]
    trace: list[CostLine]
    methods: list[Method] = field(default_factory=list)  # every method tried, cheapest first
    settings: Settings = field(default_factory=Settings)
    shown: int = 0  # index in ``methods`` of the method this assessment describes
    base_flags: list[ReviewFlag] = field(default_factory=list)  # flags of the baseline pull
    single: Method | None = None  # the one-piece method with the pull the user looked at

    @property
    def chosen(self) -> Method | None:
        return self.methods[self.shown] if self.methods else None

    @property
    def recommended(self) -> Method | None:
        """The system's own pick (cheapest method), whichever one is shown."""
        return self.methods[0] if self.methods else None

    def view(self, index: int) -> ToolingAssessment:
        """The same comparison, described for another method: its decision, pieces, cores,
        reasons and flags (the user picked it in the table)."""
        if not self.methods or not 0 <= index < len(self.methods) or index == self.shown:
            return self
        m = self.methods[index]
        single = self.single or self.methods[0]
        solution, reasons, alts, mflags = _decide(self.methods, m, single, self.settings)
        if index != 0:
            rec = self.methods[0]
            reasons.insert(
                0,
                f"Bu yöntemi tablodan siz seçtiniz; sistemin önerisi '{rec.label}' "
                f"(puan {rec.cost:.1f}, bu yöntem {m.cost:.1f}).",
            )
        title = f"Kalıp kararı (kullanıcının seçtiği yöntem): {SOLUTION_LABEL[solution]}"
        trace = [
            ln.model_copy(update={"description": title, "result": float(m.n_pieces)})
            if ln.key == "tooling.decision"
            else ln
            for ln in self.trace
        ]
        return replace(
            self,
            trace=trace,
            solution=solution,
            plan=SplitPlan(m.pieces) if m.n_pieces > 1 else None,
            n_cores=m.n_cores,
            reasons=reasons,
            alternatives=alts,
            flags=[*self.base_flags, *mflags],
            shown=index,
        )

    @property
    def max_pieces(self) -> int:
        return self.settings.max_split_pieces

    @property
    def label(self) -> str:
        ch = self.chosen
        if self.solution in (Solution.SPLIT, Solution.SPLIT_CORE) and ch:
            extra = f", {self.n_cores} maça" if self.n_cores else ""
            return f"{SOLUTION_LABEL[self.solution]}: {ch.n_pieces} parça{extra}"
        if self.solution is Solution.CORE:
            return f"{SOLUTION_LABEL[self.solution]}: {self.n_cores} maça"
        return SOLUTION_LABEL[self.solution]

    @property
    def manual_review(self) -> bool:
        return bool(self.flags)


def _fmt(d: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in d) + ")"


def _method_line(i: int, m: Method, s: Settings) -> CostLine:
    return CostLine(
        key=f"method.{i}",
        description=f"Yöntem {i}: {m.label}: {m.n_pieces} parça, {m.n_cores} maça, "
        f"{m.n_unresolved} çözülemeyen bölge",
        formula="split_cost_weight × (parça − 1) + core_cost_weight × maça "
        "+ core_area_cost_weight × maçaların alan payı "
        "+ unresolved_cost_weight × çözülemeyen bölge + warn_cost_weight × düşük çekmeli alan payı",
        inputs=[
            _inp(s, "split_cost_weight"),
            _inp(s, "core_cost_weight"),
            _inp(s, "core_area_cost_weight"),
            _inp(s, "unresolved_cost_weight"),
            _inp(s, "warn_cost_weight"),
            CostInput(
                name="düşük çekmeli alan payı",
                value=m.warn_share,
                unit="ratio",
                source=Source.DERIVED,
            ),
            CostInput(
                name="maçaların alan payı", value=m.core_share, unit="ratio", source=Source.DERIVED
            ),
            CostInput(name="parça", value=float(m.n_pieces), unit="count", source=Source.DERIVED),
            CostInput(name="maça", value=float(m.n_cores), unit="count", source=Source.DERIVED),
            CostInput(
                name="çözülemeyen bölge",
                value=float(m.n_unresolved),
                unit="count",
                source=Source.DERIVED,
            ),
        ],
        result=m.cost,
        unit="puan",
        category="tooling",
    )


def _decide(
    methods: list[Method], best: Method, single: Method, s: Settings
) -> tuple[Solution, list[str], list[str], list[ReviewFlag]]:
    """Decision text and flags that belong to ``best`` among the compared ``methods``."""
    solution = best.solution
    reasons: list[str] = []
    alts: list[str] = []
    flags: list[ReviewFlag] = []
    reasons.append(
        f"seçilen yöntem: {best.label}: {best.n_pieces} parça, {best.n_cores} maça, "
        f"maliyet puanı {best.cost:.1f}"
    )
    if single is not best:
        reasons.append(
            f"tek parça kalıp: {single.n_cores} maça"
            + (f" + {single.n_unresolved} çözülemeyen bölge" if single.n_unresolved else "")
            + f", puan {single.cost:.1f}"
            + (f", maçalar yüzeyin %{single.core_share * 100:.1f}'i" if single.core_share else "")
        )
    others = [x for x in methods if x is not best]
    for m in others[:4]:
        if m is single:
            continue
        alts.append(
            f"{m.label}: {m.n_cores} maça"
            + (f", {m.n_unresolved} çözülemeyen" if m.n_unresolved else "")
            + f", puan {m.cost:.1f}"
        )
    for k, p in enumerate(best.pieces, 1):
        reasons.append(
            f"parça {k}: yüzeyin %{p.area_ratio * 100:.0f}'i, çekme {_fmt(p.pull.direction)}, "
            f"en küçük açı {p.pull.min_draft_deg:.1f}°"
            + (f", {len(p.cores)} maça" if p.cores else "")
            + (f", {len(p.unresolved)} çözülemeyen bölge" if p.unresolved else "")
        )
    if s.max_split_pieces < 2:
        reasons.append("kalıp parça sınırı 1: bölünmüş kalıp denenmedi")
    warn_share = best.warn_share
    if warn_share >= s.warn_review_ratio:
        flags.append(
            ReviewFlag(
                code="low_draft",
                reason=f"yüzeyin %{warn_share * 100:.0f}'i {s.warn_min_deg:g}–{s.min_ok_deg:g}° "
                "arasında: çıkar ama zor; farklı çekme yönlerini/yöntemleri de karşılaştırın.",
                path=f"{LIB}:draft.warn_review_ratio",
            )
        )
    if best.n_unresolved:
        flags.append(
            ReviewFlag(
                code="tooling_unresolved",
                reason=f"{best.n_unresolved} ters açı bölgesi izin verilen en fazla "
                f"{s.max_split_pieces} parçayla ve maçayla çözülemiyor (kapalı kesit → mandrel "
                "olabilir)",
                path=f"{LIB}:undercut.max_split_pieces",
            )
        )
    if best.kind in UNVERIFIED_KINDS and best.n_pieces > 1:
        flags.append(
            ReviewFlag(
                code="separation_unverified",
                reason="Kalıp parçalarının birbirinden ayrılması (çarpışma, çıkış sırası) "
                "kontrol edilmedi; yalnız her parçanın kendi çekme açısı doğrulandı.",
                path=f"{LIB}:undercut.max_split_pieces",
            )
        )
    if best.n_pieces > 1:
        flags.append(
            ReviewFlag(
                code="split_parting_line",
                reason="Ayırma düzlemi/hattı otomatik seçildi; parça sayısı, hattı ve parçaların "
                "çıkış sırası mühendis tarafından onaylanmalı.",
                path=f"{LIB}:undercut.max_split_pieces",
            )
        )
    rivals = [m for m in others if m.n_pieces != best.n_pieces and not m.n_unresolved]
    rivals.sort(key=lambda m: abs(m.cost - best.cost))
    if rivals and abs(rivals[0].cost - best.cost) <= s.threshold_margin_ratio * max(best.cost, 1.0):
        flags.append(
            ReviewFlag(
                code="method_close",
                reason=f"'{rivals[0].label}' maliyet puanı ({rivals[0].cost:.1f}) seçilene "
                f"({best.cost:.1f}) çok yakın: iki yöntemi karşılaştırın.",
                path=f"{LIB}:undercut.split_cost_weight",
            )
        )
    return solution, reasons, alts, flags


def assess(ts: TriSet, pull: F64, s: Settings) -> ToolingAssessment:
    """Tooling decision for the tool surface ``ts``: the baseline pull is judged as one piece,
    then split methods are tried (up to ``s.max_split_pieces`` pieces) and the cheapest wins."""
    pull = np.asarray(pull, dtype=float) / np.linalg.norm(pull)
    allt = np.arange(len(ts.areas))
    sc = score(ts.normals, ts.areas, pull, s)
    regions = _mark_opposite(undercut_regions(ts, allt, pull, s), s)
    noise = [r for r in regions if r.kind is RegionKind.NOISE]
    sig = [r for r in regions if r.kind is not RegionKind.NOISE]
    cores = [r for r in sig if r.kind is RegionKind.CORE]
    large = [r for r in sig if r.kind is RegionKind.SPLIT]
    flags: list[ReviewFlag] = []
    a_m2 = ts.total_area / 1e6
    trace = [
        CostLine(
            key="draft.undercut_ratio",
            description=f"Ters açı payı, çekme yönü {_fmt(sc.direction)}",
            formula="Σ A_i (asin(-n_i·d) < warn_min_deg) / Σ A_i",
            inputs=[
                CostInput(name="kalıp yüzeyi alanı", value=a_m2, unit="m2", source=Source.CAD),
                _inp(s, "warn_min_deg"),
            ],
            result=sc.undercut_ratio,
            unit="ratio",
            category="draft",
        ),
        CostLine(
            key="draft.ok_ratio",
            description="Yeterli çekme açılı alan payı",
            formula="Σ A_i (draft_i ≥ min_ok_deg) / Σ A_i",
            inputs=[_inp(s, "min_ok_deg")],
            result=sc.ok_ratio,
            unit="ratio",
            category="draft",
        ),
        CostLine(
            key="draft.min_deg",
            description="En küçük çekme açısı (gürültüye dayanıklı)",
            formula="draft değerlerinin robust_min_quantile yüzdeliği",
            inputs=[_inp(s, "robust_min_quantile")],
            result=sc.min_draft_deg,
            unit="deg",
            category="draft",
        ),
    ]
    for k, r in enumerate(regions, 1):
        trace.append(
            CostLine(
                key=f"undercut.region_{k}",
                description=f"Ters açı bölgesi {k} ({r.kind.value}): {r.why}; yüzler {r.faces}",
                formula="A_bölge / A_kalıp_yüzeyi  (maça eşiği ile karşılaştırılır)",
                inputs=[
                    CostInput(
                        name="bölge alanı", value=r.area_mm2 / 1e6, unit="m2", source=Source.CAD
                    ),
                    _inp(s, "core_max_area_ratio"),
                    _inp(s, "noise_area_ratio"),
                    _inp(s, "noise_draft_deg"),
                ],
                result=r.area_ratio,
                unit="ratio",
                category="undercut",
            )
        )
    if noise:
        flags.append(
            ReviewFlag(
                code="undercut_noise",
                reason=f"{len(noise)} küçük ve sığ ters açı lekesi maça sayılmadı (toplam yüzey "
                f"payı %{sum(r.area_ratio for r in noise) * 100:.2f}, en derin "
                f"{min(r.min_draft_deg for r in noise):.1f}°; tessellation gürültüsü ya da küçük "
                "kaburga/çıkıntı olabilir); 3B ekranda kırmızı lekeleri kontrol edin.",
                path=f"{LIB}:undercut.noise_area_ratio",
            )
        )
    margin = s.threshold_margin_ratio * s.core_max_area_ratio
    near = [r for r in sig if abs(r.area_ratio - s.core_max_area_ratio) <= margin]
    if near:
        flags.append(
            ReviewFlag(
                code="threshold_margin",
                reason=f"{len(near)} bölgenin alan oranı maça eşiğine (%"
                f"{s.core_max_area_ratio * 100:.0f}) çok yakın: maça/bölünme kararı sınırda.",
                path=f"{LIB}:undercut.core_max_area_ratio",
            )
        )

    # every way of building the tool
    single = _make_method(
        "single",
        "tek parça kalıp (seçilen çekme yönü)",
        ts,
        [MoldPiece(allt, 1.0, sc, cores, large)],
        [],
        s,
    )
    methods = [single]
    best_one = search_directions(ts.normals, ts.areas, s, (pull,), keep=3)
    if best_one and float(np.dot(best_one[0].direction, pull)) < math.cos(math.radians(2.0)):
        ps1 = _exact_pieces(ts, [allt], [[]], s, s.max_tilt_deg)
        if ps1:
            methods.append(
                _make_method("single", "tek parça kalıp (en uygun çekme yönü)", ts, ps1, [], s)
            )
    cheapest_single = min(m.cost for m in methods)
    # a split costs at least one extra piece and ties go to fewer pieces: no need to search for
    # one when a single piece is already that cheap
    if s.max_split_pieces >= 2 and cheapest_single > s.split_cost_weight:
        methods += _plane_methods(ts, pull, regions, s)
        methods += _visibility_methods(ts, pull, regions, s)
        methods += _direction_methods(ts, s)
        if large:
            rp = split_plan(ts, large, pull, s)
            covered = sum(len(p.tris) for p in rp.pieces) == len(ts.areas)
            if covered and 1 < len(rp.pieces) <= s.max_split_pieces:
                methods.append(
                    _make_method(
                        "region",
                        "ters açı bölgelerinden bölünmüş kalıp (ayrılabilirlik doğrulanmadı)",
                        ts,
                        rp.pieces,
                        [],
                        s,
                    )
                )
    methods.sort(
        key=lambda m: (round(m.cost, 6), m.n_pieces, m.n_cores, m.kind in UNVERIFIED_KINDS)
    )
    seen: set[tuple[str, int, float]] = set()
    uniq: list[Method] = []
    for m in methods:
        sig_key = (m.label, m.n_pieces, round(m.cost, 6))
        if sig_key not in seen:
            seen.add(sig_key)
            uniq.append(m)
    methods = uniq
    base_flags = list(flags)
    best = methods[0]
    solution, reasons, alts, mflags = _decide(methods, best, single, s)
    flags.extend(mflags)
    for i, m in enumerate(methods[:6], 1):
        trace.append(_method_line(i, m, s))
    trace.append(
        CostLine(
            key="tooling.decision",
            description=f"Kalıp kararı: {SOLUTION_LABEL[solution]}",
            formula="en düşük puanlı yöntem seçilir; ters açı yok → tek parça, yalnız küçük "
            "bölgeler → maça, parçalara bölünür → bölünmüş kalıp (+ kalan küçük bölge → maça)",
            inputs=[
                _inp(s, "core_max_area_ratio"),
                _inp(s, "opposite_cos"),
                _inp(s, "max_split_pieces"),
            ],
            result=float(best.n_pieces),
            unit="parça",
            category="tooling",
            flags=list(flags),
        )
    )
    return ToolingAssessment(
        sc,
        regions,
        solution,
        SplitPlan(best.pieces) if best.n_pieces > 1 else None,
        best.n_cores,
        reasons,
        alts,
        flags,
        trace,
        methods,
        s,
        0,
        list(base_flags),
        single,
    )


def pull_angles_deg(d: tuple[float, float, float]) -> dict[str, float]:
    """Angle of the pull direction to each axis (deg), for the display."""
    v = np.asarray(d, dtype=float) / np.linalg.norm(d)
    return {ax: float(np.degrees(np.arccos(np.clip(v[i], -1, 1)))) for i, ax in enumerate("XYZ")}
