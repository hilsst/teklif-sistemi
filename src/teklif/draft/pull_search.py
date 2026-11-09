"""Best pull direction for a user-selected tool surface, undercut regions and the tooling decision.

Kemal (2026-11-09): after the STEP is loaded the system no longer builds a mold. It takes the tool
surface the user picked, searches the pull direction that releases it best (ideally one that needs
neither a split tool nor a loose core), and then says which solution the remaining undercuts need:
single-piece tool, loose core(s) (maça), split tool (1/2/3 pieces), or split tool AND cores.

Conventions (same as ``teklif.geometry.surface_select``): triangle normals point out of the part,
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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from teklif.common.models import CostInput, CostLine, ReviewFlag, Source

if TYPE_CHECKING:
    from teklif.geometry.surface_select import FaceMesh

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
LIB = "thresholds.yaml"


# --- settings -------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Thresholds of the search and the decision, read from ``thresholds.yaml``."""

    min_ok_deg: float = 2.0
    warn_min_deg: float = 0.0
    warn_review_ratio: float = 0.25
    core_max_area_ratio: float = 0.10
    noise_area_ratio: float = 0.002
    noise_draft_deg: float = 0.5
    opposite_cos: float = -0.5
    max_split_pieces: int = 3
    threshold_margin_ratio: float = 0.2
    max_tilt_deg: float = 90.0
    samples: int = 600
    refine_min_step_deg: float = 0.1
    robust_min_quantile: float = 0.01
    sources: dict[str, Source] = field(default_factory=dict, compare=False)

    def source(self, name: str) -> Source:
        return self.sources.get(name, Source.DEFAULT)


_PATHS = {
    "min_ok_deg": "draft.min_ok_deg",
    "warn_min_deg": "draft.warn_min_deg",
    "warn_review_ratio": "draft.warn_review_ratio",
    "core_max_area_ratio": "undercut.core_max_area_ratio",
    "noise_area_ratio": "undercut.noise_area_ratio",
    "noise_draft_deg": "undercut.noise_draft_deg",
    "opposite_cos": "undercut.opposite_cos",
    "max_split_pieces": "undercut.max_split_pieces",
    "threshold_margin_ratio": "undercut.threshold_margin_ratio",
    "max_tilt_deg": "pull_search.max_tilt_deg",
    "samples": "pull_search.samples",
    "refine_min_step_deg": "pull_search.refine_min_step_deg",
    "robust_min_quantile": "pull_search.robust_min_quantile",
}
_UNITS = {
    "min_ok_deg": "deg",
    "warn_min_deg": "deg",
    "noise_draft_deg": "deg",
    "max_tilt_deg": "deg",
    "refine_min_step_deg": "deg",
    "max_split_pieces": "count",
    "samples": "count",
}


def settings_from_thresholds(thr: Any) -> Settings:
    """Settings from a ``teklif.library.schema.Thresholds``; a TODO (None) value keeps the default
    and is reported by the library TODO list, never invented here."""
    vals: dict[str, Any] = {}
    sources: dict[str, Source] = {}
    for name, path in _PATHS.items():
        node: Any = thr
        for part in path.split("."):
            node = getattr(node, part, None)
        if node is None or node.value is None:
            continue
        v = node.value
        vals[name] = int(v) if name in ("max_split_pieces", "samples") else float(v)
        sources[name] = node.source
    return Settings(**vals, sources=sources)


def _inp(s: Settings, name: str) -> CostInput:
    return CostInput(
        name=name,
        value=float(getattr(s, name)),
        unit=_UNITS.get(name, "ratio"),
        source=s.source(name),
        path=f"{LIB}:{_PATHS[name]}",
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
        _, vid = np.unique(np.round(v / tol_mm), axis=0, return_inverse=True)
        vid = vid.ravel()[t]
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


def search_directions(
    normals: F64, areas: F64, s: Settings, seeds: tuple[F64, ...] = (), keep: int = 4
) -> list[DirectionScore]:
    """Distinct locally best pull directions, best first (see module docstring). The runners-up
    let the engineer compare, e.g. a side pull with one core against a split tool."""
    if not len(normals):
        raise ValueError("empty surface")
    ref = mean_inward(normals, areas)
    cos_lim = math.cos(math.radians(s.max_tilt_deg)) - 1e-9
    cands = [ref, *seeds, *np.vstack([np.eye(3), -np.eye(3)]), *_fibonacci(max(s.samples, 12))]
    c = np.array([v / np.linalg.norm(v) for v in cands])
    c = c[c @ ref >= cos_lim]
    tot = float(areas.sum()) or 1.0
    bad = np.empty(len(c))
    ok = np.empty(len(c))
    for lo in range(0, len(c), 64):  # (m, 64) blocks keep the memory small on big meshes
        dr = np.degrees(np.arcsin(np.clip(-(normals @ c[lo : lo + 64].T), -1.0, 1.0)))
        dr = np.round(dr, 6)
        bad[lo : lo + 64] = (areas[:, None] * (dr < s.warn_min_deg)).sum(axis=0) / tot
        ok[lo : lo + 64] = (areas[:, None] * (dr >= s.min_ok_deg)).sum(axis=0) / tot
    starts: list[F64] = [c[0]]  # the mean inward normal always gets refined
    far = math.cos(math.radians(15.0))
    for i in np.lexsort((-ok, np.round(bad, 7))):
        if len(starts) >= keep + 1:
            break
        if all(float(c[i] @ x) < far for x in starts):
            starts.append(c[i])
    step0 = math.sqrt(4 * math.pi / max(s.samples, 12))
    found = sorted(
        (_refine(normals, areas, x, ref, cos_lim, step0, s) for x in starts),
        key=DirectionScore.key,
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
    normals: F64, areas: F64, d: F64, ref: F64, cos_lim: float, step: float, s: Settings
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
            if float(v @ ref) < cos_lim:
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


@dataclass(frozen=True)
class ToolingAssessment:
    pull: DirectionScore
    regions: list[UndercutRegion]
    solution: Solution
    plan: SplitPlan | None  # for a split solution
    n_cores: int
    reasons: list[str]
    alternatives: list[str]
    flags: list[ReviewFlag]
    trace: list[CostLine]

    @property
    def label(self) -> str:
        if self.solution in (Solution.SPLIT, Solution.SPLIT_CORE) and self.plan:
            extra = f", {self.n_cores} maça" if self.n_cores else ""
            return f"{SOLUTION_LABEL[self.solution]}: {len(self.plan.pieces)} parça{extra}"
        if self.solution is Solution.CORE:
            return f"{SOLUTION_LABEL[self.solution]}: {self.n_cores} maça"
        return SOLUTION_LABEL[self.solution]

    @property
    def manual_review(self) -> bool:
        return bool(self.flags)


def _fmt(d: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in d) + ")"


def assess(ts: TriSet, pull: F64, s: Settings) -> ToolingAssessment:
    """Tooling decision for the tool surface ``ts`` pulled along ``pull``."""
    pull = np.asarray(pull, dtype=float) / np.linalg.norm(pull)
    allt = np.arange(len(ts.areas))
    sc = score(ts.normals, ts.areas, pull, s)
    regions = _mark_opposite(undercut_regions(ts, allt, pull, s), s)
    noise = [r for r in regions if r.kind is RegionKind.NOISE]
    cores = [r for r in regions if r.kind is RegionKind.CORE]
    large = [r for r in regions if r.kind is RegionKind.SPLIT]
    reasons: list[str] = []
    alts: list[str] = []
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
                reason=f"{len(noise)} küçük ve sığ ters açı lekesi karar dışı bırakıldı "
                "(tessellation gürültüsü varsayımı); 3B ekranda kırmızı lekeleri kontrol edin.",
                path=f"{LIB}:undercut.noise_area_ratio",
            )
        )
    margin = s.threshold_margin_ratio * s.core_max_area_ratio
    near = [r for r in cores + large if abs(r.area_ratio - s.core_max_area_ratio) <= margin]
    if near:
        flags.append(
            ReviewFlag(
                code="threshold_margin",
                reason=f"{len(near)} bölgenin alan oranı maça eşiğine (%"
                f"{s.core_max_area_ratio * 100:.0f}) çok yakın: maça/bölünme kararı sınırda.",
                path=f"{LIB}:undercut.core_max_area_ratio",
            )
        )
    if sc.warn_ratio >= s.warn_review_ratio:
        flags.append(
            ReviewFlag(
                code="low_draft",
                reason=f"yüzeyin %{sc.warn_ratio * 100:.0f}'i {s.warn_min_deg:g}–{s.min_ok_deg:g}° "
                "arasında: çıkar ama zor; bu yön yerine diğer önerileri de karşılaştırın.",
                path=f"{LIB}:draft.warn_review_ratio",
            )
        )
    if sc.warn_ratio > 1e-6:
        reasons.append(
            f"yüzeyin %{sc.warn_ratio * 100:.1f}'i {s.warn_min_deg:g}–{s.min_ok_deg:g}° "
            "arasında (çıkar ama zor; uyarı)"
        )

    plan: SplitPlan | None = None
    if not cores and not large:
        solution, n_cores = Solution.SINGLE, 0
        reasons.insert(0, "seçilen yönde ters açı bölgesi yok: kalıp bölünmeden, maçasız çıkar")
    elif not large:
        solution, n_cores = Solution.CORE, len(cores)
        reasons.insert(
            0,
            f"{len(cores)} ters açı bölgesinin hepsi küçük ve yerel (her biri maça eşiği %"
            f"{s.core_max_area_ratio * 100:.0f} altında): gevşek parça (maça)",
        )
        alt = split_plan(ts, cores, pull, s)
        if not alt.unresolved and len(alt.pieces) <= s.max_split_pieces:
            alts.append(
                f"maçasız alternatif: {len(alt.pieces)} parçalı bölünmüş kalıp"
                + (f" + {alt.n_cores} maça" if alt.n_cores else "")
            )
    else:
        plan = split_plan(ts, large, pull, s)
        n_cores = plan.n_cores
        unres = plan.unresolved
        reasons.insert(
            0,
            f"{len(large)} ters açı bölgesi maça için büyük veya karşılıklı iki yanda: "
            f"ayırma hattı ters açı sınırından geçirilerek {len(plan.pieces)} parçaya bölündü",
        )
        for k, p in enumerate(plan.pieces, 1):
            reasons.append(
                f"parça {k}: yüzeyin %{p.area_ratio * 100:.0f}'i, çekme {_fmt(p.pull.direction)}, "
                f"en küçük açı {p.pull.min_draft_deg:.1f}°"
                + (f", {len(p.cores)} maça" if p.cores else "")
                + (f", {len(p.unresolved)} çözülemeyen bölge" if p.unresolved else "")
            )
        if unres or len(plan.pieces) > s.max_split_pieces:
            solution = Solution.REVIEW
            why = (
                f"{len(unres)} bölge hiçbir tek yönde çıkmıyor (kapalı kesit → mandrel olabilir)"
                if unres
                else f"{len(plan.pieces)} parça > en fazla {s.max_split_pieces}"
            )
            flags.append(ReviewFlag(code="tooling_unresolved", reason=why, path=f"{LIB}:undercut"))
            reasons.insert(0, why)
        else:
            solution = Solution.SPLIT_CORE if n_cores else Solution.SPLIT
            if n_cores:
                reasons.append(
                    f"bölündükten sonra {n_cores} küçük ters açı kalıyor: bölünmüş kalıba ek "
                    "olarak maça gerekir"
                )
        flags.append(
            ReviewFlag(
                code="split_parting_line",
                reason="Ayırma hattı ters açı sınırından otomatik geçirildi; parça sayısı, hattı "
                "ve parçaların çıkış sırası mühendis tarafından onaylanmalı.",
                path=f"{LIB}:undercut.max_split_pieces",
            )
        )
    trace.append(
        CostLine(
            key="tooling.decision",
            description=f"Kalıp kararı: {SOLUTION_LABEL[solution]}",
            formula="ters açı yok → tek parça; tüm bölgeler < maça eşiği → maça; büyük/karşılıklı "
            "bölge → bölünmüş kalıp (parça başına kalan küçük bölge → + maça)",
            inputs=[
                _inp(s, "core_max_area_ratio"),
                _inp(s, "opposite_cos"),
                _inp(s, "max_split_pieces"),
            ],
            result=float(len(plan.pieces) if plan else 1),
            unit="parça",
            category="tooling",
            flags=list(flags),
        )
    )
    return ToolingAssessment(sc, regions, solution, plan, n_cores, reasons, alts, flags, trace)


def pull_angles_deg(d: tuple[float, float, float]) -> dict[str, float]:
    """Angle of the pull direction to each axis (deg), for the display."""
    v = np.asarray(d, dtype=float) / np.linalg.norm(d)
    return {ax: float(np.degrees(np.arccos(np.clip(v[i], -1, 1)))) for i, ax in enumerate("XYZ")}
