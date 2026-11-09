"""Part features of the *selected faces* (area, holes, thickness), not of the whole STEP file.

A STEP file may hold an assembly of several parts, so the numbers shown at upload (whole-file
area, hole count, mean thickness) are not the numbers of the part the tool is built for. After
the user selects the part's mold-side faces, these are measured from the selection itself:

* area: area of the selected faces (one side of the part);
* holes: closed inner boundary loops of the selection through which a ray along the surface
  normal passes without touching the part (a through opening, not a pocket or a step);
* thickness: from sample points on the selected faces, a ray goes into the part along the
  inverse outward normal to the first surface it leaves through (the other skin); the 10 %
  trimmed mean of those distances is the mean thickness.

All three are measured on the fine tessellation, hence estimates; the panel flags them for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from teklif.geometry.surface_select import FaceMesh

F64 = NDArray[np.float64]

N_RAYS = 160  # sample points on the selection for the thickness
TRIM_FRACTION = 0.10  # share dropped at each end of the sorted distances
MIN_HIT_COS = 0.5  # a hit surface more oblique than this to the ray (edge, wall) is ignored
MIN_HIT_MM = 0.05  # closer hits are the start surface itself
HOLE_REACH_FACTOR = 3.0  # a hole centre ray must stay free for this many thicknesses
START_INSET_MM = 0.01
HOLE_REACH_FALLBACK_MM = 10.0  # hole ray length when the thickness could not be measured
HOLE_RAY_OFFSET_MM = 1.0  # the hole ray starts this far outside the surface
MIN_SAMPLES = 8  # fewer valid thickness rays than this: thickness is not trusted


@dataclass(frozen=True)
class SelectionFeatures:
    area_m2: float
    hole_count: int
    mean_thickness_mm: float  # 0 when it could not be measured
    thickness_p10_mm: float
    thickness_p90_mm: float
    n_rays: int
    n_hits: int
    n_inner_loops: int  # inner boundary loops (holes, pockets, steps, channels)
    solids: tuple[int, ...]
    notes: list[str] = field(default_factory=list)


def _solid_triangles(mesh: FaceMesh, solids: set[int]) -> tuple[F64, F64]:
    """(m, 3, 3) triangle corners and (m, 3) unit normals of all faces of the given solids."""
    tris: list[F64] = []
    norms: list[F64] = []
    for f in mesh.faces:
        if f.solid in solids and len(f.triangles):
            tris.append(f.vertices[f.triangles])
            norms.append(f.normals)
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.concatenate(tris), np.concatenate(norms)


class _Caster:
    """Ray casting against the triangles of the selected solids (bounding-sphere prefilter, then
    Moeller-Trumbore); the per-triangle arrays are built once and reused by every ray."""

    def __init__(self, tri: F64, nrm: F64) -> None:
        self.n = len(tri)
        self.nrm = nrm
        if self.n:
            self.centre = tri.mean(axis=1)
            self.radius = np.linalg.norm(tri - self.centre[:, None, :], axis=2).max(axis=1)
            self.v0 = tri[:, 0]
            self.e1 = tri[:, 1] - self.v0
            self.e2 = tri[:, 2] - self.v0

    def first_hits(
        self, origins: F64, dirs: F64, max_t: float | None = None, *, need_exit: bool
    ) -> F64:
        """Distance along each ray to the first triangle (inf when none). With ``need_exit`` the
        first surface whose outward normal points along the ray counts (the ray leaves material
        there); when that surface is more oblique than MIN_HIT_COS the ray is dropped (inf)."""
        out = np.full(len(origins), np.inf)
        if not self.n:
            return out
        reach = np.inf if max_t is None else max_t
        for k in range(len(origins)):
            o, d = origins[k], dirs[k]
            rel = self.centre - o
            along = rel @ d
            dist = np.linalg.norm(rel - along[:, None] * d, axis=1)
            cand = np.flatnonzero(
                (dist <= self.radius + 1e-6)
                & (along >= -self.radius)
                & (along <= reach + self.radius)
            )
            if not len(cand):
                continue
            e1, e2 = self.e1[cand], self.e2[cand]
            p = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, p)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            sv = o - self.v0[cand]
            u = np.einsum("ij,ij->i", sv, p) * inv
            q = np.cross(sv, e1)
            v = (q @ d) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            hit = (
                ok
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1 + 1e-9)
                & (t > MIN_HIT_MM)
                & (t <= reach)
            )
            if need_exit:
                cosx = self.nrm[cand] @ d
                hit &= cosx > 0
            if not hit.any():
                continue
            j = int(np.flatnonzero(hit)[np.argmin(t[hit])])
            if need_exit and float(cosx[j]) < MIN_HIT_COS:
                continue
            out[k] = float(t[j])
        return out


def _inside_point(lp: F64, n: F64) -> F64:
    """A point inside the closed loop ``lp`` (in its own plane) far from its boundary: the centroid
    when it lies inside, else the best of a grid over the loop (L or C shaped cut-outs)."""
    c = lp.mean(axis=0)
    a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    xy = np.c_[(lp - c) @ a, (lp - c) @ b]
    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 11)
    gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 11)
    pts = np.array([(x, y) for x in gx for y in gy])
    nxt = np.roll(xy, -1, axis=0)
    inside = np.zeros(len(pts), dtype=bool)
    for (x0, y0), (x1, y1) in zip(xy, nxt, strict=True):  # even-odd rule
        crosses = (y0 > pts[:, 1]) != (y1 > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (pts[:, 1] - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (pts[:, 0] < xc)
    if not inside.any():
        return c
    cand = pts[inside]
    clearance = np.linalg.norm(cand[:, None, :] - xy[None, ::3, :], axis=2).min(axis=1)
    best = cand[int(np.argmax(clearance))]
    return np.asarray(c + best[0] * a + best[1] * b, dtype=np.float64)


def selection_features(
    mesh: FaceMesh, ids: list[int], *, n_rays: int = N_RAYS
) -> SelectionFeatures:
    """Area, hole count and mean thickness of the selected faces of ``mesh`` (see module doc)."""
    chosen = sorted(set(ids))
    notes: list[str] = []
    area = mesh.area_m2(chosen)
    solids = tuple(sorted({mesh.faces[i].solid for i in chosen}))
    tri, nrm = _solid_triangles(mesh, set(solids))
    caster = _Caster(tri, nrm)

    # sample points: triangles of the selection drawn by area, at their centroid
    sel_tri = [mesh.faces[i] for i in chosen if len(mesh.faces[i].triangles)]
    corners = (
        np.concatenate([f.vertices[f.triangles] for f in sel_tri])
        if sel_tri
        else np.zeros((0, 3, 3))
    )
    snorm = np.concatenate([f.normals for f in sel_tri]) if sel_tri else np.zeros((0, 3))
    sarea = np.concatenate([f.tri_areas for f in sel_tri]) if sel_tri else np.zeros(0)
    thickness = 0.0
    p10 = p90 = 0.0
    hits = 0
    rays = 0
    if len(corners) and float(sarea.sum()) > 0:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(corners), size=n_rays, replace=True, p=sarea / sarea.sum())
        w = rng.dirichlet([1.0, 1.0, 1.0], size=n_rays)  # a random point inside each triangle
        centres = np.einsum("ij,ijk->ik", w, corners[pick])
        dirs = -snorm[pick]
        origins = centres + dirs * START_INSET_MM  # just inside the start surface
        dist = caster.first_hits(origins, dirs, need_exit=True)
        rays = len(pick)
        dist = dist + START_INSET_MM
        good = np.sort(dist[np.isfinite(dist)])
        hits = len(good)
        if hits >= MIN_SAMPLES:
            cut = int(hits * TRIM_FRACTION)
            core = good[cut : hits - cut] if hits - 2 * cut >= 1 else good
            thickness = float(core.mean())
            p10, p90 = float(np.percentile(good, 10)), float(np.percentile(good, 90))
            if p90 > 1.5 * max(p10, 1e-9):
                notes.append(
                    f"kalınlık değişken: %10–%90 aralığı {p10:.1f}–{p90:.1f} mm, ortalama "
                    "budanmış ortalamadır"
                )
        else:
            notes.append(
                f"kalınlık ölçülemedi: {rays} ışından yalnız {hits} tanesi karşı yüzeye ulaştı "
                "(açık yüzey veya kalın/kapalı olmayan model)"
            )

    # holes: inner closed boundary loops that are open all the way through the part
    holes = 0
    inner = 0
    loops, open_chains = mesh.boundary_loops(chosen)
    if loops:
        reach = HOLE_REACH_FACTOR * thickness if thickness > 0 else HOLE_REACH_FALLBACK_MM
        size = [float(np.linalg.norm(lp.max(axis=0) - lp.min(axis=0))) for lp in loops]
        outer = int(np.argmax(size))  # the rim is the loop that spans the most
        cc = corners.mean(axis=1) if len(corners) else np.zeros((0, 3))
        for k, lp in enumerate(loops):
            if k == outer or len(lp) < 3:
                continue
            inner += 1
            c0 = lp.mean(axis=0)
            near = int(np.argmin(np.linalg.norm(cc - c0, axis=1))) if len(cc) else 0
            n = np.linalg.svd(lp - c0)[2][2]  # best-fit plane normal of the loop
            if len(snorm) and float(n @ snorm[near]) < 0:
                n = -n
            c = _inside_point(lp, n)
            o = (c + n * HOLE_RAY_OFFSET_MM)[None]
            free = caster.first_hits(o, -n[None], reach + HOLE_RAY_OFFSET_MM, need_exit=False)
            if not np.isfinite(free[0]):
                holes += 1
        if inner > holes:
            notes.append(
                f"{inner - holes} iç sınır (cep, basamak veya kanal) delik sayılmadı: ortasından "
                "geçen ışın parçaya çarptı"
            )
    if open_chains:
        notes.append(f"{len(open_chains)} kapanmayan sınır parçası delik sayımına katılmadı")
    if len(solids) > 1:
        notes.append(
            f"seçim {len(solids)} gövdeden yüzler içeriyor ({', '.join(map(str, solids))})"
        )
    return SelectionFeatures(
        area_m2=area,
        hole_count=holes,
        mean_thickness_mm=thickness,
        thickness_p10_mm=p10,
        thickness_p90_mm=p90,
        n_rays=rays,
        n_hits=hits,
        n_inner_loops=inner,
        solids=solids,
        notes=notes,
    )
