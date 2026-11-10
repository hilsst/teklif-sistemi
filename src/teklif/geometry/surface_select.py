"""Tool-surface selection on tessellated STEP faces (teklif side).

The mesh, draft angles and selection helpers live in the standalone ``kalip_analiz`` package
(``kalip_analiz.mesh``); they are re-exported here so existing imports keep working. This module
adds what only the quoting system needs: the tool-profile outline (EOP ring for
``teklif.geometry.mold_profile``) and the trim lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kalip_analiz import mesh as _km
from kalip_analiz.mesh import MeshFace as MeshFace
from kalip_analiz.mesh import PullCandidate as PullCandidate
from kalip_analiz.mesh import _chain_loops as _chain_loops
from kalip_analiz.mesh import _cluster_points as _cluster_points
from kalip_analiz.mesh import _convex_hull as _convex_hull
from kalip_analiz.mesh import _score as _score
from kalip_analiz.mesh import _unit
from kalip_analiz.mesh import draft_angles_deg as draft_angles_deg
from kalip_analiz.mesh import faces_facing as faces_facing
from kalip_analiz.mesh import suggest_pull_directions as suggest_pull_directions
from teklif.geometry import mold_profile as mp

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class Outline:
    loops: list[NDArray[np.float64]]  # every closed boundary loop of the selection (EOP)
    open_chains: list[NDArray[np.float64]]  # boundary pieces that did not close (not filled)
    outer: int  # index of the outer rim in ``loops``
    # the outer rim (EOP of the tool profile), thinned; a point is repeated where a corner is
    # rounded (see teklif.geometry.mold_profile)
    ring: NDArray[np.float64] | None
    # per ring point: direction the surface continues past the edge; unit, except at a concave
    # corner, where it is the mitre (longer) so the MEOP/turn lines stay at their exact distance
    tangent: NDArray[np.float64] | None
    # per ring point: outward flange direction in the pull plane (unit; mitre at concave corners).
    # The flange edge itself is the exact offset of the turn line (mold_profile.flange_outline);
    # this is only its fallback
    out_dir: NDArray[np.float64] | None


@dataclass
class FaceMesh(_km.FaceMesh):
    """``kalip_analiz.mesh.FaceMesh`` plus the tool-profile outline and the trim lengths."""

    def outline(self, ids: list[int], pull: Vec) -> Outline | None:
        """EOP loops (all boundary loops of the selection) and the tool profile frame: the largest
        loop itself (the real trim line, not a hull), with, per point, the direction the surface
        continues past the edge and the outward direction in the pull plane."""
        loops, open_chains = self.boundary_loops(ids)
        if not loops:
            return None
        d = _unit(np.asarray(pull, dtype=float))
        a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)

        def area2d(lp: NDArray[np.float64]) -> float:
            x, y = lp @ a, lp @ b
            return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)

        areas = [area2d(lp) for lp in loops]
        # largest outline seen along pull; walls parallel to the pull give equal areas at both
        # rims, then the rim towards the mold opening (smallest position along pull) is the EOP
        tied = [i for i, a_ in enumerate(areas) if a_ >= 0.9 * max(areas)]
        outer = min(tied, key=lambda i: float((loops[i] @ d).mean()))
        edges = mp.mesh_boundary_edges(
            [(self.faces[i].vertices, self.faces[i].triangles, self.faces[i].normals) for i in ids]
        )
        corner_deg, max_pts = 5.0, 300
        while True:
            rim = mp.decimate_loop(loops[outer], max_pts, corner_deg)
            if len(rim) < 3:
                return Outline(loops, open_chains, outer, None, None, None)
            cont = mp.segment_continuation(rim, edges)
            pts, tan = mp.join_corners(rim, cont)
            idx, out_dir = mp.flange_dirs(pts, tan, d)
            # the offset clean-up grows with the square of the ring size: a serrated rim is thinned
            # harder (larger corner angle, fewer points) until the ring stays workable
            if len(idx) <= mp.MAX_RING_POINTS or corner_deg >= 160.0:
                break
            corner_deg, max_pts = corner_deg * 2, max(60, max_pts // 2)
        return Outline(loops, open_chains, outer, pts[idx], tan[idx], out_dir)

    def trim_lengths_mm(self, ids: list[int], pull: Vec) -> tuple[float, float]:
        """(all boundary loops, outer loop) trim lengths, mm. A rim of a shell has an inner and an
        outer edge on the planar cap face next to the selection (wall thickness); the longer of the
        two is used, as the trim runs along the part's open edge."""
        loops, _ = self.boundary_loops(ids)
        if not loops:
            return self.boundary_length_mm(ids), self.outer_perimeter_mm(ids, pull)

        def length(lp: NDArray[np.float64]) -> float:
            return float(np.linalg.norm(lp - np.roll(lp, -1, axis=0), axis=1).sum())

        chosen = set(ids)
        owners: dict[int, int] = {}
        for i in chosen:
            for key, _ in self.faces[i].edges:
                owners[key] = owners.get(key, 0) + 1
        rim_keys = {k for k, n in owners.items() if n == 1}
        caps = [
            f.id
            for f in self.faces
            if f.id not in chosen and f.planar and any(k in rim_keys for k, _ in f.edges)
        ]
        cap_loops: list[NDArray[np.float64]] = []
        for c in caps:
            cap_loops += self.boundary_loops([c])[0]

        def extra(lp: NDArray[np.float64]) -> float:
            """How much longer the twin edge is than this rim (0 if it is not longer)."""
            own = length(lp)
            best = own
            if not cap_loops:
                return 0.0
            match = min(
                cap_loops,
                key=lambda c: float(
                    np.linalg.norm(c[:: max(1, len(c) // 20), None] - lp[None, ::7], axis=2)
                    .min(axis=0)
                    .mean()
                ),
            )
            probe = lp[:: max(1, len(lp) // 20)]
            if float(np.linalg.norm(match[None] - probe[:, None], axis=2).min(axis=1).max()) > 1.0:
                return 0.0  # no cap loop coincides with this rim
            for c in cap_loops:
                if c is match:
                    continue
                gap = float(np.linalg.norm(c[None] - probe[:, None], axis=2).min(axis=1).max())
                if gap <= 100.0:  # the other edge of the same wall
                    best = max(best, length(c))
            return best - own

        adds = [extra(lp) for lp in loops]
        oi = self.outline(ids, pull)
        outer_add = adds[oi.outer] if oi is not None else 0.0
        return (
            self.boundary_length_mm(ids) + sum(adds),
            self.outer_perimeter_mm(ids, pull) + outer_add,
        )


def load_face_mesh(
    path: Path | str,
    deflection_mm: float | None = None,
    angle_deg: float | None = None,
    with_edges: bool = False,
) -> FaceMesh:
    """``kalip_analiz.mesh.load_face_mesh`` (deflections from ``thresholds.yaml`` when None)."""
    return FaceMesh(_km.load_face_mesh(path, deflection_mm, angle_deg, with_edges).faces)


def triangulate_polygon(xy: NDArray[np.float64]) -> list[tuple[int, int, int]]:
    """Ear clipping of a simple polygon (any orientation). Empty when it cannot be done."""
    n = len(xy)
    if n < 3 or n > 600:
        return []
    x, y = xy[:, 0], xy[:, 1]
    sign = 1.0 if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0 else -1.0
    idx = list(range(n))
    out: list[tuple[int, int, int]] = []

    def cross(a: int, b: int, c: int) -> float:
        return float((x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])) * sign

    def inside(p: int, a: int, b: int, c: int) -> bool:
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(p, a, b, c) for p in idx if p not in (a, b, c)):
                continue
            out.append((a, b, c))
            idx.pop(k)
            break
        else:
            return []
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))
    return out
