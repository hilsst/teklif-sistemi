"""Tessellated STEP faces for the 3D review screen, draft analysis and tool-surface selection.

The user picks the faces that form the tool surface (the customer's part surface decides which
side that is). From that selection come:
- tool-side draft angles for a pull direction (normals point out of the part; the tool is on the
  side the selected faces face, so draft = asin(-n . d) for pull direction d),
- the pull direction itself (the direction that releases the selection best),
- the selection's area and its **boundary length** (outer edge + every cut-out and hole, which
  is the trim length under the rule that all openings and edges are trimmed).
Everything here is an estimate from a tessellation and is flagged for review by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from teklif.ingest.step_props import StepUnavailableError, load_step_shape

Vec = NDArray[np.float64]
LINEAR_DEFLECTION_MM = 2.0
# 3 deg keeps a 2-5 m2 panel under ~5 s (0.5 deg took minutes); facet normals are good to ~1.5 deg
ANGULAR_DEFLECTION_DEG = 3.0
VIEW_DEFLECTION_MM = 5.0  # coarse mesh for the browser view only
VIEW_ANGLE_DEG = 10.0


@dataclass
class MeshFace:
    id: int
    area_mm2: float
    planar: bool
    vertices: NDArray[np.float64]  # (n, 3)
    triangles: NDArray[np.int64]  # (m, 3), wound so that normals point out of the part
    normals: NDArray[np.float64]  # (m, 3) unit normals per triangle
    tri_areas: NDArray[np.float64]  # (m,)
    edges: list[tuple[int, float]] = field(default_factory=list)  # (edge key, length mm)
    edge_polys: dict[int, NDArray[np.float64]] = field(default_factory=dict)  # key -> (n, 3)

    @property
    def mean_normal(self) -> Vec:
        w = (self.normals * self.tri_areas[:, None]).sum(axis=0)
        n = float(np.linalg.norm(w))
        return w / n if n > 0 else w


@dataclass(frozen=True)
class Outline:
    loops: list[NDArray[np.float64]]  # every closed boundary loop of the selection (EOP)
    open_chains: list[NDArray[np.float64]]  # boundary pieces that did not close (not filled)
    outer: int  # index of the outer rim in ``loops``
    ring: NDArray[np.float64] | None  # outer envelope on the surface (flange root)
    offset: NDArray[np.float64] | None  # MEOP: the envelope pushed out by the flange


def _cluster_points(pts: NDArray[np.float64], tol: float) -> list[int]:
    """Cluster id per point: points closer than ``tol`` end up together (STEP files often carry
    separate vertex copies at the same place, so edges cannot be joined by vertex identity)."""
    n = len(pts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: dict[tuple[int, int, int], list[int]] = {}
    for i, p in enumerate(pts):
        c = tuple(int(v) for v in np.floor(p / tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((c[0] + dx, c[1] + dy, c[2] + dz), []):
                        if float(np.linalg.norm(pts[j] - p)) <= tol:
                            parent[find(i)] = find(j)
        cells.setdefault((c[0], c[1], c[2]), []).append(i)
    return [find(i) for i in range(n)]


def _chain_loops(
    segs: list[NDArray[np.float64]], ends: list[tuple[object, object]]
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Join edge polylines end to end by their end-vertex keys: (closed loops, open chains)."""
    at: dict[object, list[int]] = {}
    for i, (ka, kb) in enumerate(ends):
        at.setdefault(ka, []).append(i)
        if kb != ka:
            at.setdefault(kb, []).append(i)
    used = [False] * len(segs)

    def grow(head: object, chain: list[NDArray[np.float64]]) -> object:
        while True:
            nxt = next((j for j in at.get(head, []) if not used[j]), None)
            if nxt is None:
                return head
            used[nxt] = True
            sg = segs[nxt]
            ka, kb = ends[nxt]
            if ka == head:
                head = kb
            else:
                sg, head = sg[::-1], ka
            chain.append(sg)
            if head == start_key:
                return head

    closed: list[NDArray[np.float64]] = []
    opened: list[NDArray[np.float64]] = []
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        chain = [segs[start]]
        start_key = ends[start][0]
        head = grow(ends[start][1], chain)
        if head == start_key:
            closed.append(np.vstack([c[:-1] for c in chain]))
        else:  # walk the other way from the start to pick up the rest of an open path
            back: list[NDArray[np.float64]] = []
            start_key = None
            grow(ends[start][0], back)
            pts = [c[::-1] for c in reversed(back)] + chain if back else chain
            opened.append(np.vstack(pts))
    return closed, opened


@dataclass
class FaceMesh:
    faces: list[MeshFace]

    def area_m2(self, ids: list[int]) -> float:
        return sum(self.faces[i].area_mm2 for i in ids) / 1e6

    def outer_perimeter_mm(self, ids: list[int], pull: Vec) -> float:
        """Perimeter of the selection's outline seen along pull (convex hull; ignores holes)."""
        d = _unit(np.asarray(pull, dtype=float))
        pts = np.vstack([self.faces[i].vertices for i in ids if len(self.faces[i].vertices)])
        a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        xy = np.unique(np.round(np.c_[pts @ a, pts @ b], 2), axis=0).astype(np.float64)
        hull = _convex_hull(xy)
        return float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum())

    def boundary_loops(
        self, ids: list[int]
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """Closed 3D polylines of the selection's boundary: the rim, cut-outs, holes, channels."""
        chosen = set(ids)
        owners: dict[int, int] = {}
        polys: dict[int, NDArray[np.float64]] = {}
        for i in chosen:
            for key, _ in self.faces[i].edges:
                owners[key] = owners.get(key, 0) + 1
            polys.update(self.faces[i].edge_polys)
        keys = [k for k, n in owners.items() if n == 1 and k in polys and len(polys[k]) > 1]
        segs = [polys[k] for k in keys]
        pts = np.array([p for sg in segs for p in (sg[0], sg[-1])])
        ids_ = _cluster_points(pts, 0.1)
        ends: list[tuple[object, object]] = [
            (ids_[2 * i], ids_[2 * i + 1]) for i in range(len(segs))
        ]
        return _chain_loops(segs, ends)

    def outline(self, ids: list[int], pull: Vec, flange_mm: float) -> Outline | None:
        """EOP loops (all boundary loops of the selection) and the MEOP outline: the largest loop's
        outer envelope seen along pull, pushed out by ``flange_mm`` at the same height."""
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

        outer = max(range(len(loops)), key=lambda i: area2d(loops[i]))
        pts = loops[outer]
        xy = np.c_[pts @ a, pts @ b]
        rounded = cast(list[list[float]], np.round(xy, 2).tolist())
        lookup: dict[tuple[float, float], NDArray[np.float64]] = {}
        for q, xyz in zip(rounded, list(pts), strict=True):
            old = lookup.get((q[0], q[1]))
            if old is None or float(xyz @ d) > float(old @ d):
                lookup[(q[0], q[1])] = xyz
        hull = _convex_hull(np.unique(np.round(xy, 2), axis=0).astype(np.float64))
        if len(hull) < 3:
            return Outline(loops, open_chains, outer, None, None)
        ring = np.array([lookup[(round(float(h[0]), 2), round(float(h[1]), 2))] for h in hull])
        cen = hull.mean(axis=0)
        n = len(hull)
        edge_n = []
        for i in range(n):
            e = hull[(i + 1) % n] - hull[i]
            nn = np.array([e[1], -e[0]])
            nn /= np.linalg.norm(nn) or 1.0
            if np.dot(nn, (hull[i] + hull[(i + 1) % n]) / 2 - cen) < 0:
                nn = -nn
            edge_n.append(nn)
        out = np.empty_like(ring)
        for i in range(n):
            n1, n2 = edge_n[i - 1], edge_n[i]
            m = (n1 + n2) / (1 + float(np.dot(n1, n2)))
            ml = float(np.linalg.norm(m))
            if ml > 2.0:  # sharp corner: cap the miter spike
                m = m * (2.0 / ml)
            out[i] = ring[i] + flange_mm * (m[0] * a + m[1] * b)
        return Outline(loops, open_chains, outer, ring, out)

    def boundary_length_mm(self, ids: list[int]) -> float:
        """Total length of edges with exactly one selected face on them (rim, cut-outs, holes)."""
        chosen = set(ids)
        per_edge: dict[int, list[float]] = {}
        owners: dict[int, int] = {}
        for i in chosen:
            for key, length in self.faces[i].edges:
                per_edge.setdefault(key, []).append(length)
                owners[key] = owners.get(key, 0) + 1
        return sum(lens[0] for key, lens in per_edge.items() if owners[key] == 1)


def _convex_hull(p: NDArray[np.float64]) -> NDArray[np.float64]:
    pts = sorted(map(tuple, p))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        h: list[tuple[float, float]] = []
        for q in seq:
            while len(h) >= 2 and (
                (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])
                <= 0
            ):
                h.pop()
            h.append(q)
        return h

    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def _unit(v: Vec) -> Vec:
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero-length direction")
    return v / n


def load_face_mesh(
    path: Path | str,
    deflection_mm: float = LINEAR_DEFLECTION_MM,
    angle_deg: float = ANGULAR_DEFLECTION_DEG,
    with_edges: bool = False,
) -> FaceMesh:
    """Read a STEP file and tessellate every face (needs OCP). ``with_edges`` also samples the
    edge polylines (only the 3D view needs them)."""
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GCPnts import GCPnts_QuasiUniformDeflection
        from OCP.GeomAbs import GeomAbs_Plane
        from OCP.GProp import GProp_GProps
        from OCP.IMeshTools import IMeshTools_Parameters
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc

    shape, _ = load_step_shape(path)
    params = IMeshTools_Parameters()
    params.Deflection = deflection_mm
    params.Angle = math.radians(angle_deg)
    params.Relative = False
    params.InParallel = True
    BRepMesh_IncrementalMesh(shape, params)

    faces: list[MeshFace] = []
    sampled: set[int] = set()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = _as_face(explorer.Current())
        explorer.Next()
        loc = TopLoc_Location()
        tri: Any = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:  # keep the id aligned with the B-rep face order
            faces.append(_empty_face(len(faces)))
            continue
        trsf = loc.Transformation()
        verts = np.array(
            [
                (p.X(), p.Y(), p.Z())
                for p in (tri.Node(i + 1).Transformed(trsf) for i in range(tri.NbNodes()))
            ]
        )
        tris = np.array(
            [tri.Triangle(i + 1).Get() for i in range(tri.NbTriangles())], dtype=np.int64
        )
        tris = tris - 1
        if face.Orientation() == TopAbs_REVERSED:
            tris = tris[:, [0, 2, 1]]
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        cross = np.cross(b - a, c - a)
        lens = np.linalg.norm(cross, axis=1)
        normals = cross / np.where(lens > 0, lens, 1.0)[:, None]
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        edges: list[tuple[int, float]] = []
        polys: dict[int, NDArray[np.float64]] = {}
        ex = TopExp_Explorer(face, TopAbs_EDGE)
        seen: set[int] = set()
        while ex.More():
            ed = ex.Current()
            key = hash(ed)
            lp = GProp_GProps()
            BRepGProp.LinearProperties_s(ed, lp)
            if key not in seen:
                seen.add(key)
                edges.append((key, lp.Mass()))
                if with_edges:
                    edge = _as_edge(ed)
                    if key not in sampled:
                        sampled.add(key)
                        polys[key] = _edge_polyline(
                            edge, BRepAdaptor_Curve, GCPnts_QuasiUniformDeflection, deflection_mm
                        )
            else:  # seam: used twice by this face, never a boundary
                edges.append((key, lp.Mass()))
                edges.append((key, lp.Mass()))
            ex.Next()
        faces.append(
            MeshFace(
                id=len(faces),
                area_mm2=props.Mass(),
                planar=BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane,
                vertices=verts,
                triangles=tris,
                normals=normals,
                tri_areas=lens / 2,
                edges=edges,
                edge_polys=polys,
            )
        )
    if not faces:
        raise StepUnavailableError("STEP file produced no tessellated faces")
    return FaceMesh(faces)


def _edge_polyline(
    edge: Any, adaptor: Any, sampler: Any, deflection_mm: float
) -> NDArray[np.float64]:
    try:
        curve = adaptor(edge)
        try:
            g = sampler(curve, max(deflection_mm / 2, 0.2))
            if not g.IsDone():
                raise RuntimeError("sampling failed")
            pts = [g.Value(i + 1) for i in range(g.NbPoints())]
        except Exception:  # fall back to the two ends
            pts = [curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())]
        return np.array([(p.X(), p.Y(), p.Z()) for p in pts], dtype=float)
    except Exception:  # an edge that cannot be sampled must not fail the whole load
        return np.zeros((0, 3))


def _empty_face(face_id: int) -> MeshFace:
    z3 = np.zeros((0, 3))
    return MeshFace(face_id, 0.0, False, z3, np.zeros((0, 3), dtype=np.int64), z3, np.zeros(0))


def _as_edge(shape: Any) -> Any:
    from OCP.TopoDS import TopoDS

    cast_ = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
    return cast_(shape)


def _as_face(shape: Any) -> Any:
    from OCP.TopoDS import TopoDS

    cast = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    return cast(shape)


# --- draft ---------------------------------------------------------------------------


def draft_angles_deg(mesh: FaceMesh, ids: list[int], pull: Vec) -> dict[int, NDArray[np.float64]]:
    """Per-triangle draft angle (deg) of the selected tool-side faces for a pull direction.

    Positive: releases; 0: vertical wall; negative: undercut (blocks release).
    """
    d = _unit(np.asarray(pull, dtype=float))
    return {i: np.degrees(np.arcsin(np.clip(-(mesh.faces[i].normals @ d), -1.0, 1.0))) for i in ids}


@dataclass(frozen=True)
class PullCandidate:
    direction: tuple[float, float, float]
    label: str
    undercut_area_ratio: float  # share of the selection with negative draft
    min_ok_ratio: float  # share of the selection at or above min_ok_deg


def _score(
    mesh: FaceMesh, ids: list[int], d: Vec, warn_min: float, min_ok: float
) -> tuple[float, float]:
    total = bad = ok = 0.0
    for deg, f in zip(
        draft_angles_deg(mesh, ids, d).values(), (mesh.faces[i] for i in ids), strict=True
    ):
        total += f.tri_areas.sum()
        bad += f.tri_areas[deg < 0.0].sum()
        ok += f.tri_areas[deg >= min_ok].sum()
    return (bad / total if total else 0.0), (ok / total if total else 0.0)


def suggest_pull_directions(
    mesh: FaceMesh, ids: list[int], warn_min_deg: float, min_ok_deg: float
) -> list[PullCandidate]:
    """Candidate pull directions for the selected tool surface, best first.

    Candidates: the mean of the inward selection normals and the six axes. Best = smallest share
    of undercut area, then largest share with enough draft.
    """
    if not ids:
        raise ValueError("no faces selected")
    total = np.zeros(3)
    for i in ids:
        f = mesh.faces[i]
        total += (f.normals * f.tri_areas[:, None]).sum(axis=0)
    cands: list[tuple[str, Vec]] = []
    if float(np.linalg.norm(total)) > 0:
        cands.append(("seçilen yüzeyin ortalama normali", _unit(-total)))
    for axis, name in ((0, "X"), (1, "Y"), (2, "Z")):
        for sign in (1.0, -1.0):
            v = np.zeros(3)
            v[axis] = sign
            cands.append((f"{'+' if sign > 0 else '-'}{name}", v))
    out: list[PullCandidate] = []
    for label, v in cands:
        bad, ok = _score(mesh, ids, v, warn_min_deg, min_ok_deg)
        out.append(PullCandidate((float(v[0]), float(v[1]), float(v[2])), label, bad, ok))
    out.sort(key=lambda c: (c.undercut_area_ratio, -c.min_ok_ratio))
    return out


def faces_facing(mesh: FaceMesh, direction: Vec, tol_deg: float) -> list[int]:
    """Faces whose mean outward normal is within tol_deg of direction (helper for selection)."""
    d = _unit(np.asarray(direction, dtype=float))
    lim = math.cos(math.radians(tol_deg))
    return [f.id for f in mesh.faces if float(f.mean_normal @ d) >= lim]


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
