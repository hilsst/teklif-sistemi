"""Tessellated STEP faces: per-face triangles with outward normals, draft angles, face helpers.

The user picks the faces that form the tool surface (the customer's part surface decides which
side that is). Conventions: triangle normals point out of the part, i.e. into the tool, so for a
pull direction d the tool-side draft angle is ``asin(-n . d)``; positive releases.

A ``FaceMesh`` comes from a STEP file (``load_face_mesh``, needs OCP / ``cadquery-ocp``) or from
plain arrays (``mesh_from_arrays``: any mesher, STL, trimesh...). Everything here is an estimate
from a tessellation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kalip_analiz.settings import MeshParams
from kalip_analiz.step import StepUnavailableError, load_step_shape

Vec = NDArray[np.float64]


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
    solid: int = 0  # which solid of an assembly the face belongs to (for hiding parts)

    @property
    def mean_normal(self) -> Vec:
        w = (self.normals * self.tri_areas[:, None]).sum(axis=0)
        n = float(np.linalg.norm(w))
        return w / n if n > 0 else w


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
    deflection_mm: float | None = None,
    angle_deg: float | None = None,
    with_edges: bool = False,
) -> FaceMesh:
    """Read a STEP file and tessellate every face (needs OCP). ``with_edges`` also samples the
    edge polylines (boundary loops and the 3D view need them). Deflections left ``None`` come
    from ``thresholds.yaml`` (``mesh.linear_deflection_mm``, ``mesh.angular_deflection_deg``)."""
    if deflection_mm is None or angle_deg is None:
        mp = MeshParams()
        deflection_mm = mp.linear_deflection_mm if deflection_mm is None else deflection_mm
        angle_deg = mp.angular_deflection_deg if angle_deg is None else angle_deg
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp, BRepGProp_Face
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GCPnts import GCPnts_QuasiUniformDeflection
        from OCP.GeomAbs import GeomAbs_Plane
        from OCP.GProp import GProp_GProps
        from OCP.IMeshTools import IMeshTools_Parameters
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
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

    solid_of: dict[int, int] = {}  # face hash -> solid index (assemblies have several solids)
    sol_ex = TopExp_Explorer(shape, TopAbs_SOLID)
    n_solid = 0
    while sol_ex.More():
        fe = TopExp_Explorer(sol_ex.Current(), TopAbs_FACE)
        while fe.More():
            solid_of.setdefault(hash(fe.Current()), n_solid)
            fe.Next()
        n_solid += 1
        sol_ex.Next()

    faces: list[MeshFace] = []
    poly_cache: dict[int, NDArray[np.float64]] = {}
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
        normals = _exact_normals(face, tri, trsf, tris, normals, BRepGProp_Face)
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
                    if key not in poly_cache:  # sampled once, given to every face on the edge
                        poly_cache[key] = _edge_polyline(
                            edge, BRepAdaptor_Curve, GCPnts_QuasiUniformDeflection, deflection_mm
                        )
                    polys[key] = poly_cache[key]
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
                solid=solid_of.get(hash(face), 0),
            )
        )
    if not faces:
        raise StepUnavailableError("STEP file produced no tessellated faces")
    return FaceMesh(faces)


def _exact_normals(
    face: Any,
    tri: Any,
    trsf: Any,
    tris: NDArray[np.int64],
    facet: NDArray[np.float64],
    face_props_cls: Any,
) -> NDArray[np.float64]:
    """Triangle normals from the true surface (mean of the vertex normals) instead of the flat
    facet: a cylinder wall tessellated coarsely has facets tilted by up to ~1 deg, which made a
    true 0 deg wall read as negative draft. Falls back to the facet normal where the surface
    normal is undefined (cone/sphere apex) or the face has no UV nodes."""
    if not tri.HasUVNodes():
        return facet
    from OCP.gp import gp_Pnt, gp_Vec

    fp = face_props_cls(face)
    vn = np.zeros((tri.NbNodes(), 3))
    ok = np.zeros(tri.NbNodes(), dtype=bool)
    for i in range(tri.NbNodes()):
        uv = tri.UVNode(i + 1)
        pnt, vec = gp_Pnt(), gp_Vec()
        try:
            fp.Normal(uv.X(), uv.Y(), pnt, vec)
        except Exception:
            continue
        v = np.array([vec.X(), vec.Y(), vec.Z()])
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            vn[i] = v / n
            ok[i] = True
    rot = trsf.VectorialPart()
    m = np.array([[rot.Value(r + 1, c + 1) for c in range(3)] for r in range(3)])
    vn = vn @ m.T
    mean = vn[tris].sum(axis=1)
    lens = np.linalg.norm(mean, axis=1)
    good = ok[tris].all(axis=1) & (lens > 1e-9)
    out = facet.copy()
    if good.any():
        cand = mean[good] / lens[good][:, None]
        if float((cand * facet[good]).sum(axis=1).mean()) < 0:
            cand = -cand
        out[good] = cand
    return out


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
    # rounded so a true 0 deg wall is not pushed to -1e-14 (and red) by floating point noise
    return {
        i: np.round(np.degrees(np.arcsin(np.clip(-(mesh.faces[i].normals @ d), -1.0, 1.0))), 6)
        + 0.0
        for i in ids
    }


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


def mesh_from_arrays(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    face_of_triangle: NDArray[np.int64] | None = None,
    normals: NDArray[np.float64] | None = None,
    with_edges: bool = True,
) -> FaceMesh:
    """A ``FaceMesh`` from a plain triangle mesh (no OCP needed).

    ``triangles`` must be wound so that ``(b - a) x (c - a)`` points out of the part (or pass
    ``normals``). ``face_of_triangle`` groups triangles into selectable faces (ids 0..n-1);
    without it every triangle is its own face. Face boundary edges are taken from the mesh, so
    boundary loops and trim lengths work too; ``with_edges=False`` skips them (the pull-direction
    and split/core analysis does not need edges and large meshes load much faster).
    """
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(triangles, dtype=np.int64)
    fo = np.arange(len(t)) if face_of_triangle is None else np.asarray(face_of_triangle)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    cross = np.cross(b - a, c - a)
    lens = np.linalg.norm(cross, axis=1)
    nrm = (
        cross / np.where(lens > 0, lens, 1.0)[:, None]
        if normals is None
        else np.asarray(normals, dtype=float)
    )
    # edge key = merged vertex pair, so faces sharing an edge share its key
    _, inv = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    vid = inv.ravel()
    n_faces = int(fo.max()) + 1 if len(fo) else 0
    faces: list[MeshFace] = []
    first = {int(vid[i]): i for i in range(len(v))} if with_edges else {}
    order = np.argsort(fo, kind="stable")
    bounds = np.searchsorted(fo[order], np.arange(n_faces + 1))
    for fid in range(n_faces):
        sel = order[bounds[fid] : bounds[fid + 1]]
        if not len(sel):
            faces.append(_empty_face(fid))
            continue
        used, local = np.unique(t[sel], return_inverse=True)
        ft = local.reshape(-1, 3).astype(np.int64)
        count: dict[tuple[int, int], int] = {}
        for tri in t[sel] if with_edges else ():
            for i in range(3):
                p, q = int(vid[tri[i]]), int(vid[tri[(i + 1) % 3]])
                k = (min(p, q), max(p, q))
                count[k] = count.get(k, 0) + 1
        edges: list[tuple[int, float]] = []
        polys: dict[int, NDArray[np.float64]] = {}
        for (p, q), n in count.items():
            if n != 1:  # interior edge of the face
                continue
            key = hash((p, q))
            seg = np.array([v[first[p]], v[first[q]]])
            edges.append((key, float(np.linalg.norm(seg[1] - seg[0]))))
            polys[key] = seg
        faces.append(
            MeshFace(
                id=fid,
                area_mm2=float(lens[sel].sum() / 2),
                planar=bool(np.allclose(nrm[sel], nrm[sel][0], atol=1e-6)),
                vertices=v[used],
                triangles=ft,
                normals=nrm[sel],
                tri_areas=lens[sel] / 2,
                edges=edges,
                edge_polys=polys,
            )
        )
    return FaceMesh(faces)
