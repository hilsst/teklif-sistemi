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

    @property
    def mean_normal(self) -> Vec:
        w = (self.normals * self.tri_areas[:, None]).sum(axis=0)
        n = float(np.linalg.norm(w))
        return w / n if n > 0 else w


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

    def flange_rings(
        self, ids: list[int], pull: Vec, flange_mm: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """EOP and MEOP outlines in 3D: the selection's outline seen along pull (convex hull,
        vertices on the surface) and that outline pushed out by ``flange_mm`` at the same height."""
        d = _unit(np.asarray(pull, dtype=float))
        pts = np.vstack([self.faces[i].vertices for i in ids if len(self.faces[i].vertices)])
        a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        xy = np.c_[pts @ a, pts @ b]
        lookup: dict[tuple[float, float], NDArray[np.float64]] = {}
        rounded = cast(list[list[float]], np.round(xy, 2).tolist())
        for q, xyz in zip(rounded, list(pts), strict=True):
            lookup.setdefault((q[0], q[1]), xyz)
        hull = _convex_hull(np.unique(np.round(xy, 2), axis=0).astype(np.float64))
        if len(hull) < 3:
            return None
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
            out[i] = ring[i] + flange_mm * (m[0] * a + m[1] * b)
        return ring, out

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
) -> FaceMesh:
    """Read a STEP file and tessellate every face (needs OCP)."""
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
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
            )
        )
    if not faces:
        raise StepUnavailableError("STEP file produced no tessellated faces")
    return FaceMesh(faces)


def _empty_face(face_id: int) -> MeshFace:
    z3 = np.zeros((0, 3))
    return MeshFace(face_id, 0.0, False, z3, np.zeros((0, 3), dtype=np.int64), z3, np.zeros(0))


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
