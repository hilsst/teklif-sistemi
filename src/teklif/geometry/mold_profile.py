"""Mold (tool) profile around the outer trim line (EOP) of the selected mold surface.

Kemal's rule (2026-11-05/06), per point of the EOP:
- the surface goes on past the EOP in its own direction ``to_meop`` (20 mm) to the MEOP line,
  then ``to_turn`` (40 mm, 30-50) more to the flange turn line,
- there the flange turns and runs ``flange`` (150 mm) outward in a plane perpendicular to the pull
  ("flanşlar çekme yönüne her zaman dik").

The geometry has to survive real parts: rims that switch between a top plane and a wall (the
continuation direction flips there), concave cut-outs tighter than the flange (a plain offset
folds over itself), and rim pieces running along the pull (zero length seen along the pull).
So:
- the continuation is taken per rim *segment* from the mesh triangle that owns that edge (found
  near the segment's ends: the rim is sampled coarser than the mesh, so a chord across a small
  round lies off the meshed arc in its middle),
- at a vertex, nearly equal directions are averaged; a convex change (the extensions open a gap)
  becomes a fan of copies of the vertex (a rounded corner), a concave change (they overlap)
  becomes one mitred point, so both offset lines keep their exact distance,
- the MEOP and turn lines are untangled the same way: where one crosses itself (a cut-out
  tighter than the offset, a notch narrower than two offsets) the loop is cut off, and points
  left inside another part's band go onto the envelope,
- the flange's outer edge is not built point by point (per-point outward directions are noisy:
  the edge zigzagged beside straight EOP runs and folded over at steps, Kemal 2026-11-06). It is
  the exact offset, in the plane perpendicular to the pull, of what the turn line covers seen
  along the pull: the outer boundary of the turn line, every edge moved out by the flange, arcs
  round convex corners, then the outer boundary of that curve (``flange_outline``). So it is
  straight beside straight runs, round at convex corners, one clean corner at concave ones, never
  crosses itself, and noise far below the flange width cannot show. The turn line is then paired
  with it by a monotone matching that keeps every strip triangle turned the same way
  (``pair_rings``), so the flange strip covers its band once: no fold, no overlap. The rings
  repeat points where one side needs more of them than the other (a rounded corner's arc, a
  concave corner's fan).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]

SMOOTH_DEG = 10.0  # continuation changes up to this are averaged into one point
FAN_STEP_DEG = 10.0  # angular step of a rounded (fan) corner
PLAN_SMOOTH_DEG = 12.0  # plan corners up to this get a plain mitre for the flange direction
MITRE_CAP = 3.0  # longest mitre (x the distance) at a concave corner


MAX_RING_POINTS = 1500  # offset clean-up is O(m^2): thin rims with more points than this


def plane_frame(pull: F) -> tuple[F, F, F]:
    """(unit pull d, a, b): a and b span the plane perpendicular to the pull."""
    d = np.asarray(pull, dtype=float)
    d = d / np.linalg.norm(d)
    a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return d, a, np.cross(d, a)


def signed_area(xy: F) -> float:
    return float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))


def decimate_loop(loop: F, max_pts: int = 300, corner_deg: float = 5.0) -> F:
    """Loop points without duplicates, thinned: every 3D corner (> 5 deg, which includes the rim
    turning from a plane edge onto a wall edge) is kept, smooth runs at most every perimeter/300."""
    pts = [loop[0]]
    for p in loop[1:]:
        if float(np.linalg.norm(p - pts[-1])) > 0.05:
            pts.append(p)
    if len(pts) > 1 and float(np.linalg.norm(pts[0] - pts[-1])) <= 0.05:
        pts.pop()
    arr = np.array(pts)
    n = len(arr)
    if n <= max_pts:
        return arr
    e = np.roll(arr, -1, axis=0) - arr
    ln = np.linalg.norm(e, axis=1)
    step = float(ln.sum()) / max_pts
    cos5 = math.cos(math.radians(corner_deg))
    keep = [0]
    acc = 0.0
    for i in range(1, n):
        acc += float(ln[i - 1])
        den = float(ln[i - 1] * ln[i])
        corner = den > 0 and float(e[i - 1] @ e[i]) / den < cos5
        if corner or acc >= step:
            keep.append(i)
            acc = 0.0
    return arr[keep]


def mesh_boundary_edges(
    faces: list[tuple[F, NDArray[np.int64], F]],
) -> tuple[F, F, F, F]:
    """Edges used by a single triangle of their face, as (start, end, triangle normal, opposite
    vertex) arrays: the triangles that own the face boundaries (rim and seams)."""
    sa, sb, sn, so = [], [], [], []
    for verts, tris, normals in faces:
        m = len(tris)
        if m == 0:
            continue
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        opp = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
        owner = np.tile(np.arange(m), 3)
        lo, hi = e.min(axis=1), e.max(axis=1)
        key = lo * (len(verts) + 1) + hi
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        once = cnt[inv] == 1
        sa.append(verts[e[once, 0]])
        sb.append(verts[e[once, 1]])
        sn.append(normals[owner[once]])
        so.append(verts[opp[once]])
    if not sa:
        z = np.zeros((0, 3))
        return z, z, z, z
    return np.vstack(sa), np.vstack(sb), np.vstack(sn), np.vstack(so)


def nearest_segment(q: F, a: F, b: F, k: int = 32) -> tuple[NDArray[np.int64], F]:
    """For each query point: index of the nearest segment a[i]-b[i] and the distance to it.
    Candidates are the k segments with the smallest lower bound |q - mid| - half length (one
    matrix product), then the exact distance among those."""
    mid = (a + b) / 2
    half = np.linalg.norm(b - a, axis=1) / 2
    k = min(k, len(a))
    mm = (mid * mid).sum(axis=1)
    idx = np.empty(len(q), dtype=np.int64)
    dist = np.empty(len(q))
    step = max(1, 4_000_000 // max(len(a), 1))
    for s in range(0, len(q), step):
        qq = q[s : s + step]
        d2 = (qq * qq).sum(axis=1)[:, None] + mm[None] - 2.0 * (qq @ mid.T)
        bound = np.sqrt(np.maximum(d2, 0.0)) - half[None]
        if k < len(a):
            cand = np.argpartition(bound, k - 1, axis=1)[:, :k]
        else:
            cand = np.tile(np.arange(len(a)), (len(qq), 1))
        ca, cb = a[cand], b[cand]
        ab = cb - ca
        rel = qq[:, None, :] - ca
        t = np.clip((rel * ab).sum(axis=2) / np.maximum((ab * ab).sum(axis=2), 1e-18), 0.0, 1.0)
        diff = rel - t[..., None] * ab
        dd = (diff * diff).sum(axis=2)
        j = np.argmin(dd, axis=1)
        rows = np.arange(len(qq))
        best = cand[rows, j]
        bd = np.sqrt(dd[rows, j])
        if k < len(a):
            # exact: a segment outside the k candidates is nearer only if its lower bound is below
            # the best distance found; those (rare) queries are redone against every segment
            member = np.zeros(bound.shape, dtype=bool)
            np.put_along_axis(member, cand, True, axis=1)
            for r in np.nonzero(((bound < bd[:, None]) & ~member).any(axis=1))[0]:
                ab_all = b - a
                rel_all = qq[r] - a
                t_all = np.clip(
                    (rel_all * ab_all).sum(axis=1)
                    / np.maximum((ab_all * ab_all).sum(axis=1), 1e-18),
                    0.0,
                    1.0,
                )
                d_all = ((rel_all - t_all[:, None] * ab_all) ** 2).sum(axis=1)
                best[r] = int(np.argmin(d_all))
                bd[r] = float(np.sqrt(d_all.min()))
        idx[s : s + step] = best
        dist[s : s + step] = bd
    return idx, dist


def segment_continuation(ring: F, edges: tuple[F, F, F, F]) -> F:
    """Per ring segment (point k to k+1): the unit direction in which the surface goes on past
    it, in the plane of the triangle that owns the edge, perpendicular to the edge, away from the
    triangle (a wall parallel to the pull continues along the pull, a flat sheet outward)."""
    ea, eb, en, eo = edges
    nxt = np.roll(ring, -1, axis=0)
    seg = nxt - ring
    sl = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = seg / np.where(sl > 0, sl, 1.0)
    out = np.zeros_like(ring)
    if len(ea) == 0:
        return out
    # the mesh edge that owns the segment: looked up just inside either end of the segment, which
    # (a sampled point of the true trim curve) the mesh boundary passes through, and the nearer
    # match kept. Not at the segment's midpoint: the trim edge is sampled far coarser than the
    # face mesh, so a chord across a small fillet lies up to the sampling deflection (~2 mm on
    # Kemal's TP0007903) inside the meshed arc, nearer to the far side of a narrow strip
    eps = np.minimum(0.5, 0.25 * sl)
    i0, d0 = nearest_segment(ring + eps * seg, ea, eb)
    i1, d1 = nearest_segment(nxt - eps * seg, ea, eb)
    idx = np.where(d0 <= d1, i0, i1)
    c = np.cross(seg, en[idx])
    cl = np.linalg.norm(c, axis=1)
    # inside the surface: from the owning mesh edge towards the triangle's far vertex, measured
    # across that edge. Not from the ring segment's midpoint: the far vertex of a thin fan
    # triangle along a meshed arc is itself on the arc, so seen from the chord it can lie on the
    # wrong side (the continuation then pointed into the part)
    e_dir = eb[idx] - ea[idx]
    e_dir = e_dir / np.maximum(np.linalg.norm(e_dir, axis=1, keepdims=True), 1e-12)
    rel = eo[idx] - ea[idx]
    inward = rel - (rel * e_dir).sum(axis=1, keepdims=True) * e_dir
    for k in range(len(ring)):
        if cl[k] > 1e-6:
            v = c[k] / cl[k]
            out[k] = -v if float(v @ inward[k]) > 0 else v
        else:  # degenerate: straight away from the owning triangle
            v = -inward[k] + float(inward[k] @ seg[k]) * seg[k]
            vl = float(np.linalg.norm(v))
            out[k] = v / vl if vl > 1e-9 else 0.0
    return out


def _rotate(v: F, axis: F, ang: float) -> F:
    """Rodrigues rotation of v about the unit axis."""
    ca, sa = math.cos(ang), math.sin(ang)
    return v * ca + np.cross(axis, v) * sa + axis * float(axis @ v) * (1 - ca)


def join_corners(ring: F, cont: F) -> tuple[F, F]:
    """Per-vertex continuation from the per-segment one: (points, tangents), where a point is
    repeated for a fan at a convex corner. Tangents are unit except at concave corners, where they
    are the mitre (1/cos(half angle), capped) so the MEOP/turn lines keep their exact distance from
    both edges."""
    n = len(ring)
    seg = np.roll(ring, -1, axis=0) - ring
    sl = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = seg / np.where(sl > 0, sl, 1.0)
    smooth = math.radians(SMOOTH_DEG)
    step = math.radians(FAN_STEP_DEG)
    pts: list[F] = []
    tans: list[F] = []
    for k in range(n):
        c1, c2 = cont[k - 1], cont[k]
        if not c1.any() or not c2.any():
            t = c1 if c1.any() else c2
            pts.append(ring[k])
            tans.append(t)
            continue
        dot = float(np.clip(c1 @ c2, -1.0, 1.0))
        th = math.acos(dot)
        s = c1 + c2
        sn = float(np.linalg.norm(s))
        if th <= smooth:
            pts.append(ring[k])
            tans.append(s / sn)
        elif float(c2 @ seg[k - 1]) - float(c1 @ seg[k]) > 0:  # convex: the extensions part
            axis = np.cross(c1, c2)
            al = float(np.linalg.norm(axis))
            if al > 1e-6:
                axis = axis / al
            else:  # opposite directions: turn about the rim
                e = seg[k - 1] + seg[k]
                e = e - float(e @ c1) * c1
                el = float(np.linalg.norm(e))
                axis = e / el if el > 1e-9 else np.cross(c1, [0.0, 0.0, 1.0])
                axis = axis / (float(np.linalg.norm(axis)) or 1.0)
            steps = math.ceil(th / step)
            for j in range(steps + 1):
                pts.append(ring[k])
                tans.append(_rotate(c1, axis, th * j / steps))
        else:  # concave: the extensions overlap, one mitred point
            pts.append(ring[k])
            if sn < 1e-6:
                tans.append(c1)
            else:
                tans.append(s / sn * min(1.0 / math.cos(th / 2), MITRE_CAP))
    return np.array(pts), np.array(tans)


def _rot2(v: F, ang: float) -> F:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _mitre2(n1: F, n2: F) -> F:
    s = n1 + n2
    sn = float(np.linalg.norm(s))
    if sn < 1e-9:
        return n1
    dot = float(np.clip(n1 @ n2, -1.0, 1.0))
    half = math.acos(dot) / 2
    return s / sn * min(1.0 / max(math.cos(half), 1e-9), MITRE_CAP)


def flange_dirs(pts: F, tangent: F, pull: F) -> tuple[NDArray[np.int64], F]:
    """Outward flange direction in the pull plane for each output point: (index into pts per
    output point, 3D direction). The direction is the outward normal of the EOP seen along the
    pull; points that coincide in plan (a rim piece running along the pull) share one plan corner.
    A convex plan corner is rounded with extra copies of a point (the flange edge becomes an arc);
    a concave one gets the mitre."""
    _, a, b = plane_frame(pull)
    n = len(pts)
    xy = np.c_[pts @ a, pts @ b]
    t2 = np.c_[tangent @ a, tangent @ b]
    tl = np.linalg.norm(tangent, axis=1)
    step2 = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    step3 = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    joined = (step2 < 0.5) | (step2 < 0.25 * step3)  # joined[i]: i and i+1 share a plan point
    if joined.all() or n < 3:
        d2 = t2 / np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12)
        return np.arange(n), d2[:, :1] * a + d2[:, 1:] * b
    start = (int(np.argmin(joined)) + 1) % n  # first point after a real plan step
    # a group runs on while the points stay at one plan point compared with the way they go
    # along the rim (a wall edge along the pull, also a drafted or curved one)
    groups: list[list[int]] = [[start]]
    path = 0.0
    for s in range(1, n):
        i = (start + s) % n
        path += float(step3[(i - 1) % n])
        far = float(np.linalg.norm(xy[i] - xy[groups[-1][0]]))
        if far < 0.5 or far < 0.3 * path:
            groups[-1].append(i)
        else:
            groups.append([i])
            path = 0.0
    sign = 1.0 if signed_area(xy) > 0 else -1.0
    g_n = len(groups)
    normals = []
    for g in range(g_n):
        e = xy[groups[(g + 1) % g_n][0]] - xy[groups[g][-1]]
        el = float(np.linalg.norm(e)) or 1.0
        normals.append(sign * np.array([e[1], -e[0]]) / el)
    plan_smooth = math.radians(PLAN_SMOOTH_DEG)
    step = math.radians(FAN_STEP_DEG)
    idx: list[int] = []
    dirs: list[F] = []
    for g in range(g_n):
        nin, nout = normals[g - 1], normals[g]
        th = sign * math.atan2(nin[0] * nout[1] - nin[1] * nout[0], float(nin @ nout))
        members = groups[g]
        if th <= plan_smooth:  # straight, gently convex or concave: one mitred direction
            m = _mitre2(nin, nout)
            for i in members:
                idx.append(i)
                dirs.append(m)
            continue
        # convex plan corner: follow the members' own surface direction where it lies in the
        # corner's range (a flat sheet's rounded corner), round off where it jumps
        cand: list[float | None] = []
        for i in members:
            ln = float(np.linalg.norm(t2[i]))
            c: float | None = None
            if tl[i] > 0 and ln >= 0.5 * tl[i]:
                u = t2[i] / ln
                phi = sign * math.atan2(nin[0] * u[1] - nin[1] * u[0], float(nin @ u))
                if -0.1 <= phi <= th + 0.1:
                    c = min(max(phi, 0.0), th)
            cand.append(c)
        known = [j for j, c in enumerate(cand) if c is not None]
        alphas: list[float] = []
        for j, c in enumerate(cand):
            if c is not None:
                alphas.append(c)
            elif not known or j < known[0]:
                alphas.append(0.0)
            elif j > known[-1]:
                alphas.append(th)
            else:
                alphas.append(alphas[-1])
        prev = 0.0
        for i, al in zip(members, alphas, strict=True):
            if al - prev > plan_smooth:  # fill the jump with a fan around this point
                k = math.ceil((al - prev) / step)
                for s in range(1, k):
                    idx.append(i)
                    dirs.append(_rot2(nin, sign * (prev + (al - prev) * s / k)))
            idx.append(i)
            dirs.append(_rot2(nin, sign * al))
            prev = max(prev, al)
        if th - prev > 1e-9:
            k = max(1, math.ceil((th - prev) / step))
            for s in range(1, k + 1):
                idx.append(members[-1])
                dirs.append(_rot2(nin, sign * (prev + (th - prev) * s / k)))
    d2 = np.array(dirs)
    return np.array(idx, dtype=np.int64), d2[:, :1] * a + d2[:, 1:] * b


def _groups(p: F, tol: float) -> tuple[NDArray[np.int64], F]:
    """Group id per point of a closed ring (consecutive points closer than tol share one) and
    the unique points."""
    n = len(p)
    gid = np.zeros(n, dtype=np.int64)
    reps = [0]
    for i in range(1, n):
        if float(np.linalg.norm(p[i] - p[reps[-1]])) > tol:
            reps.append(i)
        gid[i] = len(reps) - 1
    if len(reps) > 1 and float(np.linalg.norm(p[reps[-1]] - p[0])) <= tol:
        gid[gid == len(reps) - 1] = 0
        reps.pop()
    return gid, p[reps]


def _closest(p1: F, q1: F, p2: F, q2: F) -> tuple[float, F]:
    """Distance between segments p1-q1 and p2-q2 and the midpoint of their closest points
    (Ericson, Real-Time Collision Detection 5.1.9; handles parallel segments)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    den = a * e - b * b
    s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 * a * e else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2)), (c1 + c2) / 2


def _crossings(u: F, tol: float) -> list[tuple[int, int, F]]:
    """Pairs of non-adjacent segments of the closed ring u (2D or 3D) that cross or touch within
    tol: (i, j, crossing point) with i < j; segment i runs from u[i] to u[i+1]."""
    m = len(u)
    v = np.roll(u, -1, axis=0)
    lo = np.minimum(u, v) - tol
    hi = np.maximum(u, v) + tol
    over = np.ones((m, m), dtype=bool)
    for k in range(u.shape[1]):
        over &= (lo[:, None, k] <= hi[None, :, k]) & (lo[None, :, k] <= hi[:, None, k])
    over = np.triu(over, 2)
    over[0, m - 1] = False
    out: list[tuple[int, int, F]] = []
    for i, j in zip(*(x.tolist() for x in np.nonzero(over)), strict=True):
        if float(np.linalg.norm(v[i] - u[i])) < 1e-12 or float(np.linalg.norm(v[j] - u[j])) < 1e-12:
            continue
        dist, x = _closest(u[i], v[i], u[j], v[j])
        if dist <= tol:
            out.append((i, j, x))
    return out


def _perimeter(p: F) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def untangle(p: F, sign: float | None = None, tol: float = 1e-3, max_iter: int = 200) -> F:
    """The closed ring p (2D or 3D) with every self-crossing removed: at a crossing the ring splits
    into two loops; the smaller one is cut off and its points collapse onto the crossing point.
    Smaller: by area in the ring's own orientation when ``sign`` is given (2D, +1 counter-
    clockwise), else by length (3D; segments only meet where the ring runs in one plane there).
    Point count and order are kept, so the result still pairs point by point with the ring it was
    offset from. This relies on the offset being made from outward directions: a continuation
    pointing into the part makes a loop as large as the corner it is on, and cutting it collapses
    that corner (see ``segment_continuation``)."""
    p = np.array(p, dtype=float)

    def size(loop: F) -> float:
        if sign is None:
            return _perimeter(np.vstack([loop, loop[:1]]))
        return sign * signed_area(loop)

    for _ in range(max_iter):
        gid, u = _groups(p, tol)
        m = len(u)
        if m < 4:
            break
        pairs = _crossings(u, tol / 2)
        if not pairs:
            break
        pairs.sort(key=lambda c: min(c[1] - c[0], m - (c[1] - c[0])))
        used = np.zeros(m, dtype=bool)
        changed = False
        for i, j, x in pairs:
            if used[i] or used[j] or used[(i + 1) % m] or used[(j + 1) % m]:
                continue
            inner = np.vstack([x, u[i + 1 : j + 1]])
            outer = np.vstack([x, u[j + 1 :], u[: i + 1]])
            if size(inner) <= size(outer):
                p[(gid >= i + 1) & (gid <= j)] = x
                used[i : min(j + 2, m)] = True
                changed = True
            else:  # the part outside i..j is the loop to drop (rare): redo from scratch
                p[(gid > j) | (gid <= i)] = x
                changed = True
                break
        if not changed:
            break
    return p


def drop_reversed(off: F, base: F, tol: float = 1e-3, max_iter: int = 50) -> F:
    """Offset ring ``off`` with every segment that runs against its segment on ``base`` (the line
    it was offset from) collapsed to its midpoint: such a piece is swallowed by its neighbours (a
    short edge next to a concave corner) and would otherwise stick out as a spike."""
    off = np.array(off, dtype=float)
    n = len(off)
    seg_b = np.roll(base, -1, axis=0) - base
    long_b = np.linalg.norm(seg_b, axis=1) > tol
    for _ in range(max_iter):
        seg_o = np.roll(off, -1, axis=0) - off
        bad = np.nonzero(long_b & ((seg_o * seg_b).sum(axis=1) < -tol * tol))[0]
        if len(bad) == 0:
            break
        for i in bad.tolist():
            j = (i + 1) % n
            off[i] = off[j] = (off[i] + off[j]) / 2
    return off


def polyline_distance(pts: F, ring: F) -> F:
    """Distance from each point to the closed polyline ``ring`` (2D or 3D)."""
    return nearest_segment(pts, ring, np.roll(ring, -1, axis=0))[1]


def _line_meet(p0: F, p1: F, q0: F, q1: F) -> F | None:
    """Where the line p0->p1 carried on forward meets the line q0->q1 carried on backward
    (midpoint of the closest points), or None when they are parallel or meet behind."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    den = a * c - b * b
    if a < 1e-12 or c < 1e-12 or den < 1e-9 * a * c:
        return None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    if s < 1.0 - 1e-6 or t > 1e-6:
        return None
    return (p0 + s * u + q0 + t * v) / 2


def _distinct(off: F, i0: int, step: int) -> F | None:
    """The first point after (step 1) or before (step -1) point i0 that is not at the same place."""
    n = len(off)
    for s in range(1, 6):
        q: F = off[(i0 + step * s) % n]
        if float(np.linalg.norm(q - off[i0])) > 1e-6:
            return q
    return None


def _push_out(
    p: F, base: F, dist: float, tol: float, max_move: float, corner: bool = False
) -> F | None:
    """The point nearest to p that is at least ``dist`` from the closed polyline ``base``: pushed
    straight away from the one segment it is too close to, or onto where the bands of two
    segments meet (two rounded corners side by side). With ``corner`` the point is a corner of
    the envelope and is moved, in or out, onto where the bands of its two nearest segments meet.
    None when that is further than max_move or does not settle."""
    a, ab = base, np.roll(base, -1, axis=0) - base
    abab = np.maximum((ab * ab).sum(axis=1), 1e-18)
    q = np.array(p, dtype=float)
    for _ in range(16):
        t = np.clip(((q - a) * ab).sum(axis=1) / abab, 0.0, 1.0)
        away = q - (a + t[:, None] * ab)
        dd = np.linalg.norm(away, axis=1)
        order = np.argsort(dd)
        if dd[order[0]] < 1e-9:
            return None
        # the nearest segment, and a second one where it is in touch as well (with ``corner``:
        # the nearest other one); two nearly parallel directions (one feature split in pieces)
        # count as one
        g = [away[order[0]] / dd[order[0]]]
        r = [dist - dd[order[0]]]
        for j in order[1:8].tolist():
            if dd[j] < 1e-9 or (not corner and dd[j] > dist + tol):
                break
            gj = away[j] / dd[j]
            if abs(float(gj @ g[0])) < 0.999:
                g.append(gj)
                r.append(dist - dd[j])
                break
        if corner:
            if len(g) < 2:
                return None
            if max(abs(r[0]), abs(r[1])) <= tol / 2:
                break
        elif r[0] <= tol / 2:
            break
        if len(g) == 1:
            q = q + g[0] * r[0]
            continue
        gm = np.array(g)
        step, *_ = np.linalg.lstsq(gm @ gm.T, np.array(r), rcond=None)
        q = q + step @ gm
        if float(np.linalg.norm(q - p)) > max_move:
            return None
    else:
        return None
    if float(np.linalg.norm(q - p)) > max_move:
        return None
    return q


def drop_invalid(off: F, base: F, dist: float) -> F:
    """Offset ring ``off`` with the points that lie closer than ``dist`` to ``base`` (the line
    they were offset from) replaced: such a point sits inside another part's band (the end of a
    rounded corner next to a short edge, a capped mitre). A point short by no more than a fan
    chord's sag (where ``untangle`` cut two rounded corners' chords) is pushed onto the envelope
    (``_push_out``); each remaining run moves to where the offset lines on either side meet (the
    true offset corner), else onto the straight piece between its neighbours."""
    off = np.array(off, dtype=float)
    n = len(off)
    tol = max(1e-4, 1e-6 * dist)
    d = polyline_distance(off, base)
    # a point short by no more than a fan chord's sag (a crossing on the chord of a rounded corner,
    # just inside its arc) is pushed out onto the envelope, which it is next to already
    sag = dist * (1.0 - math.cos(math.radians(FAN_STEP_DEG) / 2)) + tol
    near = (d < dist - tol) & (d >= dist - sag)
    for i in np.nonzero(near)[0].tolist():
        q = _push_out(off[i], base, dist, tol, 10 * sag)  # sliding along a band, too
        if q is not None:
            off[i] = q
            d[i] = dist
    valid = d >= dist - tol
    if valid.all() or not valid.any():
        return off
    start = int(np.argmax(valid))
    k = 0
    while k < n:
        i = (start + k) % n
        if valid[i]:
            k += 1
            continue
        run = []
        while k < n and not valid[(start + k) % n]:
            run.append((start + k) % n)
            k += 1
        v0, v1 = (run[0] - 1) % n, (run[-1] + 1) % n
        before, after = _distinct(off, v0, -1), _distinct(off, v1, 1)
        x = None
        if before is not None and after is not None:
            x = _line_meet(before, off[v0], off[v1], after)
        chord = off[v1] - off[v0]
        gap = float(np.linalg.norm(chord))
        if x is not None and float(np.linalg.norm(x - off[v0])) <= 3 * (gap + dist):
            # the lines on either side are chords of rounded corners or straight: where they meet
            # is near the envelope's corner, which is where the bands of two pieces meet
            q = _push_out(x, base, dist, tol, 0.1 * dist, corner=True)
            if q is not None and float(polyline_distance(q[None], base)[0]) >= dist - tol:
                x = q
            off[run] = x
            continue
        # no corner to meet at (the lines on either side run on in line): the run goes onto the
        # straight piece between its neighbours, each point where it falls, in order
        t_prev = 0.0
        for i in run:
            t = float((off[i] - off[v0]) @ chord) / gap**2 if gap > 1e-9 else 0.5
            t_prev = min(max(t, t_prev), 1.0)
            off[i] = off[v0] + t_prev * chord
    return off


def tidy(off: F, base: F, dist: float, sign: float | None = None) -> F:
    """Offset ring without loops (``untangle``), points inside another part's band
    (``drop_invalid``) or reversed pieces (``drop_reversed``)."""
    # in 3D two pieces of the offset that cross on the carried-on surface pass each other at the
    # small height differences of a not-quite-flat trim line (STEP points ~0.01 mm apart in z on
    # a plane), so they count as crossing within a tolerance that grows with the distance
    tol = 1e-3 if sign is not None else max(1e-3, 1e-3 * dist)
    out = untangle(off, sign, tol)
    out = drop_invalid(out, base, dist)
    out = drop_reversed(out, base)
    out = untangle(out, sign, tol)
    # points the clean-up left a hair apart (a loop cut next to a collapsed run) become one: a
    # strip quad between them is then empty instead of a sliver turned an arbitrary way
    for i in range(1, len(out)):
        if float(np.linalg.norm(out[i] - out[i - 1])) < 5e-3:
            out[i] = out[i - 1]
    return out


# --- flange: exact offset of the turn line in the pull plane ----------------------------------


def _cross2(u: F, v: F) -> F:
    out: F = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return out


def _merge_close(xy: F, tol: float) -> NDArray[np.int64]:
    """Representative id per point: points closer than tol share one (grid + union-find)."""
    n = len(xy)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    cells: dict[tuple[int, int], list[int]] = {}
    keys = np.floor(xy / tol).astype(np.int64)
    for i in range(n):
        cx, cy = int(keys[i, 0]), int(keys[i, 1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), []):
                    if float(np.hypot(*(xy[i] - xy[j]))) <= tol:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
        cells.setdefault((cx, cy), []).append(i)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def outer_boundary(loop: F, tol: float = 1e-4) -> F:
    """Outer boundary (counter-clockwise, 2D) of the closed polyline ``loop``, which may cross,
    touch or run back over itself: the edge of the unbounded face of the planar arrangement of
    its segments. Segments are split where they cross or where a vertex of another one lies on
    them (collinear overlaps then share their pieces), and the walk round the outside keeps,
    at each vertex, the sharpest right turn."""
    kept = [np.asarray(loop[0], dtype=float)]
    for x in np.asarray(loop, dtype=float)[1:]:
        if float(np.hypot(*(x - kept[-1]))) > tol:
            kept.append(x)
    while len(kept) > 1 and float(np.hypot(*(kept[-1] - kept[0]))) <= tol:
        kept.pop()
    p = np.array(kept)
    q = np.roll(p, -1, axis=0)
    m = len(p)
    if m < 3:
        return p
    extra: list[F] = []  # crossing points; ids 0..m-1: starts, m..2m-1: ends, then these
    splits: list[list[tuple[float, int]]] = [[(0.0, i), (1.0, m + i)] for i in range(m)]
    d = q - p
    ln = np.hypot(d[:, 0], d[:, 1])
    lo = np.minimum(p, q) - tol
    hi = np.maximum(p, q) + tol
    chunk = 512
    for s0 in range(0, m, chunk):
        rows = np.arange(s0, min(m, s0 + chunk))
        over = (
            (lo[rows, None, 0] <= hi[None, :, 0])
            & (lo[None, :, 0] <= hi[rows, None, 0])
            & (lo[rows, None, 1] <= hi[None, :, 1])
            & (lo[None, :, 1] <= hi[rows, None, 1])
        )
        over &= np.arange(m)[None, :] > rows[:, None]  # each pair once
        ii, jj = np.nonzero(over)
        if len(ii) == 0:
            continue
        ii = rows[ii]
        r, sv = d[ii], d[jj]
        w = p[jj] - p[ii]
        den = _cross2(r, sv)
        par = np.abs(den) <= 1e-12 * ln[ii] * ln[jj]
        safe = np.where(par, 1.0, den)
        t = _cross2(w, sv) / safe
        u = _cross2(w, r) / safe
        ti, tj = tol / ln[ii], tol / ln[jj]
        proper = ~par & (t > ti) & (t < 1 - ti) & (u > tj) & (u < 1 - tj)
        for k in np.nonzero(proper)[0].tolist():
            vid = 2 * m + len(extra)
            extra.append(p[ii[k]] + t[k] * r[k])
            splits[int(ii[k])].append((float(t[k]), vid))
            splits[int(jj[k])].append((float(u[k]), vid))
        # a vertex of one segment lying on the other (T junctions, collinear overlaps)
        for seg, other in ((ii, jj), (jj, ii)):
            for end, off in ((p, 0), (q, m)):
                e = end[other]
                rel = e - p[seg]
                tt = (rel * d[seg]).sum(axis=1) / ln[seg] ** 2
                dist = np.abs(_cross2(d[seg], rel)) / ln[seg]
                tl = tol / ln[seg]
                on = (dist <= tol) & (tt > tl) & (tt < 1 - tl)
                for k in np.nonzero(on)[0].tolist():
                    splits[int(seg[k])].append((float(tt[k]), off + int(other[k])))
    allp = np.vstack([p, q, np.array(extra).reshape(-1, 2)])
    rep = _merge_close(allp, tol)
    nbrs: dict[int, set[int]] = {}
    for sp in splits:
        sp.sort()
        ids = [int(rep[v]) for _, v in sp]
        for x, y in itertools.pairwise(ids):
            if x != y:
                nbrs.setdefault(x, set()).add(y)
                nbrs.setdefault(y, set()).add(x)
    if not nbrs:
        return p
    verts = np.array(sorted(nbrs))
    start = int(verts[np.lexsort((allp[verts, 1], allp[verts, 0]))[0]])
    back = np.array([0.0, 1.0])
    v = start
    out = [allp[start]]
    first: tuple[int, int] | None = None
    for _ in range(4 * len(allp) + 8):
        cand = list(nbrs[v])
        vec = allp[cand] - allp[v]
        ang = np.arctan2(_cross2(back[None], vec), vec @ back) % (2 * math.pi)
        ang = np.where(ang <= 1e-12, 2 * math.pi, ang)
        w = cand[int(np.argmin(ang))]
        if first is None:
            first = (v, w)
        elif (v, w) == first:
            break
        out.append(allp[w])
        back = allp[v] - allp[w]
        v = w
    res = np.array(out[:-1]) if len(out) > 1 and np.allclose(out[-1], out[0]) else np.array(out)
    return res


def raw_offset(poly: F, dist: float, step_deg: float = FAN_STEP_DEG) -> F:
    """Untrimmed outward offset of the counter-clockwise polygon ``poly``: every edge moved out by
    ``dist``, an arc round each convex vertex, and at each concave vertex the two offset edges
    joined through the vertex itself. The outer boundary of this curve is the exact offset (the
    edge of the union of all discs of radius ``dist`` on the polygon)."""
    n = len(poly)
    e = np.roll(poly, -1, axis=0) - poly
    el = np.hypot(e[:, 0], e[:, 1])
    u = e / np.maximum(el, 1e-12)[:, None]
    nrm = np.c_[u[:, 1], -u[:, 0]]  # right of travel: outward for counter-clockwise
    step = math.radians(step_deg)
    out: list[F] = []
    for k in range(n):
        out.append(poly[k] + dist * nrm[k])
        out.append(poly[(k + 1) % n] + dist * nrm[k])
        j = (k + 1) % n
        cr = float(_cross2(u[k], u[j]))
        dt = float(u[k] @ u[j])
        th = math.atan2(cr, dt)
        if dt < -0.999999:  # turns back on itself (a spike): round the tip
            th = math.pi
        if th > 1e-9:  # convex: arc from this edge's normal to the next one's
            steps = max(1, math.ceil(th / step))
            for s in range(1, steps):
                out.append(poly[j] + dist * _rot2(nrm[k], th * s / steps))
        elif th < -1e-9:  # concave: through the vertex
            out.append(poly[j].copy())
    return np.array(out)


def simplify_closed(p: F, tol: float) -> F:
    """Closed polyline without points that lie within tol of the line through their neighbours
    (duplicates, collinear points and noise far below the flange's scale)."""
    p = np.asarray(p, dtype=float)
    for _ in range(64):
        n = len(p)
        if n <= 3:
            break
        a, b = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        ab = b - a
        lab = np.hypot(ab[:, 0], ab[:, 1])
        dev = np.where(
            lab > 1e-12,
            np.abs(_cross2(ab, p - a)) / np.maximum(lab, 1e-12),
            np.hypot(*(p - a).T),
        )
        # only where the point lies between its neighbours (not the tip of a spike)
        between = ((p - a) * ab).sum(axis=1) >= 0
        between &= ((b - p) * ab).sum(axis=1) >= 0
        drop = (dev <= tol) & between
        if not drop.any():
            break
        idx = np.nonzero(drop)[0]
        # never two neighbours in one pass, so each decision sees its true neighbours
        sel = [int(idx[0])]
        for i in idx[1:].tolist():
            if i != sel[-1] + 1:
                sel.append(i)
        if len(sel) > 1 and sel[-1] == n - 1 and sel[0] == 0:
            sel.pop()
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        if mask.sum() < 3:
            break
        p = p[mask]
    return p


def flange_outline(turn2: F, dist: float) -> F:
    """Flange outer edge in the pull plane (counter-clockwise): the exact ``dist`` offset of what
    the turn line covers seen along the pull. Straight where the turn line is straight, round at
    convex corners, a clean corner at concave ones; it never crosses itself, and noise on the turn
    line far smaller than ``dist`` cannot show (it lies inside the discs of its neighbours)."""
    tol = max(1e-4, 1e-7 * dist)
    base = outer_boundary(turn2, tol)
    if len(base) < 3:  # a turn line seen as a point or a line: round it off anyway
        base = turn2
    base = simplify_closed(base, 1e-3)
    if signed_area(base) < 0:
        base = base[::-1]
    edge = simplify_closed(outer_boundary(raw_offset(base, dist), tol), 0.01)
    # where two rounded corners meet (a concave corner of the edge) the crossing of their chords
    # lies up to a chord's sag inside the true arcs: such a point goes out onto the envelope
    sag = dist * (1.0 - math.cos(math.radians(FAN_STEP_DEG) / 2))
    d = polyline_distance(edge, base)
    ptol, reach = 1e-6 * dist, 4 * sag + 0.05 * dist
    for i in np.nonzero(d < dist - ptol)[0].tolist():
        # onto where the bands of its two pieces meet (both exactly at dist), else just out
        q = _push_out(edge[i], base, dist, ptol, reach, corner=True)
        if q is None or float(polyline_distance(q[None], base)[0]) < dist - ptol:
            q = _push_out(edge[i], base, dist, ptol, reach)
        if q is not None:
            edge[i] = q
    # a point moved out along the envelope can pass its neighbour by a hair: cut such loops, and
    # points left a hair apart become one
    edge = untangle(edge, 1.0, tol)
    kept = [edge[0]]
    for x in edge[1:]:
        if float(np.hypot(*(x - kept[-1]))) > 1e-3:
            kept.append(x)
    while len(kept) > 3 and float(np.hypot(*(kept[-1] - kept[0]))) <= 1e-3:
        kept.pop()
    return simplify_closed(np.array(kept), 1e-6 * dist)


BAD_TRIANGLE = 1e6  # pairing cost of a strip triangle turned the wrong way (a fold) ...
BAD_AREA = 1e4  # ... and per mm² of it (where a fold cannot be avoided, the smallest one)


def pair_rings(inner: F, outer: F, sign: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Monotone pairing of two closed 2D rings for a strip of triangles between them: (index into
    inner, index into outer) per strip point, every step advancing one ring or both. Among the
    pairings that turn every strip triangle the same way as the rings (``sign``: +1 counter-
    clockwise; then the strip cannot fold or overlap), the one with the shortest pairs (dynamic
    programming over both rings, from an anchor where the pair is certain). ``outer`` must run
    the same way round as ``inner``."""
    n, mo = len(inner), len(outer)
    i0 = int(np.argmax(inner[:, 0]))  # a point on the inner ring's hull: its pair is clear
    reach = float(np.hypot(*(outer - inner[i0]).T).min())
    j0 = int(np.argmin(np.hypot(*(outer - (inner[i0] + [reach, 0.0])).T)))
    t = np.roll(inner, -i0, axis=0)
    o = np.roll(outer, -j0, axis=0)
    t = np.vstack([t, t[:1]])
    o = np.vstack([o, o[:1]])
    scale = float(np.ptp(np.vstack([t, o]), axis=0).max()) or 1.0
    eps = 1e-9 * scale * scale

    def bad(x: F) -> F:
        out: F = np.where(sign * x > eps, BAD_TRIANGLE + BAD_AREA * np.abs(x), 0.0)
        return out

    from_left = np.zeros((n + 1, mo + 1), dtype=bool)
    diag = np.zeros((n + 1, mo + 1), dtype=bool)
    c = np.hypot(*(o - t[0]).T)
    # triangle (t_i, o_j, o_{j-1}) for j >= 1: outer ring advances
    pen_o = np.r_[0.0, bad(_cross2(o[1:] - t[0], o[:-1] - t[0]))]
    cost = np.cumsum(pen_o + c)
    for i in range(1, n + 1):
        ti, tp = t[i], t[i - 1]
        c = np.hypot(*(o - ti).T)
        tri_t = bad(_cross2(ti - tp, o - tp))  # (t_{i-1}, t_i, o_j)
        tri_d = tri_t[1:] + bad(_cross2(o[1:] - tp, o[:-1] - tp))  # + (t_{i-1}, o_j, o_{j-1})
        up = cost + tri_t
        dg = np.r_[np.inf, cost[:-1] + tri_d]
        diag[i] = dg < up
        bcost = np.minimum(up, dg) + c
        w = np.r_[0.0, bad(_cross2(o[1:] - ti, o[:-1] - ti)) + c[1:]]
        s = np.cumsum(w)
        new = s + np.minimum.accumulate(bcost - s)
        from_left[i] = new < bcost - 1e-9 * np.maximum(1.0, np.abs(bcost))
        cost = new
    ii, jj = n, mo
    path: list[tuple[int, int]] = []
    while ii > 0 or jj > 0:
        path.append((ii, jj))
        if ii == 0 or from_left[ii, jj]:
            jj -= 1
        elif diag[ii, jj]:
            ii, jj = ii - 1, jj - 1
        else:
            ii -= 1
    path.append((0, 0))
    path.reverse()
    pr = np.array(path[:-1], dtype=np.int64)  # (n, mo) is (0, 0) again
    return (pr[:, 0] + i0) % n, (pr[:, 1] + j0) % mo


def profile_rings(
    ring: F,
    tangent: F,
    out_dir: F,
    pull: F,
    to_meop: float,
    to_turn: float,
    flange: float,
) -> dict[str, F]:
    """EOP / MEOP / flange-turn / flange-edge rings, equal length, point by point (strip i runs
    between point i and i+1 of two rings), and "src": per point, the index (as float) of the
    input ring point it belongs to. The rings repeat an input point where the flange edge needs
    more points than the turn line (a rounded corner) and the other way round.

    The flange edge is not made point by point: it is the exact ``flange`` offset of the turn
    line in the plane perpendicular to the pull (``flange_outline``), and the turn line is then
    paired with it (``pair_rings``) so that no strip triangle folds back. ``out_dir`` is only the
    fallback when that offset cannot be made."""
    _, a, b = plane_frame(pull)
    # the extensions lie on the surface carried on past the edge; where an offset line runs into
    # itself in that plane (a notch narrower than two extensions, a short edge next to a concave
    # corner) the loop is cut off
    meop = tidy(ring + tangent * to_meop, ring, to_meop)
    turn = tidy(ring + tangent * (to_meop + to_turn), ring, to_meop + to_turn)
    turn2 = np.c_[turn @ a, turn @ b]
    sign = 1.0 if signed_area(np.c_[ring @ a, ring @ b]) > 0 else -1.0
    n = len(ring)
    edge2 = flange_outline(turn2, flange)
    ok = len(edge2) >= 3 and bool(np.isfinite(edge2).all())
    if ok:  # on the envelope: at the flange distance from the turn line (a chord's sag aside)
        gap = polyline_distance(edge2, turn2)
        ok = bool(gap.min() >= flange * 0.99) and bool(gap.max() <= flange * 1.01)
    if ok:
        if sign < 0:
            edge2 = edge2[::-1]
        src, oj = pair_rings(turn2, edge2, sign)
        off = edge2[oj] - turn2[src]
        edge = turn[src] + off[:, :1] * a + off[:, 1:] * b
        return {
            "eop": ring[src],
            "meop": meop[src],
            "turn": turn[src],
            "edge": edge,
            "src": src.astype(float),
        }
    raw2 = np.c_[(turn + out_dir * flange) @ a, (turn + out_dir * flange) @ b]
    edge2 = tidy(raw2, turn2, flange, sign)
    off = edge2 - turn2
    edge = turn + off[:, :1] * a + off[:, 1:] * b
    return {
        "eop": ring,
        "meop": meop,
        "turn": turn,
        "edge": edge,
        "src": np.arange(n, dtype=float),
    }
