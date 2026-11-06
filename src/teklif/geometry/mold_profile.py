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
- the continuation is taken per rim *segment* from the mesh triangle that owns that edge,
- at a vertex, nearly equal directions are averaged; a convex change (the extensions open a gap)
  becomes a fan of copies of the vertex (a rounded corner), a concave change (they overlap)
  becomes one mitred point, so both offset lines keep their exact distance,
- the flange direction is the outward normal of the EOP seen along the pull (pieces running along
  the pull are merged into one plan point), rounded the same way at convex plan corners,
- the flange's outer edge is untangled in the pull plane: where it crosses itself (a cut-out
  tighter than the flange, a notch narrower than two flanges) the loop is cut off and its points
  collapse to the crossing, so the flange never folds back over the cut-out.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

F = NDArray[np.float64]

SMOOTH_DEG = 10.0  # continuation changes up to this are averaged into one point
FAN_STEP_DEG = 10.0  # angular step of a rounded (fan) corner
PLAN_SMOOTH_DEG = 12.0  # plan corners up to this get a plain mitre for the flange direction
MITRE_CAP = 3.0  # longest mitre (x the distance) at a concave corner


def plane_frame(pull: F) -> tuple[F, F, F]:
    """(unit pull d, a, b): a and b span the plane perpendicular to the pull."""
    d = np.asarray(pull, dtype=float)
    d = d / np.linalg.norm(d)
    a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return d, a, np.cross(d, a)


def signed_area(xy: F) -> float:
    return float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))


def decimate_loop(loop: F, max_pts: int = 300) -> F:
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
    cos5 = math.cos(math.radians(5.0))
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
        idx[s : s + step] = cand[rows, j]
        dist[s : s + step] = np.sqrt(dd[rows, j])
    return idx, dist


def segment_continuation(ring: F, edges: tuple[F, F, F, F]) -> F:
    """Per ring segment (point k to k+1): the unit direction in which the surface goes on past
    it, in the plane of the triangle that owns the edge, perpendicular to the edge, away from the
    triangle (a wall parallel to the pull continues along the pull, a flat sheet outward)."""
    ea, eb, en, eo = edges
    nxt = np.roll(ring, -1, axis=0)
    mid = (ring + nxt) / 2
    seg = nxt - ring
    sl = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = seg / np.where(sl > 0, sl, 1.0)
    out = np.zeros_like(ring)
    if len(ea) == 0:
        return out
    idx, _ = nearest_segment(mid, ea, eb)
    c = np.cross(seg, en[idx])
    cl = np.linalg.norm(c, axis=1)
    inward = eo[idx] - mid  # towards the owning triangle's far vertex: inside the surface
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
    offset from."""
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


def drop_invalid(off: F, base: F, dist: float) -> F:
    """Offset ring ``off`` with the points that lie closer than ``dist`` to ``base`` (the line
    they were offset from) replaced: such a point sits inside another part's band (the end of a
    rounded corner next to a short edge, a capped mitre). Each run of them moves to where the
    offset lines on either side meet (the true offset corner), else onto the straight piece
    between its neighbours."""
    off = np.array(off, dtype=float)
    n = len(off)
    valid = polyline_distance(off, base) >= dist - max(1e-4, 1e-6 * dist)
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
    out = untangle(off, sign)
    out = drop_invalid(out, base, dist)
    out = drop_reversed(out, base)
    return untangle(out, sign)


def profile_rings(
    ring: F,
    tangent: F,
    out_dir: F,
    pull: F,
    to_meop: float,
    to_turn: float,
    flange: float,
) -> dict[str, F]:
    """EOP / MEOP / flange-turn / flange-edge rings (equal length, point by point)."""
    _, a, b = plane_frame(pull)
    # the extensions lie on the surface carried on past the edge; where an offset line runs into
    # itself in that plane (a notch narrower than two extensions, a short edge next to a concave
    # corner) the loop is cut off, as for the flange
    meop = tidy(ring + tangent * to_meop, ring, to_meop)
    turn = tidy(ring + tangent * (to_meop + to_turn), ring, to_meop + to_turn)
    raw = turn + out_dir * flange
    turn2 = np.c_[turn @ a, turn @ b]
    raw2 = np.c_[raw @ a, raw @ b]
    sign = 1.0 if signed_area(np.c_[ring @ a, ring @ b]) > 0 else -1.0
    edge2 = tidy(raw2, turn2, flange, sign)
    off = edge2 - turn2
    edge = turn + off[:, :1] * a + off[:, 1:] * b
    return {"eop": ring, "meop": meop, "turn": turn, "edge": edge}
