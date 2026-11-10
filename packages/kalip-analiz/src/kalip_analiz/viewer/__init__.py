"""3D viewer (three.js): draft map, mold pieces, parting walls and numbered, coloured cores.

The browser side is ``static/index.html`` (+ ``three.min.js``, MIT). It runs in two hosts:

* **Streamlit**: ``streamlit_component()`` declares it as a component; call it with
  ``**viewer_payload(...)``. Clicks come back as ``{"toggles": [face ids], "n", "t"}`` (left click,
  after the pause ``viewer.select_pause_s``) or ``{"hide": face id}`` (right click).
* **Standalone page**: ``export_html(payload, path, side_html)`` writes one self-contained HTML
  file (three.js and the data inlined) that any browser opens; clicking faces lists their ids.

The Python helpers here turn an analysis (``kalip_analiz.pull_search``) into that payload:
per-triangle colour index into the viewer ``PALETTE``, overlay triangles (parting walls) and
polylines (pull arrows, core pins with labels M1, M2..., unresolved regions Ç1...).
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kalip_analiz import pull_search as ps
from kalip_analiz.mesh import FaceMesh, draft_angles_deg
from kalip_analiz.settings import ViewerParams

STATIC_DIR = Path(__file__).with_name("static")

GREEN, YELLOW, RED, GREY, BLUE = "#2e9e4f", "#e6b800", "#d62728", "#b0b0b0", "#4a78c2"
PALETTE_IDX = {GREY: 0, BLUE: 1, GREEN: 2, YELLOW: 3, RED: 4}
# colours of the analysis view (indices into the viewer's PALETTE): one per mold piece, one for the
# parting walls, one per core
PIECE_COLOR_IDX0 = 8
PIECE_HEX = ("#3b82f6", "#f97316", "#a855f7", "#14b8a6", "#ca8a04", "#ec4899")
WALL_COLOR_IDX = 15
WALL_HEX = "#cbd5e1"
CORE_COLOR_IDX0 = 16
CORE_NAMES = ("pembe", "turkuaz", "yeşil", "sarı", "gül kırmızısı", "gök mavisi")
CORE_HEXES = ("#ff2d95", "#22d3ee", "#84cc16", "#facc15", "#e11d48", "#0ea5e9")
RED_IDX = 4  # a region nothing releases (unresolved)
UNRESOLVED_HEX = 0xD62728

Extra = tuple[NDArray[np.float64], list[int], list[dict[str, Any]]]


# --- colours -----------------------------------------------------------------------------


def triangle_colors(
    mesh: FaceMesh,
    selected: Sequence[int],
    pull: Sequence[float] | None,
    min_ok: float,
    warn_min: float,
) -> list[str]:
    """Draft map, one colour per triangle in face order. Selected faces: green >= min_ok, yellow
    between warn_min and min_ok, red below (undercut); blue: no pull yet; grey: not selected."""
    chosen = set(selected)
    ids = list(selected)
    angles = draft_angles_deg(mesh, ids, np.array(pull)) if pull is not None and ids else {}
    out: list[str] = []
    for f in mesh.faces:
        if f.id not in chosen:
            out += [GREY] * len(f.triangles)
        elif f.id in angles:
            a = angles[f.id]
            out += [GREEN if x >= min_ok else (YELLOW if x >= warn_min else RED) for x in a]
        else:
            out += [BLUE] * len(f.triangles)
    return out


def _nearest(src: NDArray[np.float64], dst: NDArray[np.float64]) -> NDArray[np.int64]:
    """Index of the nearest ``dst`` point for every ``src`` point (chunked brute force)."""
    out = np.empty(len(src), dtype=np.int64)
    for lo in range(0, len(src), 256):
        d = ((src[lo : lo + 256, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        out[lo : lo + 256] = d.argmin(axis=1)
    return out


def piece_colors(
    view: FaceMesh, selected: Sequence[int], ts: ps.TriSet, method: ps.Method
) -> list[int]:
    """Palette index per view triangle: each mold piece its own colour, every core its own colour,
    regions nothing releases red. The view mesh may be coarser than the analysis mesh, so every
    view triangle takes the label of the nearest analysis triangle on the same face."""
    label = np.asarray(PIECE_COLOR_IDX0 + method.piece_of % len(PIECE_HEX), dtype=np.int64)
    for tag, _, r in core_regions(method):
        if tag.startswith("Ç"):
            label[r.tris] = RED_IDX
        else:
            label[r.tris] = CORE_COLOR_IDX0 + (int(tag[1:]) - 1) % len(CORE_HEXES)
    order = np.argsort(ts.face, kind="stable")
    faces_sorted = ts.face[order]
    chosen = set(selected)
    out: list[int] = []
    for f in view.faces:
        n = len(f.triangles)
        if f.id not in chosen or not n:
            out += [PALETTE_IDX[GREY]] * n
            continue
        lo, hi = np.searchsorted(faces_sorted, [f.id, f.id + 1])
        idx = order[lo:hi]
        if not len(idx):
            out += [PALETTE_IDX[BLUE]] * n
            continue
        near = _nearest(f.vertices[f.triangles].mean(axis=1), ts.centroids[idx])
        out += label[idx[near]].tolist()
    return out


# --- cores ---------------------------------------------------------------------------------


def core_hex(tag: str) -> int:
    """Colour of core ``M<n>`` as an int (the same colour paints its triangles)."""
    return int(CORE_HEXES[(int(tag[1:]) - 1) % len(CORE_HEXES)][1:], 16)


def core_regions(method: ps.Method) -> list[tuple[str, int, ps.UndercutRegion]]:
    """Every core (M1, M2, ...) and unresolved region (Ç1, ...) of a method with its piece number
    (1-based), in the order the markers and the table use."""
    out: list[tuple[str, int, ps.UndercutRegion]] = []
    for k, p in enumerate(method.pieces, 1):
        for r in p.cores:
            out.append((f"M{len(out) + 1}", k, r))
    n_core = len(out)
    for k, p in enumerate(method.pieces, 1):
        for r in p.unresolved:
            out.append((f"Ç{len(out) - n_core + 1}", k, r))
    return out


def core_markers(
    method: ps.Method, diag: float, params: ViewerParams | None = None
) -> list[dict[str, Any]]:
    """A numbered pin per core: a cross at the region's centre and a leader line out of the
    surface that ends in the label (sizes from ``viewer.core_marker_ratio`` /
    ``viewer.core_leader_ratio`` of the part diagonal)."""
    vp = params or ViewerParams()
    lines: list[dict[str, Any]] = []
    arm = vp.core_marker_ratio * diag
    for tag, _, r in core_regions(method):
        c = np.array(r.center, dtype=float)
        n = np.array(r.outward, dtype=float)
        n = n / (float(np.linalg.norm(n)) or 1.0)
        hex_ = UNRESOLVED_HEX if tag.startswith("Ç") else core_hex(tag)
        tip = c + n * vp.core_leader_ratio * diag
        for d in np.eye(3):
            lines.append(
                {
                    "hex": hex_,
                    "label": "",
                    "pts": np.concatenate([c - d * arm, c + d * arm]).round(1).tolist(),
                }
            )
        lines.append({"hex": hex_, "label": tag, "pts": np.concatenate([c, tip]).round(1).tolist()})
    return lines


def core_rows(method: ps.Method) -> list[dict[str, Any]]:
    """Table of the cores: where they are and how big (the numbers the markers carry)."""
    rows = []
    for tag, k, r in core_regions(method):
        rows.append(
            {
                "no": tag,
                "renk": "kırmızı"
                if tag.startswith("Ç")
                else CORE_NAMES[(int(tag[1:]) - 1) % len(CORE_NAMES)],
                "kalıp parçası": k,
                "alan [mm²]": round(r.area_mm2),
                "yüzey payı [%]": round(r.area_ratio * 100, 3),
                "en derin açı [°]": round(r.min_draft_deg, 1),
                "konum X/Y/Z [mm]": " / ".join(f"{x:.0f}" for x in r.center),
                "yüz no": ", ".join(map(str, r.faces[:6])) + ("…" if len(r.faces) > 6 else ""),
            }
        )
    return rows


def piece_legend(method: ps.Method) -> str:
    """HTML legend: a colour swatch per mold piece, per core and for the parting wall."""
    sw = "<span style='color:{c};font-size:1.2em'>■</span> {t}"
    items = [
        sw.format(c=PIECE_HEX[k % len(PIECE_HEX)], t=f"kalıp parça {k + 1}")
        for k in range(method.n_pieces)
    ]
    if method.n_cores:
        items += [
            sw.format(c=CORE_HEXES[i % len(CORE_HEXES)], t=f"M{i + 1}")
            for i in range(method.n_cores)
        ]
        items.append("M1, M2…: maçalar (kendi renginde yüzey + artı ve çizgi işareti)")
    if method.n_unresolved:
        items.append(sw.format(c=RED, t="çözülemeyen ters açı"))
    if method.n_pieces > 1:
        items.append(sw.format(c=WALL_HEX, t="ayırma yüzeyi (yüzeye dik)"))
    return " &nbsp; ".join(items)


# --- overlays ------------------------------------------------------------------------------


def analysis_extra(ts: ps.TriSet, method: ps.Method, params: ViewerParams | None = None) -> Extra:
    """Parting walls (strips perpendicular to the tool surface along every cut between two
    pieces), for a split tool one pull arrow per piece in the piece's colour, and the core pins."""
    vp = params or ViewerParams()
    diag = float(np.linalg.norm(np.ptp(ts.centroids, axis=0))) or 1.0
    walls = ps.parting_walls(ts, method.piece_of, vp.parting_wall_ratio * diag)
    lines: list[dict[str, Any]] = []
    if method.n_pieces > 1:
        for k, p in enumerate(method.pieces):
            d = np.asarray(p.pull.direction, dtype=float)
            c = np.average(ts.centroids[p.tris], axis=0, weights=ts.areas[p.tris])
            ln = vp.piece_arrow_ratio * diag
            a, _ = ps._basis(d)
            tip, tail = c, c - d * ln
            hex_ = int(PIECE_HEX[k % len(PIECE_HEX)][1:], 16)
            back = tip - d * 0.15 * ln
            segs = [(tail, tip), (tip, back + a * 0.07 * ln), (tip, back - a * 0.07 * ln)]
            for p0, p1 in segs:
                lines.append(
                    {"hex": hex_, "label": "", "pts": np.concatenate([p0, p1]).round(1).tolist()}
                )
    lines += core_markers(method, diag, vp)
    return walls, [WALL_COLOR_IDX] * len(walls), lines


# --- payload -------------------------------------------------------------------------------


def viewer_payload(
    mesh: FaceMesh,
    colors: Sequence[str | int],
    key: str,
    pull: Sequence[float] | None = None,
    extra: Extra | None = None,
    hidden: frozenset[int] = frozenset(),
    selected: Sequence[int] = (),
    ack: int = 0,
    sync_ms: int | None = None,
    readonly: bool = False,
) -> dict[str, Any]:
    """Arrays for the three.js viewer: triangle soup, owning face per triangle, colour index.

    ``colors``: one per triangle, a hex from this module or a palette index. ``hidden``: solids
    of an assembly left out of the picture. ``sync_ms``: selection pause (default
    ``viewer.select_pause_s``)."""
    if sync_ms is None:
        sync_ms = round(ViewerParams().select_pause_s * 1000)
    pos = np.round(
        np.vstack([f.vertices[f.triangles].reshape(-1, 3) for f in mesh.faces if len(f.triangles)]),
        1,
    )
    tri_face = [f.id for f in mesh.faces for _ in range(len(f.triangles))]
    color_idx = [c if isinstance(c, int) else PALETTE_IDX[c] for c in colors]
    if hidden:
        keep = np.array(
            [f.solid not in hidden for f in mesh.faces for _ in range(len(f.triangles))]
        )
        pos = pos.reshape(-1, 9)[keep].reshape(-1, 3)
        tri_face = [t for t, k in zip(tri_face, keep, strict=True) if k]
        color_idx = [c for c, k in zip(color_idx, keep, strict=True) if k]
    lines: list[dict[str, Any]] = []
    extra_sig = 0
    if extra is not None:
        ftris, fcolors, lines = extra
        extra_sig = len(ftris) * 1000 + int(abs(float(np.round(ftris.sum(), 1))) * 10) % 1000
        if len(ftris):
            pos = np.vstack([pos, np.round(ftris.reshape(-1, 3), 1)])
            tri_face += [-1] * len(ftris)
            color_idx += fcolors
    return {
        "lines": lines,
        "mesh_key": key,
        "selected": [int(i) for i in selected],
        "ack": int(ack),
        "sync_ms": int(sync_ms),
        "pull": [float(x) for x in pull] if pull is not None else None,
        "pos": pos.ravel().tolist(),
        "tri_face": tri_face,
        "tri_color": color_idx,
        "faces": [[f.id, round(f.area_mm2 / 1e6, 4)] for f in mesh.faces if f.solid not in hidden],
        "geom_rev": [*sorted(hidden), *([extra_sig] if extra_sig else [])],
        "readonly": readonly,
    }


# --- hosts ---------------------------------------------------------------------------------


def streamlit_component(name: str = "kalip_viewer") -> Any:
    """The viewer as a Streamlit component (needs ``streamlit``). Call the result with
    ``**viewer_payload(...), height=560, key=..., default=None``."""
    import streamlit.components.v1 as components

    return components.declare_component(name, path=str(STATIC_DIR))


def _script_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def export_html(
    payload: dict[str, Any], path: str | Path, side_html: str = "", title: str = "Kalıp analizi"
) -> Path:
    """Write a self-contained HTML page (three.js and the data inlined) showing ``payload``;
    ``side_html`` goes into a panel on the right (decision, legend, core table)."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    three = (STATIC_DIR / "three.min.js").read_text(encoding="utf-8")
    data = (
        f"<title>{html.escape(title)}</title>"
        f"<script>window.KALIP_VIEWER_DATA={_script_json(payload)};"
        f"window.KALIP_SIDE_HTML={_script_json(side_html)};</script>"
        f"<script>{three}</script>"
    )
    page = page.replace('<script src="three.min.js"></script>', data, 1)
    out = Path(path)
    out.write_text(page, encoding="utf-8")
    return out


def rows_html(rows: Sequence[dict[str, Any]]) -> str:
    """A small HTML table (for ``side_html``)."""
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(str(k))}</th>" for k in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"
