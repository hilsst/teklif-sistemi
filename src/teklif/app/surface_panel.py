"""3D review screen: rotate/zoom, draft-angle colouring and tool-surface (face) selection.

Streamlit cannot take 3D click selection, so faces are picked from a numbered list (hover
shows the number) with helpers; a three.js component can replace this later.
The selection is the customer's part surface that lies on the tool. It drives tool area, trim
length (full boundary: outer edge, cut-outs and holes are all trimmed) and the pull direction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from numpy.typing import NDArray

from teklif.geometry.surface_select import (
    ANGULAR_DEFLECTION_DEG,
    LINEAR_DEFLECTION_MM,
    VIEW_ANGLE_DEG,
    VIEW_DEFLECTION_MM,
    FaceMesh,
    Outline,
    draft_angles_deg,
    faces_facing,
    load_face_mesh,
    suggest_pull_directions,
    triangulate_polygon,
)

GREEN, YELLOW, RED, GREY, BLUE = "#2e9e4f", "#e6b800", "#d62728", "#b0b0b0", "#4a78c2"
AXES = {
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
}


@dataclass(frozen=True)
class SurfaceSelection:
    face_ids: list[int]
    area_m2: float
    boundary_m: float
    outer_m: float
    pull: tuple[float, float, float]
    pull_label: str
    undercut_ratio: float
    ok_ratio: float

    def trim_m(self, all_openings: bool) -> float:
        """Trim length: whole boundary when every opening is trimmed, else the outer rim only."""
        return self.boundary_m if all_openings else self.outer_m

    def note(self) -> str:
        return (
            f"kalıp yüzeyi kullanıcı seçimi: {len(self.face_ids)} yüz, {self.area_m2:.3f} m², "
            f"sınır (tüm dış kenar, boşluk ve delikler trim) {self.boundary_m:.2f} m; çekme yönü "
            f"{self.pull_label} {tuple(round(x, 3) for x in self.pull)}; ters açı payı "
            f"%{self.undercut_ratio * 100:.1f}"
        )


def triangle_colors(
    mesh: FaceMesh,
    selected: list[int],
    pull: tuple[float, float, float] | None,
    min_ok: float,
    warn_min: float,
) -> list[str]:
    """One colour per triangle, in face order. Selected faces: green >= min_ok, yellow, red < 0."""
    chosen = set(selected)
    angles = draft_angles_deg(mesh, selected, np.array(pull)) if pull and selected else {}
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


_VIEWER = components.declare_component("face_viewer", path=str(Path(__file__).parent / "viewer"))
_PALETTE_IDX = {GREY: 0, BLUE: 1, GREEN: 2, YELLOW: 3, RED: 4}
FLANGE_COLOR_IDX = 5
FILL_COLOR_IDX = 6
EXT_COLOR_IDX = 7


def mold_submesh(mesh: FaceMesh, selected: list[int]) -> FaceMesh:
    """Only the selected faces, renumbered 0..n-1 (the read-only mold-surface view)."""
    chosen = set(selected)
    return FaceMesh(
        [
            dataclasses.replace(f, id=i)
            for i, f in enumerate(f for f in mesh.faces if f.id in chosen)
        ]
    )


Extra = tuple[NDArray[np.float64], list[int], list[dict[str, Any]]]


@dataclass(frozen=True)
class ToolProfile:
    """Mold profile outside the part (Kemal 2026-11-05): the surface runs ``to_meop_mm`` past
    the EOP (trim line) to the MEOP line, ``to_turn_mm`` further, then the flange turns 90°
    and runs ``flange_mm``."""

    to_meop_mm: float = 20.0
    to_turn_mm: float = 40.0  # 30-50 mm
    flange_mm: float = 150.0


def profile_rings(
    outline: Outline, pull: NDArray[np.float64], prof: ToolProfile
) -> dict[str, NDArray[np.float64]] | None:
    """EOP / MEOP / flange-turn / flange-edge rings of the tool profile."""
    if outline.ring is None or outline.tangent is None or outline.out_dir is None:
        return None
    ring, t, od = outline.ring, outline.tangent, outline.out_dir
    d = pull / np.linalg.norm(pull)
    meop = ring + t * prof.to_meop_mm
    turn = meop + t * prof.to_turn_mm
    # the flange always lies in a plane perpendicular to the pull, running outward
    f = od - (od @ d)[:, None] * d
    ln = np.linalg.norm(f, axis=1, keepdims=True)
    f = f / np.where(ln > 0, ln, 1.0)
    edge = turn + f * prof.flange_mm
    return {"eop": ring, "meop": meop, "turn": turn, "edge": edge}


def mold_extras(outline: Outline, pull: NDArray[np.float64], prof: ToolProfile) -> Extra:
    """Mold additions around the selected surface: surface extension (EOP -> MEOP -> flange turn),
    the 90° flange, fill over every hole/channel loop, and the EOP / MEOP lines."""
    tris: list[list[NDArray[np.float64]]] = []
    colors: list[int] = []
    rings = profile_rings(outline, pull, prof)

    def strip(a: NDArray[np.float64], b: NDArray[np.float64], color: int) -> None:
        n = len(a)
        for i in range(n):
            j = (i + 1) % n
            tris.extend([[a[i], a[j], b[j]], [a[i], b[j], b[i]]])
            colors.extend([color] * 2)

    if rings is not None:
        strip(rings["eop"], rings["meop"], EXT_COLOR_IDX)
        strip(rings["meop"], rings["turn"], EXT_COLOR_IDX)
        strip(rings["turn"], rings["edge"], FLANGE_COLOR_IDX)
    for k, lp in enumerate(outline.loops):
        if k == outline.outer or len(lp) < 3:
            continue
        d = pull / np.linalg.norm(pull)
        a = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        xy = np.c_[lp @ a, lp @ np.cross(d, a)]
        for i, j, m in triangulate_polygon(xy):  # the mold runs over the hole/channel
            tris.append([lp[i], lp[j], lp[m]])
            colors.append(FILL_COLOR_IDX)

    def flat(r: NDArray[np.float64]) -> list[float]:
        return list(map(float, np.round(np.vstack([r, r[:1]]), 1).ravel()))

    lines: list[dict[str, Any]] = [
        {"hex": 0xC000C0, "label": "EOP" if k == outline.outer else "", "pts": flat(lp)}
        for k, lp in enumerate(outline.loops)
    ]
    lines += [
        {"hex": 0xC000C0, "label": "", "pts": list(map(float, np.round(ch, 1).ravel()))}
        for ch in outline.open_chains
        if len(ch) > 1
    ]
    if rings is not None:
        lines += [
            {"hex": 0xE06000, "label": "MEOP", "pts": flat(rings["meop"])},
            {"hex": 0x607080, "label": "flanş dönüşü", "pts": flat(rings["turn"])},
            {"hex": 0xE06000, "label": "", "pts": flat(rings["edge"])},
        ]
    return (np.array(tris) if tris else np.zeros((0, 3, 3))), colors, lines


def viewer_payload(
    mesh: FaceMesh,
    colors: list[str],
    key: str,
    pull: tuple[float, float, float] | None = None,
    extra: Extra | None = None,
) -> dict[str, Any]:
    """Arrays for the three.js viewer: triangle soup, owning face per triangle, colour index."""
    pos = np.round(
        np.vstack([f.vertices[f.triangles].reshape(-1, 3) for f in mesh.faces if len(f.triangles)]),
        1,
    )
    tri_face = [f.id for f in mesh.faces for _ in range(len(f.triangles))]
    color_idx = [_PALETTE_IDX[c] for c in colors]
    lines: list[dict[str, Any]] = []
    if extra is not None:
        ftris, fcolors, lines = extra
        if len(ftris):
            pos = np.vstack([pos, np.round(ftris.reshape(-1, 3), 1)])
            tri_face += [-1] * len(ftris)
            color_idx += fcolors
    return {
        "lines": lines,
        "mesh_key": key,
        "pull": list(pull) if pull else None,
        "pos": pos.ravel().tolist(),
        "tri_face": tri_face,
        "tri_color": color_idx,
        "faces": [[f.id, round(f.area_mm2 / 1e6, 4)] for f in mesh.faces],
    }


def build_figure(mesh: FaceMesh, colors: list[str]) -> Any:
    """One Mesh3d trace per colour (keeps the payload small) + face-number markers for hover."""
    import plotly.graph_objects as go

    colour = np.array(colors)
    verts, tris, off = [], [], 0
    for f in mesh.faces:
        verts.append(f.vertices)
        tris.append(f.triangles + off)
        off += len(f.vertices)
    v = np.round(np.vstack(verts), 1)
    t = np.vstack(tris)
    fig = go.Figure()
    for c in dict.fromkeys(colors):
        tt = t[colour == c]
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=tt[:, 0],
                j=tt[:, 1],
                k=tt[:, 2],
                color=c,
                flatshading=True,
                hoverinfo="skip",
            )
        )
    cen = np.array([f.vertices.mean(axis=0) for f in mesh.faces])
    fig.add_trace(
        go.Scatter3d(
            x=cen[:, 0],
            y=cen[:, 1],
            z=cen[:, 2],
            mode="markers",
            marker={"size": 2, "color": "black"},
            hovertext=[f"Yüz {f.id} · {f.area_mm2 / 1e6:.3f} m²" for f in mesh.faces],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        scene={"aspectmode": "data"}, margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=520
    )
    return fig


def _load(
    data: bytes, deflection_mm: float, angle_deg: float, with_edges: bool = False
) -> FaceMesh:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "part.step"
        path.write_bytes(data)
        return load_face_mesh(path, deflection_mm, angle_deg, with_edges)


def render_surface_panel(
    data: bytes, thresholds: Any, key: str, flange_mm: float = 150.0
) -> SurfaceSelection | None:
    key = f"{key}_{hashlib.sha256(data).hexdigest()[:16]}"  # new file -> new mesh and selection
    mk = f"{key}_mesh"
    vk = mk + "v"
    if vk not in st.session_state:
        try:
            with st.spinner("3B model hazırlanıyor (büyük parçada birkaç saniye sürer)..."):
                st.session_state[vk] = _load(data, VIEW_DEFLECTION_MM, VIEW_ANGLE_DEG, True)
        except Exception as exc:
            st.warning(f"3B ekran açılamadı ({exc}); trim çevresi elle girilebilir.")
            return None
    view: FaceMesh = st.session_state[vk]
    mesh: FaceMesh = view  # replaced by the fine analysis mesh once something is selected
    sk = f"{key}_sel"
    st.session_state.setdefault(sk, [])
    pending = st.session_state.pop(f"{key}_pending", None)
    if pending is not None:  # a click in the 3D view toggles the face
        st.session_state[sk] = sorted(set(st.session_state[sk]) ^ {pending})
    min_ok = float(thresholds.draft.min_ok_deg.value)
    warn_min = float(thresholds.draft.warn_min_deg.value)

    st.markdown(
        "**Kalıp yüzeyi seçimi.** Önce müşterinin parça yüzeyinin kalıba oturan tarafını seçin; "
        "çekme yönü, ters açı, kalıp alanı ve trim çevresi bu seçimden hesaplanır."
    )
    h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
    axis = h1.selectbox("Şu yöne bakan yüzleri seç", list(AXES), key=f"{key}_axis")
    tol = h2.number_input("Tolerans [°]", 1.0, 90.0, 30.0, key=f"{key}_tol")

    def _add() -> None:
        ids = faces_facing(view, np.array(AXES[axis]), float(tol))
        st.session_state[sk] = sorted(set(st.session_state[sk]) | set(ids))

    def _all() -> None:
        st.session_state[sk] = [f.id for f in mesh.faces]

    def _clear() -> None:
        st.session_state[sk] = []

    h3.button("Ekle", on_click=_add, key=f"{key}_add")
    h4.button("Temizle", on_click=_clear, key=f"{key}_clr")
    st.button("Tüm yüzleri seç", on_click=_all, key=f"{key}_all")
    labels = {f.id: f"Yüz {f.id} · {f.area_mm2 / 1e6:.3f} m²" for f in mesh.faces}
    selected = st.multiselect(
        "Seçili yüzler",
        list(labels),
        format_func=lambda i: labels[i],
        key=sk,
    )

    pull: tuple[float, float, float] | None = None
    pull_label = ""
    cand = None
    if selected:
        if mk not in st.session_state:
            with st.spinner("Seçim için ince analiz ağı hazırlanıyor..."):
                st.session_state[mk] = _load(data, LINEAR_DEFLECTION_MM, ANGULAR_DEFLECTION_DEG)
        mesh = st.session_state[mk]
        cands = suggest_pull_directions(mesh, selected, warn_min, min_ok)
        names = [c.label for c in cands]  # stable labels so the choice survives reruns
        st.caption(
            "Öneriler (ters açı payı / yeterli çekme payı): "
            + "; ".join(
                f"{c.label} %{c.undercut_area_ratio * 100:.1f}/%{c.min_ok_ratio * 100:.0f}"
                for c in cands[:4]
            )
        )
        pick = st.selectbox(
            "Çekme yönü (en iyi öneri başta)", [*names, "Elle gir"], key=f"{key}_pull"
        )
        if pick == "Elle gir":
            x, y, z = (
                st.number_input(a, value=v, key=f"{key}_pv{a}")
                for a, v in (("X", 0.0), ("Y", 0.0), ("Z", 1.0))
            )
            vec = np.array([x, y, z], dtype=float)
            n = float(np.linalg.norm(vec))
            if n > 0:
                u = vec / n
                pull, pull_label = (float(u[0]), float(u[1]), float(u[2])), "elle"
        else:
            cand = cands[names.index(pick)]
            pull, pull_label = cand.direction, cand.label
    colors = triangle_colors(view, selected, pull, min_ok, warn_min)
    show_mold = bool(selected) and pull is not None
    left, right = st.columns([3, 2]) if show_mold else (st.container(), None)
    with left:
        click = _VIEWER(
            **viewer_payload(view, colors, key, pull), height=560, key=f"{key}_viewer", default=None
        )
    if right is not None and pull is not None:
        with right:
            sub = mold_submesh(view, selected)
            p1, p2, p3 = st.columns(3)
            prof = ToolProfile(
                p1.number_input("EOP→MEOP [mm]", 0.0, 200.0, 20.0, 5.0, key=f"{key}_pm"),
                p2.number_input(
                    "MEOP→flanş dönüşü [mm]",
                    0.0,
                    200.0,
                    40.0,
                    5.0,
                    key=f"{key}_pt",
                    help="Genelde 30-50 mm",
                ),
                p3.number_input("Flanş [mm]", 0.0, 500.0, flange_mm, 10.0, key=f"{key}_pf"),
            )
            outline = view.outline(selected, np.array(pull))
            flange = mold_extras(outline, np.array(pull), prof) if outline else None
            mold_key = (
                f"{key}_mold_{hash((tuple(selected), tuple(round(x, 3) for x in pull), prof))}"
            )
            d = np.array(pull)
            side = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side)
            tilt = d + 0.7 * side + (0.0 if abs(d[2]) > 0.9 else 0.25 * np.array([0, 0, 1.0]))
            tilt /= np.linalg.norm(tilt)
            _VIEWER(
                **viewer_payload(
                    sub,
                    triangle_colors(sub, list(range(len(sub.faces))), pull, min_ok, warn_min),
                    mold_key,
                    pull,
                    flange,
                ),
                view_dir=tilt.tolist(),
                readonly=True,
                height=560,
                key=f"{key}_moldviewer",
                default=None,
            )
            st.caption(
                "**Kalıp profili** (EOP'den dışa): yüzey EOP'nin "
                f"{prof.to_meop_mm:.0f} mm ötesine uzanır (**MEOP**), {prof.to_turn_mm:.0f} mm "
                "daha gider, sonra flanş çekme yönüne dik düzlemde "
                f"{prof.flange_mm:.0f} mm gider. Mor: "
                "**EOP** (dış kenar, delikler, kanallar; açık mavi: kalıbın delik/kanal üstünden "
                "geçtiği alan), açık yeşil: yüzey uzantısı, turuncu: flanş."
            )
    last = f"{key}_lastclick"
    ident = (click.get("n"), click.get("t")) if click else None
    if click and ident != st.session_state.get(last):  # n restarts when the iframe is rebuilt
        st.session_state[last] = ident
        # the multiselect already exists in this run: apply the toggle at the top of the next one
        st.session_state[f"{key}_pending"] = int(click["face"])
        st.rerun()
    if st.checkbox("Eski görünüm (Plotly)", False, key=f"{key}_plotly"):
        st.plotly_chart(build_figure(view, colors), width="stretch", key=f"{key}_fig")
    st.caption(
        f"Renkler (seçili yüzler): yeşil ≥ {min_ok:g}° çekme açısı, sarı {warn_min:g}–{min_ok:g}°, "
        "kırmızı ters açı; mavi: çekme yönü henüz yok; gri: seçili değil. "
        "Yüze tıklayarak seçip bırakabilirsiniz. Eksenler: X kırmızı, Y yeşil, Z mavi (parçanın "
        "sol-alt köşesinde); mor ok seçili çekme yönüdür."
    )
    if not selected or pull is None:
        return None
    from teklif.geometry.surface_select import _score

    bad, ok = _score(mesh, selected, np.array(pull), warn_min, min_ok)
    boundary_mm, outer_mm = mesh.trim_lengths_mm(selected, np.array(pull))
    return SurfaceSelection(
        face_ids=list(selected),
        area_m2=mesh.area_m2(selected),
        boundary_m=boundary_mm / 1000,
        outer_m=outer_mm / 1000,
        pull=pull,
        pull_label=pull_label,
        undercut_ratio=bad,
        ok_ratio=ok,
    )
