"""3D review screen: rotate/zoom, draft-angle colouring and tool-surface (face) selection.

Streamlit cannot take 3D click selection, so faces are picked from a numbered list (hover
shows the number) with helpers; a three.js component can replace this later.
The selection is the customer's part surface that lies on the tool. It drives tool area, trim
length (full boundary: outer edge, cut-outs and holes are all trimmed) and the pull direction.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from teklif.geometry.surface_select import (
    ANGULAR_DEFLECTION_DEG,
    LINEAR_DEFLECTION_MM,
    VIEW_ANGLE_DEG,
    VIEW_DEFLECTION_MM,
    FaceMesh,
    draft_angles_deg,
    faces_facing,
    load_face_mesh,
    suggest_pull_directions,
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


def viewer_payload(
    mesh: FaceMesh, colors: list[str], key: str, pull: tuple[float, float, float] | None = None
) -> dict[str, Any]:
    """Arrays for the three.js viewer: triangle soup, owning face per triangle, colour index."""
    pos = np.round(
        np.vstack([f.vertices[f.triangles].reshape(-1, 3) for f in mesh.faces if len(f.triangles)]),
        1,
    )
    tri_face = [f.id for f in mesh.faces for _ in range(len(f.triangles))]
    return {
        "mesh_key": key,
        "pull": list(pull) if pull else None,
        "pos": pos.ravel().tolist(),
        "tri_face": tri_face,
        "tri_color": [_PALETTE_IDX[c] for c in colors],
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


def _load(data: bytes, deflection_mm: float, angle_deg: float) -> FaceMesh:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "part.step"
        path.write_bytes(data)
        return load_face_mesh(path, deflection_mm, angle_deg)


def render_surface_panel(data: bytes, thresholds: Any, key: str) -> SurfaceSelection | None:
    key = f"{key}_{hashlib.sha256(data).hexdigest()[:16]}"  # new file -> new mesh and selection
    mk = f"{key}_mesh"
    vk = mk + "v"
    if vk not in st.session_state:
        try:
            with st.spinner("3B model hazırlanıyor (büyük parçada birkaç saniye sürer)..."):
                st.session_state[vk] = _load(data, VIEW_DEFLECTION_MM, VIEW_ANGLE_DEG)
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
    click = _VIEWER(
        **viewer_payload(view, colors, key, pull), height=560, key=f"{key}_viewer", default=None
    )
    last = f"{key}_lastclick"
    if click and click.get("n", 0) != st.session_state.get(last):
        st.session_state[last] = click["n"]
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
    return SurfaceSelection(
        face_ids=list(selected),
        area_m2=mesh.area_m2(selected),
        boundary_m=mesh.boundary_length_mm(selected) / 1000,
        outer_m=mesh.outer_perimeter_mm(selected, np.array(pull)) / 1000,
        pull=pull,
        pull_label=pull_label,
        undercut_ratio=bad,
        ok_ratio=ok,
    )
