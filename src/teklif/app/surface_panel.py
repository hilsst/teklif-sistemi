"""3D review screen: rotate/zoom, draft-angle colouring and tool-surface (face) selection.

Streamlit cannot take 3D click selection, so faces are picked from a numbered list (hover
shows the number) with helpers; a three.js component can replace this later.
The selection is the customer's part surface that lies on the tool. It drives tool area, trim
length (full boundary: outer edge, cut-outs and holes are all trimmed) and the pull direction.

Kemal (2026-11-09): the side screen no longer builds a mold (surface extension and flange). It
shows the best pull direction for the selected tool surface and the tooling decision (single
piece, core, split, split + core) from ``teklif.draft.pull_search``.
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

from teklif.draft import pull_search as ps
from teklif.geometry import mold_profile
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
    extension_mm: float = 60.0  # EOP -> flange turn along the surface (EOP->MEOP + MEOP->turn)
    flange_mm: float = 150.0
    solids: tuple[int, ...] = ()  # solids of an assembly that the selected faces belong to
    assessment: ps.ToolingAssessment | None = None  # tooling decision for this pull

    def trim_m(self, all_openings: bool) -> float:
        """Trim length: whole boundary when every opening is trimmed, else the outer rim only."""
        return self.boundary_m if all_openings else self.outer_m

    def note(self) -> str:
        return (
            f"kalıp yüzeyi kullanıcı seçimi: {len(self.face_ids)} yüz, {self.area_m2:.3f} m², "
            f"sınır (tüm dış kenar, boşluk ve delikler trim) {self.boundary_m:.2f} m; çekme yönü "
            f"{self.pull_label} {tuple(round(x, 3) for x in self.pull)}; ters açı payı "
            f"%{self.undercut_ratio * 100:.1f}"
            + (f"; sistem kalıp kararı: {self.assessment.label}" if self.assessment else "")
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
    """EOP / MEOP / flange-turn / flange-edge rings of the tool profile (equal length)."""
    if outline.ring is None or outline.tangent is None or outline.out_dir is None:
        return None
    return mold_profile.profile_rings(
        outline.ring,
        outline.tangent,
        outline.out_dir,
        np.asarray(pull, dtype=float),
        prof.to_meop_mm,
        prof.to_turn_mm,
        prof.flange_mm,
    )


def viewer_payload(
    mesh: FaceMesh,
    colors: list[str],
    key: str,
    pull: tuple[float, float, float] | None = None,
    extra: Extra | None = None,
    hidden: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Arrays for the three.js viewer: triangle soup, owning face per triangle, colour index.
    ``hidden``: solids of an assembly left out of the picture (right click in the viewer)."""
    pos = np.round(
        np.vstack([f.vertices[f.triangles].reshape(-1, 3) for f in mesh.faces if len(f.triangles)]),
        1,
    )
    tri_face = [f.id for f in mesh.faces for _ in range(len(f.triangles))]
    color_idx = [_PALETTE_IDX[c] for c in colors]
    if hidden:
        keep = np.array(
            [f.solid not in hidden for f in mesh.faces for _ in range(len(f.triangles))]
        )
        pos = pos.reshape(-1, 9)[keep].reshape(-1, 3)
        tri_face = [t for t, k in zip(tri_face, keep, strict=True) if k]
        color_idx = [c for c, k in zip(color_idx, keep, strict=True) if k]
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
        "faces": [[f.id, round(f.area_mm2 / 1e6, 4)] for f in mesh.faces if f.solid not in hidden],
        "geom_rev": sorted(hidden),
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
    data: bytes,
    thresholds: Any,
    key: str,
    flange_mm: float = 150.0,
    only_solids: frozenset[int] | None = None,
) -> SurfaceSelection | None:
    """``only_solids``: for an assembly, show and select only these solids (the chosen part)."""
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
    hk = f"{key}_hidden" + ("_" + ",".join(map(str, sorted(only_solids))) if only_solids else "")
    base_hidden: frozenset[int] = (
        frozenset(f.solid for f in view.faces) - only_solids if only_solids else frozenset()
    )
    hide_face = st.session_state.pop(f"{key}_hide_pending", None)
    if hide_face is not None and 0 <= hide_face < len(view.faces):
        # right click on an element of an assembly hides its whole solid (never the last one)
        solids = {f.solid for f in view.faces}
        gone = set(st.session_state.get(hk, ())) | {view.faces[hide_face].solid}
        if solids - gone - set(base_hidden):
            st.session_state[hk] = sorted(gone)
    user_hidden = frozenset(st.session_state.get(hk, ()))
    hidden = user_hidden | base_hidden
    visible_ids = {f.id for f in view.faces if f.solid not in hidden}
    sk = f"{key}_sel"
    st.session_state.setdefault(sk, [])
    st.session_state[sk] = [i for i in st.session_state[sk] if i in visible_ids]
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
        st.session_state[sk] = sorted(set(st.session_state[sk]) | (set(ids) & visible_ids))

    def _all() -> None:
        st.session_state[sk] = sorted(visible_ids)

    def _clear() -> None:
        st.session_state[sk] = []

    h3.button("Ekle", on_click=_add, key=f"{key}_add")
    h4.button("Temizle", on_click=_clear, key=f"{key}_clr")
    st.button("Tüm yüzleri seç", on_click=_all, key=f"{key}_all")
    labels = {
        f.id: f"Yüz {f.id} · {f.area_mm2 / 1e6:.3f} m²" for f in mesh.faces if f.id in visible_ids
    }
    selected = st.multiselect(
        "Seçili yüzler",
        list(labels),
        format_func=lambda i: labels[i],
        key=sk,
    )

    pull: tuple[float, float, float] | None = None
    pull_label = ""
    settings = ps.settings_from_thresholds(thresholds)
    tris: ps.TriSet | None = None
    if selected:
        if mk not in st.session_state:
            with st.spinner("Seçim için ince analiz ağı hazırlanıyor..."):
                st.session_state[mk] = _load(data, LINEAR_DEFLECTION_MM, ANGULAR_DEFLECTION_DEG)
        mesh = st.session_state[mk]
        tris = ps.triset_from_mesh(mesh, selected)
        sk_search = f"{key}_ps_{hash((tuple(selected), settings))}"
        if sk_search not in st.session_state:
            with st.spinner("En uygun çekme yönü aranıyor..."):
                st.session_state[sk_search] = ps.search_directions(
                    tris.normals, tris.areas, settings
                )
        found: list[ps.DirectionScore] = st.session_state[sk_search]
        options: dict[str, tuple[float, float, float]] = {}
        for k, f in enumerate(found):
            options[f"en uygun #{k + 1} (optimize)" if k else "en uygun (optimize)"] = f.direction
        for c in suggest_pull_directions(mesh, selected, warn_min, min_ok):
            options.setdefault(c.label, c.direction)
        st.caption(
            "Optimize edilen yönler (ters açı payı / yeterli çekme payı / en küçük açı): "
            + "; ".join(
                f"#{k + 1} %{f.undercut_ratio * 100:.1f} / %{f.ok_ratio * 100:.0f} / "
                f"{f.min_draft_deg:.1f}°"
                for k, f in enumerate(found)
            )
        )
        pick = st.selectbox(
            "Çekme yönü (en uygun başta)", [*options, "Elle gir"], key=f"{key}_pull"
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
            pull, pull_label = options[pick], pick
    colors = triangle_colors(view, selected, pull, min_ok, warn_min)
    show_side = bool(selected) and pull is not None
    left, right = st.columns(2) if show_side else (st.container(), None)
    with left:
        click = _VIEWER(
            **viewer_payload(view, colors, key, pull, hidden=hidden),
            height=560,
            key=f"{key}_viewer",
            default=None,
        )
    assessment: ps.ToolingAssessment | None = None
    if right is not None and pull is not None and tris is not None:
        ak = f"{key}_assess_{hash((tuple(selected), tuple(round(x, 6) for x in pull), settings))}"
        if ak not in st.session_state:
            with st.spinner("Ters açı bölgeleri ve kalıp kararı hesaplanıyor..."):
                st.session_state[ak] = ps.assess(tris, np.array(pull), settings)
        assessment = st.session_state[ak]
        with right:
            render_assessment(assessment, pull_label)
    last = f"{key}_lastclick"
    ident = (click.get("n"), click.get("t")) if click else None
    if click and ident != st.session_state.get(last):  # n restarts when the iframe is rebuilt
        st.session_state[last] = ident
        # the multiselect already exists in this run: apply the toggle at the top of the next one
        if "hide" in click:
            st.session_state[f"{key}_hide_pending"] = int(click["hide"])
        else:
            st.session_state[f"{key}_pending"] = int(click["face"])
        st.rerun()
    if user_hidden and st.button(
        f"Gizlenen {len(user_hidden)} elemanı geri getir", key=f"{key}_unhide"
    ):
        st.session_state[hk] = []
        st.rerun()
    if st.checkbox("Eski görünüm (Plotly)", False, key=f"{key}_plotly"):
        st.plotly_chart(build_figure(view, colors), width="stretch", key=f"{key}_fig")
    st.caption(
        f"Renkler (seçili yüzler): yeşil ≥ {min_ok:g}° çekme açısı, sarı {warn_min:g}–{min_ok:g}°, "
        "kırmızı ters açı; mavi: çekme yönü henüz yok; gri: seçili değil. "
        "Yüze tıklayarak seçip bırakabilirsiniz; montajda sağ tıkla o elemanı gizlersiniz. "
        "Eksenler: X kırmızı, Y yeşil, Z mavi (parçanın sol-alt köşesinde); mor ok seçili çekme "
        "yönüdür."
    )
    if not selected or pull is None:
        return None
    from teklif.geometry.surface_select import _score

    bad, ok = _score(mesh, selected, np.array(pull), warn_min, min_ok)
    boundary_mm, outer_mm = mesh.trim_lengths_mm(selected, np.array(pull))
    prof = ToolProfile(flange_mm=flange_mm)  # tool reach for the cost estimate only (no mold view)
    return SurfaceSelection(
        face_ids=list(selected),
        area_m2=mesh.area_m2(selected),
        boundary_m=boundary_mm / 1000,
        outer_m=outer_mm / 1000,
        pull=pull,
        pull_label=pull_label,
        undercut_ratio=bad,
        ok_ratio=ok,
        extension_mm=prof.to_meop_mm + prof.to_turn_mm,
        flange_mm=prof.flange_mm,
        solids=tuple(sorted({f.solid for f in mesh.faces if f.id in set(selected)})),
        assessment=assessment,
    )


def render_assessment(a: ps.ToolingAssessment, pull_label: str) -> None:
    """Side panel: the pull direction, its draft figures and the tooling decision with its trace."""
    ang = ps.pull_angles_deg(a.pull.direction)
    st.markdown(
        f"**Çekme yönü** ({pull_label}): "
        + ", ".join(f"{x:+.3f}" for x in a.pull.direction)
        + f"  \nEksenlere açı: X {ang['X']:.1f}°, Y {ang['Y']:.1f}°, Z {ang['Z']:.1f}°"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Ters açı payı", f"%{a.pull.undercut_ratio * 100:.1f}")
    m2.metric("Yeterli açı payı", f"%{a.pull.ok_ratio * 100:.0f}")
    m3.metric("En küçük açı", f"{a.pull.min_draft_deg:.1f}°")
    box = {
        ps.Solution.SINGLE: st.success,
        ps.Solution.CORE: st.info,
        ps.Solution.SPLIT: st.warning,
        ps.Solution.SPLIT_CORE: st.warning,
        ps.Solution.REVIEW: st.error,
    }[a.solution]
    box(f"**Kalıp kararı: {a.label}**\n\n" + "\n".join(f"- {r}" for r in a.reasons))
    for alt in a.alternatives:
        st.caption(f"Alternatif: {alt}")
    for fl in a.flags:
        st.warning(f"manual_review ({fl.code}): {fl.reason}")
    if a.regions:
        st.dataframe(
            [
                {
                    "bölge": k,
                    "tür": r.kind.value,
                    "alan [m²]": round(r.area_mm2 / 1e6, 4),
                    "pay [%]": round(r.area_ratio * 100, 2),
                    "en derin açı [°]": round(r.min_draft_deg, 1),
                    "yüzler": ", ".join(map(str, r.faces)),
                }
                for k, r in enumerate(a.regions, 1)
            ],
            hide_index=True,
        )
    with st.expander("Karar gerekçesi (formül, girdi, kaynak)"):
        for line in a.trace:
            st.markdown(f"**{line.description}** = {line.result:.4g} {line.unit}")
            st.caption(
                f"formül: {line.formula} · girdiler: "
                + "; ".join(
                    f"{i.name} = {i.value:.4g} {i.unit} ({i.source.value}"
                    + (f", {i.path}" if i.path else "")
                    + ")"
                    for i in line.inputs
                )
            )
    st.caption(
        "Kalıp yüzeyi ekranda kırmızı görünen yerler seçili yöndeki ters açılardır. Eşikler "
        "kütüphanede (thresholds.yaml: draft, undercut, pull_search) değiştirilebilir."
    )
