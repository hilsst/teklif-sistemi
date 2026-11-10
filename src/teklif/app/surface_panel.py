"""3D review screen: rotate/zoom, draft-angle colouring and tool-surface (face) selection.

Streamlit cannot take 3D click selection, so faces are picked from a numbered list (hover
shows the number) with helpers; a three.js component can replace this later.
The selection is the customer's part surface that lies on the tool. It drives tool area, trim
length (full boundary: outer edge, cut-outs and holes are all trimmed) and the pull direction.

Kemal (2026-11-09): the side screen no longer builds a mold (surface extension and flange). It
shows the best pull direction for the selected tool surface and the tooling decision (single
piece, core, split, split + core) from ``kalip_analiz.pull_search``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from kalip_analiz import pull_search as ps
from kalip_analiz.settings import load_settings
from kalip_analiz.viewer import (
    Extra,
    analysis_extra,
    core_rows,
    piece_colors,
    piece_legend,
    streamlit_component,
    triangle_colors,
    viewer_payload,
)
from teklif.geometry import mold_profile
from teklif.geometry.plate_features import PlateFeatures
from teklif.geometry.selection_features import SelectionFeatures, selection_features
from teklif.geometry.surface_select import (
    FaceMesh,
    Outline,
    faces_facing,
    load_face_mesh,
    suggest_pull_directions,
)

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
    features: SelectionFeatures | None = None  # area, holes, thickness of the selected faces
    max_pieces: int = 3  # most mold pieces allowed in this analysis (user input)
    max_pieces_library: int = 3  # the library default it started from

    def trim_m(self, all_openings: bool) -> float:
        """Trim length: whole boundary when every opening is trimmed, else the outer rim only."""
        return self.boundary_m if all_openings else self.outer_m

    def part_features(self, base: PlateFeatures) -> PlateFeatures:
        """``base`` (whole file or solid) with area, thickness and hole count replaced by the
        values measured on the selected faces; a value that could not be measured stays."""
        sf = self.features
        if sf is None:
            return base
        upd: dict[str, Any] = {
            "one_side_area_m2": sf.area_m2,
            "hole_count": sf.hole_count,
            "holes": [],
        }
        if sf.mean_thickness_mm > 0:
            upd["mean_thickness_mm"] = sf.mean_thickness_mm
        notes = [*base.notes, *(f"seçili yüzeyler: {n}" for n in sf.notes)]
        return base.model_copy(update={**upd, "notes": notes})

    def note(self) -> str:
        sf = self.features
        return (
            f"kalıp yüzeyi kullanıcı seçimi: {len(self.face_ids)} yüz, {self.area_m2:.3f} m², "
            f"sınır (tüm dış kenar, boşluk ve delikler trim) {self.boundary_m:.2f} m; çekme yönü "
            f"{self.pull_label} {tuple(round(x, 3) for x in self.pull)}; ters açı payı "
            f"%{self.undercut_ratio * 100:.1f}"
            + (
                f"; seçili yüzeylerden ölçülen: {sf.hole_count} delik, kalınlık "
                + (
                    f"{sf.mean_thickness_mm:.1f} mm"
                    if sf.mean_thickness_mm > 0
                    else "ölçülemedi (dosya değeri kullanıldı, incelenmeli)"
                )
                + (" [" + "; ".join(sf.notes) + "]" if sf.notes else "")
                if sf
                else ""
            )
            + (f"; sistem kalıp kararı: {self.assessment.label}" if self.assessment else "")
            + (
                f"; kullanıcı karşılaştırma tablosundan bu yöntemi seçti (sistem önerisi: "
                f"{self.assessment.recommended.label})"
                if self.assessment
                and self.assessment.recommended is not None
                and self.assessment.shown != 0
                else ""
            )
            + (
                f"; kalıp en fazla {self.max_pieces} parça"
                + (
                    ""
                    if self.max_pieces == self.max_pieces_library
                    else f" (kullanıcı girişi, kütüphane {self.max_pieces_library})"
                )
                if self.assessment
                else ""
            )
        )


_VIEWER = streamlit_component("face_viewer")


def mold_submesh(mesh: FaceMesh, selected: list[int]) -> FaceMesh:
    """Only the selected faces, renumbered 0..n-1 (the read-only mold-surface view)."""
    chosen = set(selected)
    return FaceMesh(
        [
            dataclasses.replace(f, id=i)
            for i, f in enumerate(f for f in mesh.faces if f.id in chosen)
        ]
    )


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


def _ack(key: str) -> int:
    """Time (ms) of the last viewer click the panel has taken over: the viewer drops the toggles it
    sent up to then from its pending list."""
    last = st.session_state.get(f"{key}_lastclick")
    try:
        return int(last[1]) if last and last[1] is not None else 0
    except (TypeError, ValueError, IndexError):
        return 0


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
    mp_ = load_settings(thresholds).mesh
    key = f"{key}_{hashlib.sha256(data).hexdigest()[:16]}"  # new file -> new mesh and selection
    mk = f"{key}_mesh"
    vk = mk + "v"
    if vk not in st.session_state:
        try:
            with st.spinner("3B model hazırlanıyor (büyük parçada birkaç saniye sürer)..."):
                st.session_state[vk] = _load(data, mp_.view_deflection_mm, mp_.view_angle_deg, True)
        except Exception as exc:
            st.warning(f"3B ekran açılamadı ({exc}); trim çevresi elle girilebilir.")
            return None
    view: FaceMesh = st.session_state[vk]
    mesh: FaceMesh = view  # replaced by the fine analysis mesh once something is selected
    hk = f"{key}_hidden" + ("_" + ",".join(map(str, sorted(only_solids))) if only_solids else "")
    base_hidden: frozenset[int] = (
        frozenset(f.solid for f in view.faces) - only_solids if only_solids else frozenset()
    )
    sk = f"{key}_sel"
    vkey = f"{key}_viewer"
    last = f"{key}_lastclick"
    hide_face: int | None = st.session_state.pop(f"{key}_hide_pending", None)
    # The viewer reports a click as its component value. Its latest value is already in the
    # session state when the script starts, so it is applied here, before the widgets are built:
    # one run per click (applying it after the viewer call needed a second st.rerun()).
    click_in = st.session_state.pop(f"{key}_pendclick", None) or st.session_state.get(vkey)
    if isinstance(click_in, dict):
        ident_in = (click_in.get("n"), click_in.get("t"))
        if ident_in != st.session_state.get(last):  # n restarts when the iframe is rebuilt
            st.session_state[last] = ident_in
            if "toggles" in click_in:  # faces the user clicked; relative, so nothing is overwritten
                st.session_state[sk] = sorted(
                    set(st.session_state.get(sk, ())) ^ {int(i) for i in click_in["toggles"]}
                )
            if "hide" in click_in:
                hide_face = int(click_in["hide"])
    if hide_face is not None and 0 <= hide_face < len(view.faces):
        # right click on an element of an assembly hides its whole solid (never the last one)
        solids = {f.solid for f in view.faces}
        gone = set(st.session_state.get(hk, ())) | {view.faces[hide_face].solid}
        if solids - gone - set(base_hidden):
            st.session_state[hk] = sorted(gone)
    user_hidden = frozenset(st.session_state.get(hk, ()))
    hidden = user_hidden | base_hidden
    visible_ids = {f.id for f in view.faces if f.solid not in hidden}
    st.session_state.setdefault(sk, [])
    st.session_state[sk] = [i for i in st.session_state[sk] if i in visible_ids]
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
    lib_max_pieces = settings.max_split_pieces
    tris: ps.TriSet | None = None
    # the analysis (fine mesh, pull search, tooling decision) is slow, so it runs only when the
    # user presses the button for the current selection; any change of the selection stops showing
    # the old result until the button is pressed again
    run_key = f"{key}_analysed"

    def _start() -> None:
        st.session_state[run_key] = (
            tuple(sorted(st.session_state.get(sk, []))),
            int(st.session_state.get(f"{key}_maxp", lib_max_pieces)),
        )

    max_pieces = min(max(lib_max_pieces, 1), 6)
    if selected:
        b1, b2 = st.columns([2, 3])
        max_pieces = int(
            b1.number_input(
                "Kalıp en fazla kaç parçadan oluşabilir?",
                1,
                6,
                min(max(lib_max_pieces, 1), 6),
                key=f"{key}_maxp",
                help="1: kalıp bölünmez (yalnız maça). Varsayılan kütüphaneden "
                "(thresholds.yaml → undercut.max_split_pieces); burada değiştirirseniz yalnız bu "
                "analiz için geçerli olur ve teklif notuna yazılır.",
            )
        )
    settings = replace(settings, max_split_pieces=max_pieces)
    analysed = bool(selected) and st.session_state.get(run_key) == (
        tuple(sorted(selected)),
        max_pieces,
    )
    if selected:
        b2.button(
            "Analizi yeniden başlat" if analysed else "Analizi başlat",
            on_click=_start,
            key=f"{key}_run",
            type="secondary" if analysed else "primary",
            help="Çekme yönünü arar, bölme yöntemlerini karşılaştırır, kalıp kararını verir.",
        )
        if not analysed:
            st.info(
                "Yüzeyleri seçmeyi ve parça sınırını girmeyi bitirince **Analizi başlat**'a "
                "basın; çekme yönü, trim çevresi, bölme yöntemleri ve kalıp kararı o zaman "
                "hesaplanır."
            )
    if analysed:
        if mk not in st.session_state:
            with st.spinner("Seçim için ince analiz ağı hazırlanıyor..."):
                st.session_state[mk] = _load(
                    data, mp_.linear_deflection_mm, mp_.angular_deflection_deg, True
                )
        mesh = st.session_state[mk]
        tris = ps.triset_from_mesh(mesh, selected)
        sk_search = f"{key}_ps_{hash((tuple(selected), replace(settings, max_split_pieces=0)))}"
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
            "Tek parça için optimize edilen yönler (ters açı payı / yeterli çekme payı / en "
            "küçük açı): "
            + "; ".join(
                f"#{k + 1} %{f.undercut_ratio * 100:.1f} / %{f.ok_ratio * 100:.0f} / "
                f"{f.min_draft_deg:.1f}°"
                for k, f in enumerate(found)
            )
        )
        pick = st.selectbox(
            "Tek parça için çekme yönü (en uygun başta)",
            [*options, "Elle gir"],
            key=f"{key}_pull",
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
    assessment: ps.ToolingAssessment | None = None
    table_key: str | None = None
    if pull is not None and tris is not None:
        ak = f"{key}_assess_{hash((tuple(selected), tuple(round(x, 6) for x in pull), settings))}"
        if ak not in st.session_state:
            with st.spinner("Bölme yöntemleri karşılaştırılıyor ve kalıp kararı veriliyor..."):
                st.session_state[ak] = ps.assess(tris, np.array(pull), settings)
        assessment = st.session_state[ak]
        # the user may pick any compared method in the table; everything on screen follows it
        table_key = f"{key}_mtbl_{abs(hash(ak))}"
        picked = _picked_row(table_key)
        if assessment is not None and 0 < picked < len(assessment.methods):
            assessment = assessment.view(picked)
    paint = "draft"
    if assessment is not None and assessment.chosen is not None:
        paint = (
            "pieces"
            if st.radio(
                "Boyama",
                ["Kalıp parçaları ve maça bölgeleri", "Çekme açısı (tek parça çekme yönü)"],
                horizontal=True,
                key=f"{key}_paint",
            ).startswith("Kalıp")
            else "draft"
        )
    colors: Sequence[str | int]
    extra: Extra | None = None
    if paint == "pieces" and assessment is not None and assessment.chosen is not None:
        assert tris is not None
        colors = piece_colors(view, selected, tris, assessment.chosen)
        extra = analysis_extra(tris, assessment.chosen, load_settings(thresholds).viewer)
    else:
        colors = triangle_colors(view, selected, pull, min_ok, warn_min)
    show_side = assessment is not None
    left, right = st.columns(2) if show_side else (st.container(), None)
    with left:
        click = _VIEWER(
            **viewer_payload(
                view,
                colors,
                key,
                pull,
                extra,
                hidden=hidden,
                selected=selected,
                ack=_ack(key),
                sync_ms=round(float(thresholds.viewer.select_pause_s.value) * 1000),
            ),
            height=560,
            key=vkey,
            default=None,
        )
    if right is not None and assessment is not None:
        with right:
            render_assessment(assessment, pull_label, table_key)
    ident = (click.get("n"), click.get("t")) if click else None
    if click and ident != st.session_state.get(last):
        # not seen at the top of the run (the component value was not in the session state):
        # keep it for the next run, as before
        st.session_state[f"{key}_pendclick"] = click
        st.rerun()
    if user_hidden and st.button(
        f"Gizlenen {len(user_hidden)} elemanı geri getir", key=f"{key}_unhide"
    ):
        st.session_state[hk] = []
        st.rerun()
    if st.checkbox("Eski görünüm (Plotly)", False, key=f"{key}_plotly"):
        st.plotly_chart(
            build_figure(view, triangle_colors(view, selected, pull, min_ok, warn_min)),
            width="stretch",
            key=f"{key}_fig",
        )
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
    fk = f"{key}_selfeat_{hash(tuple(selected))}"
    if fk not in st.session_state:
        with st.spinner("Seçili yüzlerin alan, delik ve kalınlık bilgileri ölçülüyor..."):
            st.session_state[fk] = selection_features(mesh, list(selected))
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
        features=st.session_state[fk],
        max_pieces=settings.max_split_pieces,
        max_pieces_library=lib_max_pieces,
    )


def _picked_row(table_key: str) -> int:
    """Row of the compared-methods table the user clicked (0 = none: the system's pick)."""
    try:
        rows = st.session_state[table_key].selection.rows
    except (KeyError, AttributeError):
        return 0
    return int(rows[0]) if rows else 0


def render_assessment(
    a: ps.ToolingAssessment, pull_label: str, table_key: str | None = None
) -> None:
    """Side panel: the tooling decision, the mold pieces (colour, pull direction, cores), the
    methods compared and the trace of every number."""
    box = {
        ps.Solution.SINGLE: st.success,
        ps.Solution.CORE: st.info,
        ps.Solution.SPLIT: st.warning,
        ps.Solution.SPLIT_CORE: st.warning,
        ps.Solution.REVIEW: st.error,
    }[a.solution]
    box(f"**Kalıp kararı: {a.label}**\n\n" + "\n".join(f"- {r}" for r in a.reasons))
    ch = a.chosen
    if ch is not None:
        st.markdown(piece_legend(ch), unsafe_allow_html=True)
        rows = []
        for k, p in enumerate(ch.pieces, 1):
            ang = ps.pull_angles_deg(p.pull.direction)
            rows.append(
                {
                    "parça": k,
                    "alan payı [%]": round(p.area_ratio * 100, 1),
                    "çekme yönü": ", ".join(f"{x:+.2f}" for x in p.pull.direction),
                    "eksenlere açı X/Y/Z [°]": f"{ang['X']:.0f}/{ang['Y']:.0f}/{ang['Z']:.0f}",
                    "en küçük açı [°]": round(p.pull.min_draft_deg, 1),
                    "maça": len(p.cores),
                    "çözülemeyen": len(p.unresolved),
                }
            )
        st.dataframe(rows, hide_index=True)
        crow = core_rows(ch)
        if crow:
            st.markdown(
                "**Maçalar** (3B'de aynı numaralı, aynı renkte işaret: artı yüzeyin üstünde, çizgi "
                "yüzeyden dışarı doğru; tabloda konum ve boyut)"
            )
            st.dataframe(crow, hide_index=True)
        for pl in ch.planes:
            st.caption(
                f"Ayırma düzlemi: normal {', '.join(f'{x:+.2f}' for x in pl.normal)}, "
                f"n·x = {pl.offset:.1f} mm (yüzeye dik ayırma yüzeyi 3B'de açık gri)."
            )
        st.caption(
            "Çekme yönü: parçanın kalıp parçasına göre hareket yönü; kalıp parçası ters yöne "
            "kayar. Parça rengindeki oklar bu yönü gösterir."
        )
    for fl in a.flags:
        st.warning(f"manual_review ({fl.code}): {fl.reason}")
    if len(a.methods) > 1:
        st.markdown(
            f"**Karşılaştırılan yöntemler** (en fazla {a.max_pieces} parça). Bir satıra "
            "tıklayınca o çözümün çıktısı (renkler, ayırma yüzeyleri, kalıp kararı) ekrana gelir."
        )
        st.dataframe(
            [
                {
                    "yöntem": m.label,
                    "parça": m.n_pieces,
                    "maça": m.n_cores,
                    "çözülemeyen": m.n_unresolved,
                    "maça alan payı [%]": round(m.core_share * 100, 1),
                    "puan": round(m.cost, 1),
                    "durum": ("▶ gösterilen" if m is ch else "")
                    + (" ★ sistem önerisi" if m is a.recommended else ""),
                }
                for m in a.methods
            ],
            hide_index=True,
            key=table_key,
            on_select="rerun" if table_key else "ignore",
            selection_mode="single-row",
        )
        st.caption(
            "Puan = bölme ağırlığı × (parça − 1) + maça ağırlığı × maça + maça alan ağırlığı × "
            "maçaların yüzey payı + ceza × çözülemeyen bölge + düşük çekme ağırlığı × 0–2° alan "
            "payı; ağırlıklar thresholds.yaml → "
            "undercut.*_cost_weight (varsayılan, TODO: gerçek maliyet oranlarıyla kalibre "
            "edilmeli)."
        )
    if a.regions:
        ang = ps.pull_angles_deg(a.pull.direction)
        st.markdown(
            f"**Tek parça çekme yönündeki ters açı bölgeleri** ({pull_label}: "
            + ", ".join(f"{x:+.3f}" for x in a.pull.direction)
            + f"; X {ang['X']:.1f}°, Y {ang['Y']:.1f}°, Z {ang['Z']:.1f}°)"
        )
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
    m1, m2, m3 = st.columns(3)
    m1.metric("Tek parça: ters açı payı", f"%{a.pull.undercut_ratio * 100:.1f}")
    m2.metric("Yeterli açı payı", f"%{a.pull.ok_ratio * 100:.0f}")
    m3.metric("En küçük açı", f"{a.pull.min_draft_deg:.1f}°")
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
        "Eşikler kütüphanede (thresholds.yaml: draft, undercut, pull_search) değiştirilebilir."
    )
