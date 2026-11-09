"""The slow analysis in the surface panel starts only when the user presses the button."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")
pytest.importorskip("streamlit")

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from streamlit.testing.v1 import AppTest


def _app() -> None:
    import hashlib
    import os
    from pathlib import Path

    import streamlit as st

    from teklif.app.surface_panel import render_surface_panel
    from teklif.library import load_library

    data = Path(os.environ["TEKLIF_TEST_STEP"]).read_bytes()
    key = f"k_{hashlib.sha256(data).hexdigest()[:16]}"
    st.session_state.setdefault(f"{key}_sel", [0])
    sel = render_surface_panel(data, load_library().thresholds, "k")
    out = "none" if sel is None else sel.assessment.label if sel.assessment else "x"
    st.markdown(f"RESULT::{out}")


def test_analysis_runs_only_after_button(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeBox(400, 300, 4).Shape(), STEPControl_AsIs)
    path = tmp_path / "p.step"
    w.Write(str(path))
    monkeypatch.setenv("TEKLIF_TEST_STEP", str(path))
    at = AppTest.from_function(_app).run(timeout=120)
    assert not at.exception
    assert [b.label for b in at.button if b.key and b.key.endswith("_run")] == ["Analizi başlat"]
    assert any(m.value == "RESULT::none" for m in at.markdown)  # nothing computed yet
    assert not any(str(k).endswith("_mesh") for k in at.session_state)  # no fine mesh yet
    run = next(b for b in at.button if b.key and b.key.endswith("_run"))
    run.click()
    at.run(timeout=120)
    assert not at.exception
    assert any(m.value.startswith("RESULT::") and m.value != "RESULT::none" for m in at.markdown)
    assert any(str(k).endswith("_mesh") for k in at.session_state)
    # any change of the selection hides the result until the button is pressed again
    key = next(k for k in at.session_state if str(k).endswith("_sel"))
    at.session_state[key] = [0, 1]
    at.run(timeout=120)
    assert not at.exception
    assert any(m.value == "RESULT::none" for m in at.markdown)
    assert [b.label for b in at.button if b.key and b.key.endswith("_run")] == ["Analizi başlat"]


def _waist_bar(path: Path) -> None:
    """Prism with a waist: its lower sides lean inwards, its upper sides outwards, so no single
    direction releases the tool around it, but a left/right split does."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Pnt, gp_Vec

    pts = [(-50, 0), (50, 0), (30, 50), (50, 100), (-50, 100), (-30, 50)]
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in pts:
        poly.Add(gp_Pnt(x, 0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    solid = BRepPrimAPI_MakePrism(face, gp_Vec(0, 100, 0)).Shape()
    w = STEPControl_Writer()
    w.Transfer(solid, STEPControl_AsIs)
    w.Write(str(path))


def _side_faces(path: Path) -> list[int]:
    from teklif.geometry.surface_select import load_face_mesh

    mesh = load_face_mesh(path)
    # everything but the flat top and bottom: four slanted sides and the two closed ends
    return [f.id for f in mesh.faces if abs(f.mean_normal[2]) < 0.99]


def test_split_analysis_colours_pieces_and_builds_parting_walls(tmp_path: Path) -> None:
    import numpy as np

    from teklif.app.surface_panel import (
        CORE_COLOR_IDX,
        PIECE_COLOR_IDX0,
        analysis_extra,
        piece_colors,
        piece_legend,
    )
    from teklif.draft import pull_search as ps
    from teklif.geometry.surface_select import load_face_mesh
    from teklif.library import load_library

    path = tmp_path / "waist.step"
    _waist_bar(path)
    ids = _side_faces(path)
    assert len(ids) == 6
    fine = load_face_mesh(path)
    view = load_face_mesh(path, 5.0, 10.0)
    ts = ps.triset_from_mesh(fine, ids)
    s = ps.settings_from_thresholds(load_library().thresholds)
    # whatever single pull the user looked at, a two-piece split without cores is found
    for base in ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]):
        other = ps.assess(ts, np.array(base), s).chosen
        assert other is not None and other.n_pieces == 2 and other.n_cores == 0
        assert other.cost < 4.0
    a = ps.assess(ts, np.array([0.0, 0.0, 1.0]), s)
    ch = a.chosen
    assert a.solution is ps.Solution.SPLIT and ch is not None and ch.n_pieces == 2
    nrm = ch.planes[0].normal
    assert abs(nrm[2]) < 0.3 and abs(nrm[0]) > 0.6  # a vertical plane cutting left from right
    colors = piece_colors(view, ids, ts, ch)
    assert len(colors) == sum(len(f.triangles) for f in view.faces)
    used = {c for c in colors if c >= PIECE_COLOR_IDX0}
    assert used == {PIECE_COLOR_IDX0, PIECE_COLOR_IDX0 + 1}  # one colour per piece, no cores
    assert CORE_COLOR_IDX not in used
    walls, wcolors, lines = analysis_extra(ts, ch)
    assert len(walls) > 0 and len(wcolors) == len(walls)
    assert len(lines) == 6  # an arrow (shaft + two head strokes) per piece
    assert "kalıp parça 2" in piece_legend(ch)


def test_panel_shows_the_split_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "waist.step"
    _waist_bar(path)
    ids = _side_faces(path)
    monkeypatch.setenv("TEKLIF_TEST_STEP", str(path))
    monkeypatch.setenv("TEKLIF_TEST_FACES", ",".join(map(str, ids)))

    def app() -> None:
        import hashlib
        import os
        from pathlib import Path

        import streamlit as st

        from teklif.app.surface_panel import render_surface_panel
        from teklif.library import load_library

        data = Path(os.environ["TEKLIF_TEST_STEP"]).read_bytes()
        key = f"k_{hashlib.sha256(data).hexdigest()[:16]}"
        st.session_state.setdefault(
            f"{key}_sel", [int(x) for x in os.environ["TEKLIF_TEST_FACES"].split(",")]
        )
        sel = render_surface_panel(data, load_library().thresholds, "k")
        st.markdown(f"RESULT::{sel.assessment.label if sel and sel.assessment else 'none'}")
        if sel:
            st.markdown(f"NOTE::{sel.note()}")

    at = AppTest.from_function(app).run(timeout=180)
    assert not at.exception
    run = next(b for b in at.button if b.key and b.key.endswith("_run"))
    run.click()
    at.run(timeout=180)
    assert not at.exception
    assert any(m.value.startswith("RESULT::Bölünmüş kalıp: 2 parça") for m in at.markdown)
    assert any("kalıp en fazla 3 parça" in m.value for m in at.markdown)
    # the user lowers the piece limit to 1: nothing can release it, no split is offered
    mp = next(n for n in at.number_input if n.key and n.key.endswith("_maxp"))
    mp.set_value(1).run(timeout=180)
    assert not at.exception
    run = next(b for b in at.button if b.key and b.key.endswith("_run"))
    run.click()
    at.run(timeout=180)
    assert not at.exception
    assert any(
        m.value.startswith("NOTE::") and "kalıp en fazla 1 parça" in m.value for m in at.markdown
    )
    assert not any(m.value.startswith("RESULT::Bölünmüş") for m in at.markdown)
