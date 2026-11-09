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
