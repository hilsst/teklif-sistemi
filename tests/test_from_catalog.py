from __future__ import annotations

from pathlib import Path

import pytest

from teklif.common.models import Source
from teklif.costing.engine import compute_quote
from teklif.costing.from_catalog import quote_input_from_catalog
from teklif.library import Library
from teklif.storage.catalog import Catalog, MasterModel, Part, Tool


@pytest.fixture
def cat(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "t.db", cad_dir=tmp_path / "cad")


def _fill(cat: Catalog, lib: Library, with_tool_values: bool = True) -> tuple[int, int]:
    first = lib.processes[next(iter(lib.processes))]
    part = cat.add(
        Part(
            part_no="P-1",
            process=first.id,
            prepreg=lib.materials.prepregs[0].id,
            cure_cycle=first.cure_cycles[0].id,
            complexity="orta",
        )
    )
    model = cat.add(MasterModel(model_no="M-1"))
    tool = cat.add(Tool(tool_no="K-1", master_model_id=model.id))
    cat.link_tool_part(tool.id, part.id)
    cat.add_ply(part.id, 0.4, 6, name="UD")
    cat.set_param(
        "part", part.id, "part_area_m2", 0.5, "m2", Source.DERIVED, review_reason="tahmin"
    )
    cat.set_param("part", part.id, "part_perimeter_m", 3.0, "m", Source.USER)
    cat.set_param("part", part.id, "hole_count", 0, "hole", Source.USER)
    if with_tool_values:
        for key, val, unit in (
            ("tool_area_m2", 0.9, "m2"),
            ("tool_perimeter_m", 4.0, "m"),
            ("tool_footprint_x_mm", 1000, "mm"),
            ("tool_footprint_y_mm", 700, "mm"),
        ):
            cat.set_param("tool", tool.id, key, val, unit, Source.USER)
    cat.add_cost("tool", tool.id, 9000.0, "EUR", material=3000.0)
    cat.add_cost("master_model", model.id, 2000.0, "EUR")
    return part.id, tool.id


def test_missing_values_are_listed_not_guessed(cat: Catalog, filled_lib: Library) -> None:
    part_id, tool_id = _fill(cat, filled_lib, with_tool_values=False)
    res = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q-1", 10, 5)
    assert res.input is None
    assert any("kalıp torbalama alanı" in m for m in res.missing)
    no_tool = quote_input_from_catalog(cat, filled_lib, part_id, None, "Q-1", 10, 5)
    assert "kalıp seçilmedi" in no_tool.missing


def test_catalog_quote_computes_and_flags_derived_value(cat: Catalog, filled_lib: Library) -> None:
    part_id, tool_id = _fill(cat, filled_lib)
    res = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q-1", 10, 5, 0.9, "rekabet")
    assert res.input is not None and res.missing == []
    inp = res.input
    assert inp.geometry.source == Source.USER
    assert inp.tooling_nre is not None and inp.tooling_nre.value == 9000.0
    assert inp.master_nre is not None and inp.master_nre.value == 2000.0
    result = compute_quote(inp, filled_lib)
    flags = [fl.code for key, fl in result.review_items if key == "input.review"]
    assert flags == ["catalog_review"]
    assert result.has_open_review  # derived value awaits a person


def test_streamlit_form_takes_catalog_input_and_keeps_review_flag(
    tmp_path: Path, filled_lib_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from teklif.library import load_library

    lib = load_library(filled_lib_dir)
    cat = Catalog(tmp_path / "c.db", cad_dir=tmp_path / "cad")
    part_id, tool_id = _fill(cat, lib)
    res = quote_input_from_catalog(cat, lib, part_id, tool_id, "Q-9", 10, 5)
    assert res.input is not None

    monkeypatch.setenv("TEKLIF_DB", str(tmp_path / "app.sqlite"))
    app = Path(__file__).resolve().parents[1] / "src/teklif/app/streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.session_state["catalog_base"] = res.input.model_dump(mode="json")
    at.run()
    at.sidebar.text_input[0].set_value(str(filled_lib_dir)).run()
    assert not at.exception, at.exception
    assert any("katalogdaki parça" in i.value for i in at.info)
    labels = [b.label for b in at.button]
    # derived part area keeps the quote a draft: finalizing is blocked
    assert at.button[labels.index("Kesinleştir ve kaydet")].disabled


def test_master_cost_rules_and_zero_values(cat: Catalog, filled_lib: Library) -> None:
    part_id, tool_id = _fill(cat, filled_lib)
    ok = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q", 1, 1)
    assert ok.input is not None
    # tool total that already contains the master model: master is not added again
    cat.add_cost("tool", tool_id, 11000.0, "EUR", includes_master=True)
    inc = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q", 1, 1)
    assert inc.input is not None and inc.input.master_nre is None
    assert inc.input.tooling_nre is not None and inc.input.tooling_nre.value == 11000.0

    # zero / non-integer values are reported as missing, not crashes
    cat.set_param("part", part_id, "part_area_m2", 0.0, "m2", Source.USER, override_reason="test")
    cat.set_param("part", part_id, "hole_count", 2.5, "hole", Source.USER)
    bad = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q", 1, 1)
    assert bad.input is None
    assert any("sıfırdan büyük" in m for m in bad.missing)
    assert any("tam sayı" in m for m in bad.missing)


def test_master_without_cost_is_reported(cat: Catalog, filled_lib: Library) -> None:
    part_id, tool_id = _fill(cat, filled_lib)
    from sqlalchemy.orm import Session

    from teklif.storage.catalog import CostRecord

    with Session(cat.engine) as s:
        for rec in s.query(CostRecord).filter(CostRecord.entity == "master_model"):
            s.delete(rec)
        s.commit()
    res = quote_input_from_catalog(cat, filled_lib, part_id, tool_id, "Q", 1, 1)
    assert res.input is None and any("master model maliyeti" in m for m in res.missing)
