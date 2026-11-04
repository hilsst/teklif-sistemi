from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from teklif.library import Library
from teklif.storage.catalog import Catalog
from teklif.storage.catalog_import import import_labor_history, import_tool_history
from teklif.storage.history_import import Value


@pytest.fixture
def cat(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "t.db", cad_dir=tmp_path / "cad")


TOOL_ROW: dict[str, Value] = {
    "kalip_no": "K-7",
    "tarih": date(2026, 3, 1),
    "parca_no": "P-100",
    "parca_adi": "Kapak",
    "proses": "autoclave_prepreg",
    "kalip_tipi": "erkek",
    "kalip_cinsi": "cfrp",
    "kalip_zarf_x": 1200.0,
    "kalip_zarf_y": 800.0,
    "parca_alani": 0.55,
    "master_model": "tooling_board",
    "ters_aci_cozumu": "yok",
    "kalip_parca_sayisi": 1,
    "alt_yapi": "E",
    "kalip_malzeme_maliyeti": 3000.0,
    "master_model_malzeme_maliyeti": 1000.0,
    "fiili_toplam_maliyet": 15000.0,
    "para_birimi": "EUR",
}

LABOR_ROW: dict[str, Value] = {
    "is_emri_no": "IE-001",
    "tarih": date(2026, 9, 1),
    "parca_no": "P-100",
    "proses": "autoclave_prepreg",
    "operasyon": "serim",
    "is_merkezi": "TEMIZ_ODA",
    "parti_adedi": 4,
    "operator_sayisi": 2,
    "hazirlik_suresi": 40.0,
    "fiili_iscilik": 960.0,
    "parca_alani": 0.55,
    "trim_cevresi": 3.1,
    "delik_sayisi": 6,
}


def test_tool_history_creates_tool_part_master_and_cost(cat: Catalog) -> None:
    summary = import_tool_history(cat, [TOOL_ROW])
    assert summary.created == {"parça": 1, "master model": 1, "kalıp": 1, "maliyet kaydı": 1}
    part = cat.tools_for_part(1)
    assert [t.tool_no for t in part] == ["K-7"]
    cost = cat.costs("tool", part[0].id)[0]
    assert (cost.total, cost.material, cost.other) == (15000.0, 4000.0, 11000.0)
    params = cat.params("tool", part[0].id)
    assert (
        params["tool_footprint_x_mm"].value == 1200.0
        and params["tool_footprint_x_mm"].source == "user"
    )
    # second import never overwrites
    again = import_tool_history(cat, [TOOL_ROW])
    assert again.created == {} and "K-7: kalıp zaten kayıtlı" in again.skipped[0]


def test_labor_history_converts_person_minutes_to_clock_minutes(cat: Catalog, lib: Library) -> None:
    summary = import_labor_history(cat, [LABOR_ROW], lib)
    assert summary.created == {"parça": 1, "operasyon": 1}
    op = cat.operations(1, "actual")[0]
    # (960 - 40) person-min / 2 persons / 4 parts = 115 clock min per part; setup 20 clock min
    assert op.run_min_per_part == pytest.approx(115.0)
    assert op.setup_min == pytest.approx(20.0)
    assert (op.persons, op.batch_size, op.seq) == (2.0, 4, 20)
    params = cat.params("part", 1)
    assert params["part_perimeter_m"].value == 3.1 and params["hole_count"].value == 6
    assert import_labor_history(cat, [LABOR_ROW], lib).created == {}


def test_labor_row_with_setup_larger_than_total_is_skipped(cat: Catalog, lib: Library) -> None:
    bad = dict(LABOR_ROW, hazirlik_suresi=2000.0)
    summary = import_labor_history(cat, [bad], lib)
    assert summary.created == {"parça": 1}
    assert "hazırlık toplamdan büyük" in summary.skipped[0]


def test_history_edge_cases_are_skipped_not_crashed(cat: Catalog, lib: Library) -> None:
    no_persons = {k: v for k, v in LABOR_ROW.items() if k != "operator_sayisi"}
    zero_batch = dict(LABOR_ROW, is_emri_no="IE-2", parti_adedi=0)
    summary = import_labor_history(cat, [no_persons, zero_batch], lib)
    assert summary.created.get("operasyon") is None
    assert len(summary.skipped) == 2


def test_tool_import_marks_master_included_and_survives_taken_master_no(cat: Catalog) -> None:
    import_tool_history(cat, [TOOL_ROW])
    assert cat.costs("tool", 1)[0].includes_master
    clash = dict(TOOL_ROW, kalip_no="K-8")
    from teklif.storage.catalog import MasterModel

    cat.add(MasterModel(model_no="K-8-M"))
    summary = import_tool_history(cat, [clash])
    assert "K-8-M zaten kayıtlı" in summary.skipped[0] and "kalıp" not in summary.created
