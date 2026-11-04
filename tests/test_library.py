from __future__ import annotations

from pathlib import Path

import pytest

from teklif.common.units import is_known_unit
from teklif.library import Library, LibraryError, list_todos, load_library
from teklif.library.todo_report import render_todo_markdown


def test_default_library_loads(lib: Library) -> None:
    assert lib.currency.base_currency == "EUR"
    assert "autoclave_prepreg" in lib.processes
    assert lib.thresholds.draft.min_ok_deg.value == 2.0
    assert lib.thresholds.undercut.core_max_area_ratio.value == pytest.approx(0.10)


def test_autoclave_prepreg_template_covers_route(lib: Library) -> None:
    proc = lib.processes["autoclave_prepreg"]
    ops = [o.id for o in proc.operations]
    for required in (
        "kesim",
        "serim",
        "debulk",
        "torbalama",
        "kur",
        "kaliptan_cikarma",
        "trim",
        "ndt",
        "olcum",
        "boya",
        "montaj",
    ):
        assert required in ops
    assert {c.id for c in proc.cure_cycles} == {"cure_120", "cure_180"}
    assert sum(o.autoclave_cure for o in proc.operations) == 1


def test_todos_list_unfilled_values_only(lib: Library) -> None:
    todos = {(t.file, t.path) for t in list_todos(lib)}
    assert ("work_centers.yaml", "work_centers[AUTOCLAVE_YUKLEME].labor_rate") in todos
    assert ("currency.yaml", "rates.TRY") in todos
    assert ("currency.yaml", "rates.EUR") not in todos
    assert ("thresholds.yaml", "draft.min_ok_deg") not in todos
    md = render_todo_markdown(lib)
    assert f"**{len(todos)}**" in md


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("file", "old", "new", "message"),
    [
        (
            "processes/autoclave_prepreg.yaml",
            "work_center: KESIM_MASASI",
            "work_center: YOK",
            "unknown work_center 'YOK'",
        ),
        (
            "processes/autoclave_prepreg.yaml",
            'unit: min/ply, desc: "Etiketleme',
            'unit: min/m2, desc: "Etiketleme',
            "rate for ply_count must be 'min/ply'",
        ),
        (
            "processes/autoclave_prepreg.yaml",
            "qty: {value: null, unit: m/m,",
            "qty: {value: null, unit: m2/m,",
            "qty for SIZDIRMAZLIK_BANDI must be 'm/m'",
        ),
        (
            "processes/autoclave_prepreg.yaml",
            "material: PEEL_PLY",
            "material: YOK",
            "unknown consumable 'YOK'",
        ),
        (
            "work_centers.yaml",
            "currency: EUR, per: h",
            "currency: GBP, per: h",
            "unknown currency 'GBP'",
        ),
        ("materials.yaml", "unit: g/m2", "unit: gsm", "unknown unit 'gsm'"),
        (
            "materials.yaml",
            "tool_class: invar",
            "tool_class: titanyum",
            "unknown tool_class 'titanyum'",
        ),
        ("thresholds.yaml", "  yuksek:", "  cok_yuksek:", "complexity_factors keys"),
        ("currency.yaml", "EUR: {value: 1.0", "EUR: {value: 1.1", "base currency rate"),
    ],
)
def test_cross_validation_catches_errors(
    lib_copy: Path, file: str, old: str, new: str, message: str
) -> None:
    _edit(lib_copy / file, old, new)
    with pytest.raises(LibraryError) as exc:
        load_library(lib_copy)
    assert any(message in issue for issue in exc.value.issues), exc.value.issues


def test_schema_rejects_typo_in_field_name(lib_copy: Path) -> None:
    _edit(lib_copy / "autoclaves.yaml", "max_pressure:", "max_presure:")
    with pytest.raises(LibraryError):
        load_library(lib_copy)


def test_filled_value_drops_from_todo(lib_copy: Path) -> None:
    _edit(lib_copy / "currency.yaml", "TRY: {value: null", "TRY: {value: 41.2")
    todos = {(t.file, t.path) for t in list_todos(load_library(lib_copy))}
    assert ("currency.yaml", "rates.TRY") not in todos


def test_unit_vocabulary() -> None:
    assert is_known_unit("min/m2")
    assert is_known_unit("1e-6/K")
    assert is_known_unit("TRY/EUR", frozenset({"TRY", "EUR"}))
    assert not is_known_unit("min/")
    assert not is_known_unit("dakika")


def test_labor_rate_is_derived_from_wage_class(filled_lib: Library) -> None:
    wcs = {w.id: w for w in filled_lib.work_centers.work_centers}
    assert filled_lib.work_centers.engineer_rate is not None
    eng = filled_lib.work_centers.engineer_rate.value
    assert eng is not None
    assert wcs["TEMIZ_ODA"].labor.value == pytest.approx(eng * 0.5)
    assert wcs["ISLAK_SERIM"].labor.value == pytest.approx(eng * 0.2)
    assert wcs["CNC_TRIM"].labor.value == pytest.approx(eng * 1.0)
    assert wcs["TEMIZ_ODA"].labor.source.value == "derived"
    # a center without wage class keeps its own rate
    assert wcs["AUTOCLAVE_YUKLEME"].labor.source.value != "derived"


def test_derived_labor_rates_are_not_todos(lib: Library) -> None:
    paths = {t.path for t in list_todos(lib)}
    assert "engineer_rate" in paths
    assert not any(p.endswith("TEMIZ_ODA].labor_rate") for p in paths)


def test_wage_class_multiplier_must_be_positive(lib_copy: Path) -> None:
    import yaml

    path = lib_copy / "work_centers.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["wage_classes"][0]["multiplier"]["value"] = 0
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(LibraryError):
        load_library(lib_copy)


def test_library_without_wage_classes_still_loads(lib_copy: Path) -> None:
    import yaml

    path = lib_copy / "work_centers.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("engineer_rate")
    data.pop("wage_classes")
    for wc in data["work_centers"]:
        if "wage_class" in wc:
            wc.pop("wage_class")
            wc["labor_rate"] = {"value": 30.0, "currency": "EUR", "per": "h"}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    old = load_library(lib_copy)
    assert old.work_centers.engineer_rate is None
    assert all(w.labor.value in (30.0, None) for w in old.work_centers.work_centers)
