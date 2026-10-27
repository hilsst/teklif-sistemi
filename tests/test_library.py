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
    assert ("work_centers.yaml", "work_centers[TEMIZ_ODA].labor_rate") in todos
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
