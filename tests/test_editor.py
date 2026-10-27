from __future__ import annotations

import csv
from pathlib import Path

import pytest

from teklif.library import LibraryError, list_todos, load_library
from teklif.library.editor import Change, current_choice, editable_values, set_values


def test_switch_margin_formula_keeps_comments(lib_copy: Path) -> None:
    lib = set_values(
        lib_copy,
        [
            Change("thresholds.yaml", "costing.margin_basis", "margin"),
            Change("thresholds.yaml", "costing.margin_ratio", 0.2),
        ],
        "müşteri sözleşmesi marj tanımı",
        user="Kemal",
    )
    assert lib.thresholds.costing.margin_basis == "margin"
    assert lib.thresholds.costing.margin_ratio.value == pytest.approx(0.2)
    assert lib.thresholds.costing.margin_ratio.source.value == "user"
    assert current_choice(lib, "thresholds.yaml", "costing.margin_basis") == "margin"
    text = (lib_copy / "thresholds.yaml").read_text(encoding="utf-8")
    assert "# markup: fiyat = maliyet × (1 + oran)" in text
    rows = list(csv.reader((lib_copy / "changelog.csv").open(encoding="utf-8")))
    assert rows[0][:4] == ["zaman", "kullanici", "dosya", "yol"]
    assert [(r[3], r[4], r[5]) for r in rows[1:]] == [
        ("costing.margin_basis", "markup", "margin"),
        ("costing.margin_ratio", "None", "0.2"),
    ]


def test_fill_todo_price_in_other_currency(lib_copy: Path) -> None:
    lib = set_values(
        lib_copy,
        [
            Change("work_centers.yaml", "work_centers[TEMIZ_ODA].labor_rate", 1100, "TRY"),
            Change(
                "processes/autoclave_prepreg.yaml", "operations[kesim].times[ply_count].rate", 0.5
            ),
        ],
        "2026 ücret tablosu",
    )
    wc = next(w for w in lib.work_centers.work_centers if w.id == "TEMIZ_ODA")
    assert (wc.labor_rate.value, wc.labor_rate.currency) == (1100, "TRY")
    todos = {(t.file, t.path) for t in list_todos(lib)}
    assert ("work_centers.yaml", "work_centers[TEMIZ_ODA].labor_rate") not in todos


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (Change("thresholds.yaml", "costing.margin_basis", "yuzde"), ValueError),
        (Change("thresholds.yaml", "costing.yok", 1.0), KeyError),
        (
            Change("work_centers.yaml", "work_centers[TEMIZ_ODA].labor_rate", 30, "GBP"),
            LibraryError,
        ),
    ],
)
def test_invalid_change_restores_files(
    lib_copy: Path, change: Change, error: type[Exception]
) -> None:
    before = {p: p.read_bytes() for p in lib_copy.rglob("*.yaml")}
    with pytest.raises(error):
        set_values(lib_copy, [change], "deneme")
    assert {p: p.read_bytes() for p in lib_copy.rglob("*.yaml")} == before
    assert not (lib_copy / "changelog.csv").exists()


def test_reason_required(lib_copy: Path) -> None:
    with pytest.raises(ValueError):
        set_values(lib_copy, [Change("thresholds.yaml", "costing.margin_ratio", 0.1)], " ")


def test_editable_values_cover_library(lib_copy: Path) -> None:
    lib = load_library(lib_copy)
    values = editable_values(lib)
    assert sum(v.value is None for v in values) == len(list_todos(lib))
    price = next(v for v in values if v.path == "work_centers[TEMIZ_ODA].labor_rate")
    assert (price.unit, price.currency) == ("h", "EUR")
