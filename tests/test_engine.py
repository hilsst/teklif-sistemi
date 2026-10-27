from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from teklif.common.models import Override, Price, Source
from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import QuoteInput
from teklif.library import Library, load_library

MONEY = {"material", "consumable", "labor", "machine", "autoclave", "total", "price"}


def test_todo_library_never_produces_money(lib: Library, example_input: QuoteInput) -> None:
    r = compute_quote(example_input, lib)
    money = [ln for ln in r.lines if ln.category in MONEY and ln.unit == "EUR"]
    assert money
    assert all(ln.result is None for ln in money)
    assert r.manual_review
    paths = {fl.path for _, fl in r.review_items if fl.code == "missing_value"}
    assert "materials.yaml:prepregs[PP_CF_WOVEN_120].price" in paths
    assert "work_centers.yaml:work_centers[TEMIZ_ODA].labor_rate" in paths
    # Geometry-only values are still computed and traced.
    share = r.get("autoclave.share")
    assert share.inputs[0].source is Source.USER
    # NRE was entered by the user, so it is known even with an empty library.
    assert r.value("nre.tooling") == pytest.approx(18000)


def test_filled_library_hand_calculation(filled_lib: Library, example_input: QuoteInput) -> None:
    r = compute_quote(example_input, filled_lib)
    assert not r.manual_review, r.review_items
    ply_area = 1.2 * 2 + 1.1 * 8
    ply_count = 10
    # test fill: scrap 0.3, prepreg 60 EUR/m2
    assert r.value("material.ply1") == pytest.approx(1.2 * 2 * 1.3 * 60)
    assert r.value("material.ply2") == pytest.approx(1.1 * 8 * 1.3 * 60)
    b = math.log2(0.9)
    lc = sum(n**b for n in range(1, 21)) / 20
    assert r.value("learning.factor") == pytest.approx(lc)
    # kesim: no complexity, no learning; setup 30/batch 5; rates 2 min/m2, 2 min/ply; eff 0.8
    assert r.value("time.kesim") == pytest.approx((30 / 5 + 2 * ply_area + 2 * ply_count) / 0.8)
    # serim: complexity orta 1.2, learning
    serim = (30 / 5 + (2 * ply_area + 2 * ply_count) * 1.2 * lc) / 0.8
    assert r.value("time.serim") == pytest.approx(serim)
    assert r.value("labor.serim") == pytest.approx(serim / 60 * 1 * 40)
    assert r.value("machine.serim") == pytest.approx(serim / 60 * 20)
    # debulk: ceil(10 / 4) = 3 debulks × 2 min, learning, no setup
    assert r.value("time.debulk") == pytest.approx(3 * 2 * lc / 0.8)
    share = (1600 + 200) * (1100 + 200) / (6000 * 2500 * 0.6)
    assert r.value("autoclave.share") == pytest.approx(share)
    assert r.value("autoclave.cost") == pytest.approx(6 * 150 * share)
    # sealant tape: tool perimeter 5.6 m × qty 1 × (1 + 0.1) × 60 (price fill)
    assert r.value("consumable.torbalama.SIZDIRMAZLIK_BANDI") == pytest.approx(5.6 * 1.1 * 60)

    direct = r.value("cost.direct")
    assert direct is not None
    parts = [
        "material.total",
        "consumable.total",
        "labor.total",
        "machine.total",
        "autoclave.total",
    ]
    assert direct == pytest.approx(sum(r.value(k) or 0 for k in parts))
    assert r.value("cost.overhead") == pytest.approx(direct * 0.25)
    assert r.value("price.unit") == pytest.approx(direct * 1.25 * 1.15)
    assert r.value("price.order_total") == pytest.approx(direct * 1.25 * 1.15 * 20 + 18000)
    # boya is optional and not included
    assert all(not ln.key.endswith(".boya") for ln in r.lines)


def test_override_propagates(filled_lib: Library, example_input: QuoteInput) -> None:
    base = compute_quote(example_input, filled_lib)
    inp = example_input.model_copy(
        update={"overrides": {"time.serim": Override(value=120.0, reason="ustabaşı")}}
    )
    r = compute_quote(inp, filled_lib)
    assert r.get("time.serim").result == base.get("time.serim").result
    assert r.value("time.serim") == 120.0
    assert r.value("labor.serim") == pytest.approx(120 / 60 * 40)
    delta = (r.value("labor.total") or 0) - (base.value("labor.total") or 0)
    assert delta == pytest.approx((120 - (base.value("time.serim") or 0)) / 60 * 40)


def test_unknown_override_key_is_an_error(filled_lib: Library, example_input: QuoteInput) -> None:
    inp = example_input.model_copy(
        update={"overrides": {"labor.yok": Override(value=1.0, reason="x")}}
    )
    with pytest.raises(QuoteInputError):
        compute_quote(inp, filled_lib)


def test_optional_and_excluded_operations(filled_lib: Library, example_input: QuoteInput) -> None:
    inp = example_input.model_copy(
        update={"include_optional": ["boya"], "exclude_operations": ["ndt"]}
    )
    keys = {ln.key for ln in compute_quote(inp, filled_lib).lines}
    assert "labor.boya" in keys and "consumable.boya.BOYA_ASTAR" in keys
    assert "labor.ndt" not in keys
    with pytest.raises(ValueError):
        compute_quote(example_input.model_copy(update={"include_optional": ["yok"]}), filled_lib)


def test_mixed_currency(filled_lib_dir: Path, tmp_path: Path, example_input: QuoteInput) -> None:
    dst = tmp_path / "lib"
    shutil.copytree(filled_lib_dir, dst)
    wc = dst / "work_centers.yaml"
    text = wc.read_text(encoding="utf-8")
    # First work center (KESIM_MASASI) labor rate in TRY: 1600 TRY/h at 40 TRY/EUR = 40 EUR/h
    text = text.replace("value: 40.0\n    currency: EUR", "value: 1600.0\n    currency: TRY", 1)
    wc.write_text(text, encoding="utf-8")
    lib = load_library(dst)
    assert lib.work_centers.work_centers[0].labor_rate.currency == "TRY"
    r = compute_quote(example_input, lib)
    line = r.get("labor.kesim")
    fx = next(i for i in line.inputs if i.name == "fx")
    assert fx.value == pytest.approx(1 / 40)
    assert fx.unit == "EUR/TRY"
    t = r.value("time.kesim") or 0
    assert line.result == pytest.approx(t / 60 * 1600 / 40)


def test_amortized_nre(filled_lib: Library, example_input: QuoteInput) -> None:
    r = compute_quote(example_input.model_copy(update={"amortize_nre": True}), filled_lib)
    unit = r.value("price.unit") or 0
    assert r.value("price.unit_final") == pytest.approx(unit + 18000 / 20)
    assert r.value("price.order_total") == pytest.approx((unit + 900) * 20)


def test_missing_nre_is_flagged(filled_lib: Library, example_input: QuoteInput) -> None:
    r = compute_quote(example_input.model_copy(update={"tooling_nre": None}), filled_lib)
    assert ("nre.tooling", "nre_missing") in {(k, fl.code) for k, fl in r.review_items}
    assert r.value("price.order_total") is None
    r2 = compute_quote(
        example_input.model_copy(update={"tooling_nre": None, "tooling_exists": True}),
        filled_lib,
    )
    assert not r2.manual_review
    assert r2.value("price.order_total") == pytest.approx((r2.value("price.unit") or 0) * 20)


def test_nre_in_other_currency(filled_lib: Library, example_input: QuoteInput) -> None:
    nre = Price(value=400000, currency="TRY", per="pcs")
    r = compute_quote(example_input.model_copy(update={"tooling_nre": nre}), filled_lib)
    assert r.value("nre.tooling") == pytest.approx(10000)


def test_checks_raise_flags(filled_lib: Library, example_input: QuoteInput) -> None:
    geo = example_input.geometry.model_copy(update={"tool_footprint_x_mm": 9000})
    inp = example_input.model_copy(
        update={"geometry": geo, "material": "PP_CF_WOVEN_180", "plies": [example_input.plies[0]]}
    )
    codes = {fl.code for _, fl in compute_quote(inp, filled_lib).review_items}
    assert {"does_not_fit", "cure_mismatch"} <= codes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process", "rtm"),
        ("material", "YOK"),
        ("cure_cycle", "cure_250"),
        ("complexity", "cok"),
        ("autoclave", "AC9"),
        ("quote_currency", "GBP"),
    ],
)
def test_bad_references(
    filled_lib: Library, example_input: QuoteInput, field: str, value: str
) -> None:
    with pytest.raises(QuoteInputError):
        compute_quote(example_input.model_copy(update={field: value}), filled_lib)
