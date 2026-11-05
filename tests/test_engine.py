from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import yaml

from teklif.common.models import Override, Price, Source
from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import PlyGroup, QuoteInput
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
    assert "work_centers.yaml:work_centers[AUTOCLAVE_YUKLEME].labor_rate" in paths
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
    data = yaml.safe_load(wc.read_text(encoding="utf-8"))
    # engineer rate in TRY: 3200 TRY/h at 40 TRY/EUR = 80 EUR/h; KESIM_MASASI is an operator
    data["engineer_rate"].update(value=3200.0, currency="TRY")
    wc.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    lib = load_library(dst)
    assert lib.work_centers.work_centers[0].labor.currency == "TRY"
    r = compute_quote(example_input, lib)
    line = r.get("labor.kesim")
    fx = next(i for i in line.inputs if i.name == "fx")
    assert fx.value == pytest.approx(1 / 40)
    assert fx.unit == "EUR/TRY"
    t = r.value("time.kesim") or 0
    assert line.result == pytest.approx(t / 60 * 3200 / 40)


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


def test_wage_factor_scales_labor_only_and_needs_reason(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    base = compute_quote(example_input, filled_lib)
    with pytest.raises(QuoteInputError):
        compute_quote(example_input.model_copy(update={"wage_factor": 0.9}), filled_lib)
    inp = example_input.model_copy(
        update={"wage_factor": 0.9, "wage_factor_reason": "rekabetçi teklif"}
    )
    r = compute_quote(inp, filled_lib)
    assert r.value("labor.total") == pytest.approx((base.value("labor.total") or 0) * 0.9)
    assert r.value("material.total") == base.value("material.total")
    assert r.value("machine.total") == base.value("machine.total")
    line = r.get("labor.serim")
    factor = next(i for i in line.inputs if i.name == "wage_factor")
    assert factor.value == 0.9 and factor.source.value == "user"
    assert r.input.wage_factor_reason == "rekabetçi teklif"


def test_wet_layup_has_no_autoclave_and_counts_resin(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    inp = example_input.model_copy(
        update={
            "process": "el_yatirma",
            "cure_cycle": "ambient",
            "material": "KUMAS_CF_DOKUMA",
            "plies": [PlyGroup(name="kumaş", area_m2=0.5, count=4)],
        }
    )
    r = compute_quote(inp, filled_lib)
    keys = {ln.key for ln in r.lines}
    assert {"time.recine_hazirlama", "time.kesim", "time.serim", "time.islatma"} <= keys
    assert r.value("autoclave.total") == 0
    assert "autoclave.share" not in keys
    # resin: ply area 4 x 0.5 = 2 m2 x qty x (1 + waste) x price, from the filled library
    assert (r.value("consumable.islatma.RECINE") or 0) > 0
    assert (r.value("labor.islatma") or 0) > 0


def test_sandwich_quote_includes_core_operations_and_trim_margin(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    base = compute_quote(example_input, filled_lib)
    inp = example_input.model_copy(update={"process": "sandvic_autoclave"})
    r = compute_quote(inp, filled_lib)
    keys = {ln.key for ln in r.lines}
    for op in ("cekirdek_kesim", "film_serim", "cekirdek_yerlestirme", "kenar_kapama"):
        assert f"time.{op}" in keys
    assert (r.value("consumable.cekirdek_kesim.CEKIRDEK") or 0) > 0
    assert (r.value("labor.cekirdek_kesim") or 0) > 0
    assert r.value("cost.unit") > base.value("cost.unit")  # type: ignore[operator]
    trim = r.get("time.trim")
    assert "× margin" in trim.formula
    assert next(i for i in trim.inputs if i.name == "margin").value == 1.05
    # holes no longer add trim time in the sandwich template
    more_holes = inp.model_copy(
        update={"geometry": inp.geometry.model_copy(update={"hole_count": 99})}
    )
    assert compute_quote(more_holes, filled_lib).value("time.trim") == r.value("time.trim")


def test_toplu_sandwich_uses_person_hours_from_table(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    geo = example_input.geometry.model_copy(
        update={"part_area_m2": 2.64, "tool_footprint_x_mm": 1000, "tool_footprint_y_mm": 1000}
    )
    inp = example_input.model_copy(update={"process": "sandvic_toplu", "geometry": geo})
    r = compute_quote(inp, filled_lib)
    # person-hours -> minutes at operators=1, efficiency=1 (ADAMSAAT_* centers)
    assert r.value("time.serim") == pytest.approx(202.45 + 134.72 * 2.64)
    assert r.value("time.kalip_hazirlik") == pytest.approx(60)
    assert r.value("time.kaliptan_cikarma") == pytest.approx(15)
    assert r.value("time.montaj") == pytest.approx(132)
    assert r.value("time.trim") == pytest.approx(63)  # 60 x 1.05 margin
    # table fit stays within 10% of the measured 8.75 person-hours for the 2.64 m2 panel
    assert (r.value("time.serim") or 0) / 60 == pytest.approx(8.75, rel=0.1)
    assert (r.value("autoclave.total") or 0) > 0


def test_unit_price_breakdown_sums_to_final_price(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    nre = Price(value=8000.0, currency="EUR", per="pcs", source=Source.USER)
    master = Price(value=2000.0, currency="EUR", per="pcs", source=Source.USER)
    inp = example_input.model_copy(
        update={
            "tooling_exists": False,
            "tooling_nre": nre,
            "master_nre": master,
            "amortize_nre": True,
            "nre_margin": True,
            "quantity": 10,
            "nre_quantity": 20,
        }
    )
    r = compute_quote(inp, filled_lib)
    part = r.value("price.part_cost") or 0
    tool = r.value("price.share_tooling") or 0
    model = r.value("price.share_master") or 0
    profit = r.value("price.profit") or 0
    assert tool == pytest.approx(8000 / 20)  # spread over nre_quantity, not the order quantity
    assert model == pytest.approx(2000 / 20)
    assert part + tool + model + profit == pytest.approx(r.value("price.unit_final"))
    assert profit > 0
    # margin on tool/model: sold price is above its cost
    assert (r.value("price.nre_priced") or 0) > 10000


def test_nre_without_margin_or_amortization_is_unchanged(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    r = compute_quote(example_input, filled_lib)
    keys = {ln.key for ln in r.lines}
    assert "price.nre_priced" not in keys
    assert "price.share_tooling" not in keys
    assert r.value("price.profit") == pytest.approx(
        (r.value("price.unit") or 0) - (r.value("cost.unit") or 0)
    )


def test_nest_efficiency_sets_fabric_consumption(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    base = compute_quote(example_input.model_copy(update={"nest_efficiency": 1.0}), filled_lib)
    eighty = compute_quote(example_input.model_copy(update={"nest_efficiency": 0.8}), filled_lib)
    fire = compute_quote(
        example_input.model_copy(update={"nest_efficiency": 0.8, "fire_ratio": 0.1}), filled_lib
    )
    m0, m80, mf = (r.value("material.total") or 0 for r in (base, eighty, fire))
    assert m80 == pytest.approx(m0 * 1.25)  # 1 m of fabric needs 1 / 0.8 = 1.25 m
    assert mf == pytest.approx(m0 * 1.25 * 1.1)


def test_partial_price_fills_todo_values_and_lists_them(
    lib: Library, example_input: QuoteInput
) -> None:
    from teklif.costing.partial import fill_missing

    assert compute_quote(example_input, lib).value("price.unit_final") is None
    filled, items = fill_missing(lib)
    assert items and all(isinstance(i, str) for i in items)
    part = compute_quote(example_input, filled)
    assert part.value("price.unit_final") is not None
    # the real library is untouched
    assert compute_quote(example_input, lib).value("price.unit_final") is None
