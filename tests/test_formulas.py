from __future__ import annotations

import math

import pytest

from teklif.costing import formulas as f


def test_learning_factor_no_learning_is_one() -> None:
    assert f.learning_factor(1.0, 1, 1, 50) == pytest.approx(1.0)


def test_learning_factor_two_units() -> None:
    # T1 = 1, T2 = 0.8 -> mean 0.9
    assert f.learning_factor(0.8, 1, 1, 2) == pytest.approx(0.9)


def test_learning_factor_repeat_order_and_reference_unit() -> None:
    # Units 3..4 with reference unit 2: ((3/2)^b + (4/2)^b) / 2
    b = math.log2(0.85)
    expected = ((3 / 2) ** b + 2**b) / 2
    assert f.learning_factor(0.85, 2, 3, 2) == pytest.approx(expected)


@pytest.mark.parametrize("lc", [0.0, -0.1, 1.2])
def test_learning_ratio_out_of_range(lc: float) -> None:
    with pytest.raises(ValueError):
        f.learning_factor(lc, 1, 1, 1)


def test_operation_time() -> None:
    # (30/5 + (2*10 + 3*4) * 1.2 * 0.9) / 0.8
    t = f.operation_time(30, 5, [(2, 10), (3, 4)], 1.2, 0.9, 0.8)
    assert t == pytest.approx((6 + 32 * 1.2 * 0.9) / 0.8)
    with pytest.raises(ValueError):
        f.operation_time(30, 0, [], 1, 1, 0.8)
    with pytest.raises(ValueError):
        f.operation_time(30, 5, [], 1, 1, 0)


def test_cost_formulas() -> None:
    assert f.labor_cost(90, 2, 40, 1) == pytest.approx(120)
    assert f.machine_cost(30, 60, 0.5) == pytest.approx(15)
    assert f.ply_material_cost(1.5, 4, 0.25, 50, 1) == pytest.approx(375)
    assert f.consumable_cost(2, 1.5, 0.1, 10, 1) == pytest.approx(33)
    assert f.overhead(1000, 0.2) == pytest.approx(200)


def test_autoclave() -> None:
    assert f.fits_in_autoclave(1000, 3000, 50, 4000, 1200)  # rotated
    assert not f.fits_in_autoclave(1000, 3000, 200, 3100, 1200)
    share = f.autoclave_share(1000, 500, 0, 4000, 1000, 0.5)
    assert share == pytest.approx(0.25)
    assert f.autoclave_cost(8, 100, 1, share) == pytest.approx(200)


def test_price_from_cost() -> None:
    assert f.price_from_cost(100, 0.2, "markup") == pytest.approx(120)
    assert f.price_from_cost(100, 0.2, "margin") == pytest.approx(125)
    with pytest.raises(ValueError):
        f.price_from_cost(100, 1.0, "margin")
    with pytest.raises(ValueError):
        f.price_from_cost(100, 0.2, "other")
