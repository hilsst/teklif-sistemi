from __future__ import annotations

import pytest

from teklif.common.currency import MissingRateError, convert
from teklif.common.models import Override, Param, Price, Source, product_cost_line


def test_product_cost_line_computes_and_traces_inputs() -> None:
    line = product_cost_line(
        "prepreg",
        "Prepreg cost",
        {
            "area": Param(value=2.5, unit="m2", source=Source.CAD),
            "price": Price(value=80.0, currency="EUR", per="m2"),
        },
        currency="EUR",
    )
    assert line.result == pytest.approx(200.0)
    assert line.formula == "area × price"
    assert [i.source for i in line.inputs] == [Source.CAD, Source.LIBRARY]
    assert line.inputs[1].unit == "EUR/m2"
    assert not line.manual_review


def test_todo_value_never_produces_a_number() -> None:
    line = product_cost_line(
        "prepreg",
        "Prepreg cost",
        {"area": Param(value=2.5, unit="m2"), "price": Price(currency="EUR", per="m2")},
        currency="EUR",
        paths={"price": "materials.yaml:prepregs[X].price"},
    )
    assert line.result is None
    assert line.manual_review
    assert line.flags[0].code == "missing_value"
    assert line.flags[0].path == "materials.yaml:prepregs[X].price"
    assert line.inputs[1].source is Source.TODO


def test_override_wins_and_keeps_computed_result() -> None:
    line = product_cost_line(
        "x", "x", {"a": Param(value=2.0, unit="m2"), "b": Param(value=3.0, unit="ratio")}, "EUR"
    )
    line.override = Override(value=7.0, reason="supplier quote")
    assert line.result == pytest.approx(6.0)
    assert line.final == pytest.approx(7.0)


def test_param_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        Param.model_validate({"value": 1, "unit": "m", "colour": "red"})


RATES = {
    "EUR": Param(value=1.0, unit="EUR/EUR"),
    "TRY": Param(value=40.0, unit="TRY/EUR"),
    "USD": Param(value=None, unit="USD/EUR"),
}


def test_convert_between_currencies() -> None:
    assert convert(10.0, "EUR", "TRY", RATES) == pytest.approx(400.0)
    assert convert(400.0, "TRY", "EUR", RATES) == pytest.approx(10.0)
    assert convert(5.0, "USD", "USD", RATES) == 5.0


def test_convert_with_todo_rate_raises() -> None:
    with pytest.raises(MissingRateError):
        convert(10.0, "EUR", "USD", RATES)
    with pytest.raises(MissingRateError):
        convert(10.0, "EUR", "GBP", RATES)
