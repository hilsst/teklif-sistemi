from __future__ import annotations

from pathlib import Path

import pytest

from teklif.common.models import Source
from teklif.storage.catalog import Catalog, CatalogError
from teklif.storage.learning import (
    CaseInput,
    OpEntry,
    collect_model_costs,
    collect_tool_samples,
    fit_labor,
    fit_linear,
    fit_model_cost,
    labor_rows,
    save_case,
    write_cost_models,
)
from teklif.tooling_rules.cost_estimate import load_model, load_model_cost


def _eur(amount: float, currency: str) -> float | None:
    return amount if currency == "EUR" else None


def _cat(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "k.sqlite", tmp_path / "cad")


def _add(cat: Catalog, no: str, area: float, tool: float, model: float, hours: float) -> None:
    res = save_case(
        cat,
        CaseInput(
            part_no=no,
            ops=[OpEntry("serim", hours), OpEntry("trim", 1.0, actual=False)],
            tool_cost=tool,
            model_cost=model,
        ),
    )
    cat.set_param("part", res.part_id, "part_area_m2", area, "m2", _src())


def _src() -> Source:
    return Source.USER


def test_fit_recovers_a_line_and_reports_loo() -> None:
    xs = [2.5, 3.0, 3.5, 4.0, 4.5]
    ys = [4000 + 1000 * x for x in xs]
    fit = fit_linear(xs, ys)
    assert fit is not None
    assert fit.base == pytest.approx(4000) and fit.slope == pytest.approx(1000)
    assert fit.r_squared == pytest.approx(1.0) and fit.loo_max_error < 1e-9
    assert (fit.x_min, fit.x_max) == (2.5, 4.5)


def test_fit_needs_three_points_with_spread() -> None:
    assert fit_linear([1.0, 2.0], [10.0, 20.0]) is None
    assert fit_linear([2.0, 2.0, 2.0], [10.0, 11.0, 12.0]) is None


def test_case_roundtrip_feeds_fits(tmp_path: Path) -> None:
    cat = _cat(tmp_path)
    for i, area in enumerate([2.5, 3.0, 3.6, 4.2]):
        _add(cat, f"P{i}", area, 4000 + 1200 * area, 3000 + 100 * i, 2 + 2 * area)
    samples, _ = collect_tool_samples(cat, _eur)
    assert len(samples) == 4
    fit = fit_linear([s.area_m2 for s in samples], [s.cost for s in samples])
    assert fit is not None and fit.slope == pytest.approx(1200)
    assert fit_model_cost(collect_model_costs(cat, _eur)).n == 4  # type: ignore[union-attr]
    lab = fit_labor(labor_rows(cat), "serim")
    assert lab is not None and lab.slope == pytest.approx(2.0)
    # estimate entries (trim, kind estimate) are not counted as actual hours
    assert fit_labor(labor_rows(cat), "trim") is None


def test_duplicate_part_is_refused(tmp_path: Path) -> None:
    cat = _cat(tmp_path)
    save_case(cat, CaseInput(part_no="X1"))
    with pytest.raises(CatalogError):
        save_case(cat, CaseInput(part_no="X1"))


def test_cost_in_missing_currency_is_left_out(tmp_path: Path) -> None:
    cat = _cat(tmp_path)
    res = save_case(cat, CaseInput(part_no="T1", tool_cost=5000, currency="TRY"))
    cat.set_param("part", res.part_id, "part_area_m2", 3.0, "m2", _src())
    samples, notes = collect_tool_samples(cat, _eur)
    assert samples == [] and any("kuru yok" in n for n in notes)


def test_write_models_backs_up_and_reloads(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "library" / "tool_cost_model.yaml"
    path = tmp_path / "tool_cost_model.yaml"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    fit = fit_linear([2.5, 3.0, 3.5, 4.0], [7000, 8000, 9000, 10000])
    assert fit is not None
    backup = write_cost_models(path, fit, fit_model_cost([2000.0, 3000.0, 4000.0]), "test")
    assert backup is not None and backup.exists()
    m = load_model(path)
    assert m.base == pytest.approx(2000) and m.per_m2 == pytest.approx(2000)
    assert load_model_cost(path).mean == pytest.approx(3000)
    flat = fit_linear([2.0, 3.0, 4.0], [5000, 5000, 5000])
    assert flat is not None
    with pytest.raises(ValueError):
        write_cost_models(path, flat, None, "x")


def test_second_write_keeps_old_backups_and_model_block(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "library" / "tool_cost_model.yaml"
    path = tmp_path / "tool_cost_model.yaml"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    original_model = load_model_cost(path)
    fit = fit_linear([2.5, 3.0, 3.5, 4.0], [7000, 8000, 9000, 10000])
    assert fit is not None
    b1 = write_cost_models(path, fit, None, 'PANEL 24" A\\B')  # quotes/backslash are safe
    b2 = write_cost_models(path, fit, fit_model_cost([1000.0]), "again")  # one model: too few
    assert b1 is not None and b2 is not None and b1 != b2 and b1.exists() and b2.exists()
    assert load_model_cost(path) == original_model  # model block kept
    assert 'PANEL 24" A\\B' in b2.read_text(encoding="utf-8")


def test_tool_shared_by_several_parts_is_left_out(tmp_path: Path) -> None:
    cat = _cat(tmp_path)
    a = save_case(cat, CaseInput(part_no="S1", tool_cost=5000))
    b = save_case(cat, CaseInput(part_no="S2"))
    cat.set_param("part", a.part_id, "part_area_m2", 3.0, "m2", _src())
    cat.set_param("part", b.part_id, "part_area_m2", 4.0, "m2", _src())
    tool = cat.tools_for_part(a.part_id)[0]
    cat.link_tool_part(tool.id, b.part_id)
    samples, notes = collect_tool_samples(cat, _eur)
    assert samples == [] and any("birden çok" in n for n in notes)
