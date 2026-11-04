from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("OCP")

from teklif.storage.catalog import Catalog
from teklif.storage.catalog_examples import load_examples
from teklif.tooling_rules.cost_estimate import estimate_tool_cost

CSV = Path(__file__).resolve().parents[1] / "examples" / "kalip_ornekleri" / "radom_1522.csv"


def test_estimate_in_range() -> None:
    est = estimate_tool_cost(4.0)
    assert est.value == pytest.approx(4430 + 1224 * 4.0)
    assert est.in_range
    assert est.low < est.value < est.high


def test_estimate_outside_range_is_flagged() -> None:
    est = estimate_tool_cost(1.3)
    assert not est.in_range
    assert "outside the calibrated range" in est.note


def test_zero_area_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_tool_cost(0)


def test_examples_load_without_step_files(tmp_path: Path) -> None:
    cat = Catalog(tmp_path / "k.sqlite", cad_dir=tmp_path / "cad")
    first = load_examples(cat, CSV, tmp_path / "none")
    assert first.parts == 9
    assert any("no STEP file" in n for n in first.notes)
    again = load_examples(cat, CSV, tmp_path / "none")
    assert again.parts == 0 and len(again.notes) == 9
