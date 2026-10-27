from __future__ import annotations

from pathlib import Path

from teklif.costing.engine import compute_quote
from teklif.costing.inputs import QuoteInput
from teklif.costing.regression import run_regression
from teklif.library import list_todos, load_library

DEMO = Path(__file__).resolve().parents[1] / "examples"


def test_demo_library_is_complete_and_quotes(example_input: QuoteInput) -> None:
    """Fails if a new TODO field is added without updating scripts/make_demo_library.py."""
    lib = load_library(DEMO / "demo_library")
    assert list_todos(lib) == []
    result = compute_quote(example_input, lib)
    assert not result.manual_review, result.review_items
    assert result.value("price.order_total") is not None


def test_demo_reference_set_runs() -> None:
    lib = load_library(DEMO / "demo_library")
    report = run_regression(DEMO / "demo_referans_teklifler.xlsx", lib)
    assert len(report.compared) == 6
    assert len(report.within) == 4
