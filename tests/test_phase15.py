"""Phase 1.5: blueprints, multi-item quotes, quantity sweep, minimum order, CSV, draft status."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import create_engine, text

from teklif.cli import main
from teklif.common.models import Override, Price
from teklif.costing import quote_set as qs_mod
from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import QuoteInput
from teklif.costing.quote_set import (
    NreItem,
    QuoteSetInput,
    blueprint_skeleton,
    compute_quote_set,
)
from teklif.costing.sweep import quantity_sweep
from teklif.library import Library, load_library
from teklif.reporting import quote_report
from teklif.reporting.csv_export import breakdown_csv, set_breakdown_csv, sweep_csv
from teklif.reporting.quote_report import (
    pdf_bytes,
    set_pdf_bytes,
    write_set_excel,
)
from teklif.storage.db import NotFinalizableError, Store

from .conftest import REPO


def _lib_with_min_order(filled_lib_dir: Path, tmp_path: Path, minimum: float) -> Library:
    dst = tmp_path / f"lib_min_{minimum}"
    shutil.copytree(filled_lib_dir, dst)
    path = dst / "thresholds.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["costing"]["min_order_value"]["value"] = minimum
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")
    return load_library(dst)


def _part(example_input: QuoteInput, part_no: str, **update: Any) -> QuoteInput:
    return example_input.model_copy(
        update={"part_no": part_no, "tooling_exists": True, "tooling_nre": None, **update}
    )


# --- minimum order -----------------------------------------------------------------


def test_min_order_default_is_off(filled_lib: Library, example_input: QuoteInput) -> None:
    result = compute_quote(example_input, filled_lib)
    keys = {ln.key for ln in result.lines}
    assert "price.min_order_topup" not in keys
    assert "price.order_raw" not in keys


def test_min_order_tops_up_total(
    filled_lib: Library, filled_lib_dir: Path, tmp_path: Path, example_input: QuoteInput
) -> None:
    raw = compute_quote(example_input, filled_lib).value("price.order_total")
    assert raw is not None
    lib = _lib_with_min_order(filled_lib_dir, tmp_path, raw + 1000.0)
    result = compute_quote(example_input, lib)
    assert result.value("price.order_raw") == pytest.approx(raw)
    assert result.value("price.min_order_topup") == pytest.approx(1000.0)
    assert result.value("price.order_total") == pytest.approx(raw + 1000.0)
    # unit price is not touched by the minimum
    assert result.value("price.unit_final") == pytest.approx(
        compute_quote(example_input, filled_lib).value("price.unit_final")
    )


def test_min_order_below_total_adds_nothing(
    filled_lib: Library, filled_lib_dir: Path, tmp_path: Path, example_input: QuoteInput
) -> None:
    raw = compute_quote(example_input, filled_lib).value("price.order_total")
    assert raw is not None
    lib = _lib_with_min_order(filled_lib_dir, tmp_path, raw / 2)
    result = compute_quote(example_input, lib)
    assert result.value("price.min_order_topup") == 0.0
    assert result.value("price.order_total") == pytest.approx(raw)


# --- separate master model NRE -----------------------------------------------------


def test_master_model_is_a_separate_nre_line(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    inp = example_input.model_copy(
        update={
            "tooling_nre": Price(value=12000.0, currency="EUR", per="pcs"),
            "master_nre": Price(value=3000.0, currency="EUR", per="pcs"),
        }
    )
    result = compute_quote(inp, filled_lib)
    assert result.value("nre.tooling") == pytest.approx(12000.0)
    assert result.value("nre.master") == pytest.approx(3000.0)
    assert result.value("nre.total") == pytest.approx(15000.0)
    base = compute_quote(example_input, filled_lib)  # tool + master = 18000 in one line
    unit = result.value("price.unit_final")
    assert unit is not None
    assert result.value("price.order_total") == pytest.approx(unit * inp.quantity + 15000.0)
    assert base.value("nre.tooling") == pytest.approx(18000.0)
    assert "nre.master" not in {ln.key for ln in base.lines}


def test_master_without_tooling_amount_is_flagged(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    inp = example_input.model_copy(
        update={"tooling_nre": None, "master_nre": Price(value=3000.0, currency="EUR", per="pcs")}
    )
    result = compute_quote(inp, filled_lib)
    assert result.value("price.order_total") is None
    assert any(fl.code == "nre_missing" for _, fl in result.review_items)


# --- quantity sweep ----------------------------------------------------------------


def test_sweep_matches_direct_runs(filled_lib: Library, example_input: QuoteInput) -> None:
    rows = quantity_sweep(example_input, filled_lib, [10, 1, 10, 50])
    assert [r.quantity for r in rows] == [1, 10, 50]  # sorted, de-duplicated
    one = compute_quote(
        example_input.model_copy(update={"quantity": 1, "batch_size": 1}), filled_lib
    )
    assert rows[0].batch_size == 1  # batch capped at quantity
    assert rows[0].unit_price == pytest.approx(one.value("price.unit_final"))
    assert rows[0].order_total == pytest.approx(one.value("price.order_total"))
    assert rows[1].batch_size == min(example_input.batch_size, 10)
    # with NRE the effective unit price falls with quantity
    eff = [r.effective_unit_price for r in rows]
    assert eff[0] is not None and eff[1] is not None and eff[2] is not None
    assert eff[0] > eff[1] > eff[2]


def test_sweep_uses_library_quantities_by_default(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    rows = quantity_sweep(example_input, filled_lib)
    assert [r.quantity for r in rows] == sorted(filled_lib.thresholds.costing.sweep_quantities)


def test_sweep_rejects_bad_quantity(filled_lib: Library, example_input: QuoteInput) -> None:
    with pytest.raises(ValueError):
        quantity_sweep(example_input, filled_lib, [0])


# --- blueprints and quote sets -----------------------------------------------------


def test_blueprint_skeletons(lib: Library) -> None:
    sk = blueprint_skeleton(lib, "kalip_ve_parca", "T-1")
    assert sk["blueprint"] == "kalip_ve_parca"
    assert [p["part_no"] for p in sk["parts"]] == ["P1"]
    assert sk["parts"][0]["tooling_exists"] is True
    assert [(n["id"], n["nre_type"]) for n in sk["nre_items"]] == [
        ("KALIP", "tool"),
        ("MASTER", "master"),
    ]
    only_tool = blueprint_skeleton(lib, "sadece_kalip", "T-2")
    assert only_tool["parts"] == []
    with pytest.raises(QuoteInputError):
        blueprint_skeleton(lib, "yok", "T-3")


def test_blueprint_skeleton_is_incomplete_until_filled(lib: Library) -> None:
    with pytest.raises(ValueError):
        QuoteSetInput.model_validate(blueprint_skeleton(lib, "kalip_ve_parca", "T-1"))


def _set_input(example_input: QuoteInput) -> QuoteSetInput:
    return QuoteSetInput(
        quote_no="S-1",
        customer="Müşteri A",
        blueprint="kalip_ve_parca",
        parts=[
            _part(example_input, "P1"),
            _part(example_input, "P2", quantity=5, batch_size=5),
        ],
        nre_items=[
            NreItem(
                id="KALIP",
                name="Kalıp",
                nre_type="tool",
                amount=Price(value=12000.0, currency="EUR", per="pcs"),
            ),
            NreItem(
                id="MASTER",
                name="Master model",
                nre_type="master",
                amount=Price(value=3000.0, currency="EUR", per="pcs"),
            ),
        ],
    )


def test_quote_set_totals(filled_lib: Library, example_input: QuoteInput) -> None:
    result = compute_quote_set(_set_input(example_input), filled_lib)
    totals = [r.value("price.order_total") for r in result.parts]
    assert all(t is not None for t in totals)
    assert result.value("set.parts_total") == pytest.approx(sum(t for t in totals if t))
    assert result.value("set.nre_total") == pytest.approx(15000.0)
    assert result.value("set.order_total") == pytest.approx(sum(t for t in totals if t) + 15000.0)
    assert result.parts[0].input.quote_no == "S-1"
    assert result.parts[0].input.customer == "Örnek müşteri"  # a part's own customer wins
    assert not result.manual_review


def test_quote_set_applies_minimum_once(
    filled_lib: Library, filled_lib_dir: Path, tmp_path: Path, example_input: QuoteInput
) -> None:
    base = compute_quote_set(_set_input(example_input), filled_lib)
    total = base.value("set.order_total")
    assert total is not None
    lib = _lib_with_min_order(filled_lib_dir, tmp_path, total + 500.0)
    result = compute_quote_set(_set_input(example_input), lib)
    assert result.value("set.min_order_topup") == pytest.approx(500.0)
    assert result.value("set.order_total") == pytest.approx(total + 500.0)
    for part in result.parts:  # parts never carry their own top-up
        assert "price.min_order_topup" not in {ln.key for ln in part.lines}


def test_quote_set_review_flags_roll_up(lib: Library, example_input: QuoteInput) -> None:
    # TODO library: parts cannot be priced, so the set is a draft with prefixed flags
    result = compute_quote_set(_set_input(example_input), lib)
    assert result.manual_review
    assert result.value("set.order_total") is None
    assert any(key.startswith("P1/") for key, _ in result.review_items)


def test_quote_set_validation(example_input: QuoteInput) -> None:
    with pytest.raises(ValueError, match="at least one"):
        QuoteSetInput(quote_no="S-0")
    dup = _part(example_input, "P1")
    with pytest.raises(ValueError, match="Duplicate"):
        QuoteSetInput(quote_no="S-0", parts=[dup, dup])


def test_nre_only_quote_and_set_override(filled_lib: Library) -> None:
    inp = QuoteSetInput(
        quote_no="S-2",
        nre_items=[
            NreItem(
                id="KALIP",
                name="Kalıp",
                nre_type="tool",
                amount=Price(value=1000.0, currency="USD", per="pcs"),
            )
        ],
        overrides={"set.nre_total": Override(value=900.0, reason="indirim")},
    )
    result = compute_quote_set(inp, filled_lib)
    fx = filled_lib.currency.rates["EUR"].value
    usd = filled_lib.currency.rates["USD"].value
    assert fx is not None and usd is not None
    assert result.value("nre.KALIP") == pytest.approx(1000.0 * fx / usd)
    assert result.value("set.order_total") == pytest.approx(900.0)


def test_set_unknown_override_is_an_error(filled_lib: Library, example_input: QuoteInput) -> None:
    inp = _set_input(example_input).model_copy(
        update={"overrides": {"set.nope": Override(value=1.0, reason="x")}}
    )
    with pytest.raises(QuoteInputError):
        compute_quote_set(inp, filled_lib)


# --- storage: draft / final --------------------------------------------------------


def test_final_is_refused_while_review_flags_are_open(
    tmp_path: Path, lib: Library, example_input: QuoteInput
) -> None:
    result = compute_quote(example_input, lib)  # TODO library -> flags
    assert result.manual_review
    store = Store(tmp_path / "t.sqlite")
    with pytest.raises(NotFinalizableError):
        store.save_quote(result, lib, final=True)
    draft = store.save_quote(result, lib)
    assert draft.status == "draft"
    with pytest.raises(NotFinalizableError):
        store.finalize(draft.id)
    assert store.get(draft.id).status == "draft"


def test_override_with_reason_closes_the_flag(
    tmp_path: Path, filled_lib: Library, example_input: QuoteInput
) -> None:
    # nre_missing is the only flag; entering the NRE as an override decides it
    open_input = example_input.model_copy(update={"tooling_nre": None})
    result = compute_quote(open_input, filled_lib)
    assert result.has_open_review
    decided = compute_quote(
        open_input.model_copy(
            update={"overrides": {"nre.tooling": Override(value=9000.0, reason="kalıpçı teklifi")}}
        ),
        filled_lib,
    )
    assert decided.manual_review  # the flag is still shown on the line ...
    assert not decided.has_open_review  # ... but it is decided, so the quote is not a draft
    row = Store(tmp_path / "t.sqlite").save_quote(decided, filled_lib, final=True)
    assert row.status == "final" and not row.manual_review


def test_final_when_clean(tmp_path: Path, filled_lib: Library, example_input: QuoteInput) -> None:
    store = Store(tmp_path / "t.sqlite")
    result = compute_quote(example_input, filled_lib)
    assert store.save_quote(result, filled_lib, final=True).status == "final"
    draft = store.save_quote(result, filled_lib)
    assert draft.status == "draft"
    assert store.finalize(draft.id).status == "final"


def test_quote_set_storage(tmp_path: Path, filled_lib: Library, example_input: QuoteInput) -> None:
    store = Store(tmp_path / "t.sqlite")
    result = compute_quote_set(_set_input(example_input), filled_lib)
    a = store.save_quote_set(result, filled_lib, final=True)
    b = store.save_quote_set(result, filled_lib)
    assert (a.revision, b.revision) == (1, 2)
    assert (a.status, b.status) == ("final", "draft")
    assert a.item_count == 4
    loaded = store.list_quote_sets()[-1].result()
    assert loaded.value("set.order_total") == pytest.approx(result.value("set.order_total"))


def test_set_final_refused_with_flags(
    tmp_path: Path, lib: Library, example_input: QuoteInput
) -> None:
    result = compute_quote_set(_set_input(example_input), lib)
    with pytest.raises(NotFinalizableError):
        Store(tmp_path / "t.sqlite").save_quote_set(result, lib, final=True)


def test_old_database_gets_status_column(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE quotes (id INTEGER PRIMARY KEY, quote_no VARCHAR(64), "
                "revision INTEGER, created_at DATETIME, part_no VARCHAR(128), "
                "customer VARCHAR(256), currency VARCHAR(8), quantity INTEGER, "
                "unit_cost FLOAT, unit_price FLOAT, order_total FLOAT, manual_review BOOLEAN, "
                "library_hash VARCHAR(64), result_json TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO quotes (quote_no, revision, part_no, currency, quantity, "
                "manual_review, library_hash, result_json) "
                "VALUES ('Q', 1, 'P', 'EUR', 1, 1, 'h', '{}')"
            )
        )
    store = Store(path)
    assert [q.status for q in store.list_quotes()] == ["draft"]


# --- reports -----------------------------------------------------------------------


def test_csv_exports(filled_lib: Library, example_input: QuoteInput) -> None:
    result = compute_quote(example_input, filled_lib)
    csv_text = breakdown_csv(result)
    assert csv_text.startswith("﻿kalem;")
    lines = csv_text.strip().split("\r\n")
    assert len(lines) == len(result.lines) + 1
    total = next(ln for ln in lines if ln.startswith("price.order_total;"))
    assert str(round(result.value("price.order_total") or 0, 6)).replace(".", ",") in total
    dotted = breakdown_csv(result, ".")
    assert str(round(result.value("price.order_total") or 0, 6)) in dotted

    rows = quantity_sweep(example_input, filled_lib, [1, 10])
    sweep = sweep_csv(rows, "EUR").strip().split("\r\n")
    assert len(sweep) == 3 and sweep[0].startswith("\ufeffadet;parti;")

    sres = compute_quote_set(_set_input(example_input), filled_lib)
    set_lines = set_breakdown_csv(sres).strip().split("\r\n")
    assert len(set_lines) == 1 + len(sres.lines) + sum(len(p.lines) for p in sres.parts)
    assert any(ln.startswith("P2/price.order_total;") for ln in set_lines)


def test_pdf_draft_watermark_follows_manual_review(
    monkeypatch: pytest.MonkeyPatch, lib: Library, filled_lib: Library, example_input: QuoteInput
) -> None:
    seen: list[bool] = []
    real = quote_report._pdf_doc

    def spy(draft: bool = False) -> Any:
        seen.append(draft)
        return real(draft)

    monkeypatch.setattr(quote_report, "_pdf_doc", spy)
    assert pdf_bytes(compute_quote(example_input, filled_lib)).startswith(b"%PDF")
    assert pdf_bytes(compute_quote(example_input, lib)).startswith(b"%PDF")
    assert set_pdf_bytes(compute_quote_set(_set_input(example_input), filled_lib))[:4] == b"%PDF"
    assert set_pdf_bytes(compute_quote_set(_set_input(example_input), lib))[:4] == b"%PDF"
    assert seen == [False, True, False, True]


def test_set_excel(tmp_path: Path, filled_lib: Library, example_input: QuoteInput) -> None:
    from openpyxl import load_workbook

    result = compute_quote_set(_set_input(example_input), filled_lib)
    wb = load_workbook(write_set_excel(result, tmp_path / "s.xlsx"))
    assert wb.sheetnames[0] == "Ozet"
    assert "K01 P1" in wb.sheetnames and "K02 P2" in wb.sheetnames
    labels = [row[0].value for row in wb["Ozet"].iter_rows()]
    assert "Teklif toplamı" in labels


# --- CLI ---------------------------------------------------------------------------

DEMO = REPO / "examples" / "demo_library"


def test_cli_new_and_quote_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "skeleton.yaml"
    assert (
        main(
            [
                "--library",
                str(DEMO),
                "new",
                "--blueprint",
                "kalip_ve_parca",
                "--quote-no",
                "S-9",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert main(["--library", str(DEMO), "quote-set", str(out)]) == 1  # not yet filled in
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    example = yaml.safe_load((REPO / "examples" / "ornek_parca.yaml").read_text(encoding="utf-8"))
    part = {k: v for k, v in example.items() if k not in ("quote_no", "tooling_nre")}  # no quote_no
    part["part_no"] = "P1"
    part["tooling_exists"] = True
    data["parts"] = [part]
    for n in data["nre_items"]:
        n["amount"]["value"] = 1000.0
    filled = tmp_path / "filled.yaml"
    filled.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    db = tmp_path / "t.sqlite"
    assert (
        main(
            [
                "--library",
                str(DEMO),
                "quote-set",
                str(filled),
                "--xlsx",
                str(tmp_path / "s.xlsx"),
                "--pdf",
                str(tmp_path / "s.pdf"),
                "--csv",
                str(tmp_path / "s.csv"),
                "--db",
                str(db),
                "--final",
            ]
        )
        == 0
    )
    assert "[final]" in capsys.readouterr().out
    assert (tmp_path / "s.pdf").read_bytes().startswith(b"%PDF")


def test_cli_quote_sweep_and_final_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = REPO / "examples" / "ornek_parca.yaml"
    assert (
        main(
            [
                "--library",
                str(DEMO),
                "quote",
                str(example),
                "--sweep",
                "1,10",
                "--sweep-csv",
                str(tmp_path / "w.csv"),
            ]
        )
        == 0
    )
    assert "10" in capsys.readouterr().out and (tmp_path / "w.csv").exists()
    # real library still has TODO values -> flags -> final refused
    assert main(["quote", str(example), "--db", str(tmp_path / "t.sqlite"), "--final"]) == 1
    assert main(["quote", str(example), "--final"]) == 1


def test_cli_blueprints(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["blueprints"]) == 0
    assert "kalip_ve_parca" in capsys.readouterr().out
    assert qs_mod  # module importable


def test_example_quote_set_runs_on_demo_library() -> None:
    data = yaml.safe_load(
        (REPO / "examples" / "ornek_teklif_seti.yaml").read_text(encoding="utf-8")
    )
    result = compute_quote_set(QuoteSetInput.model_validate(data), load_library(DEMO))
    assert not result.has_open_review
    assert result.value("set.nre_total") == pytest.approx(15000.0)
    assert [r.input.part_no for r in result.parts] == ["P-1001", "P-1002"]


# --- review findings (Opus) --------------------------------------------------------


def test_empty_override_reason_is_rejected() -> None:
    with pytest.raises(ValueError):
        Override(value=1.0, reason="   ")
    assert Override(value=1.0, reason="  neden ").reason == "neden"


def test_only_missing_value_flags_are_decided_by_override() -> None:
    from teklif.common.models import CostLine, ReviewFlag

    def line(code: str) -> CostLine:
        return CostLine(
            key="k",
            description="",
            formula="",
            inputs=[],
            result=None,
            unit="EUR",
            flags=[ReviewFlag(code=code, reason="x")],
            override=Override(value=1.0, reason="karar"),
        )

    assert line("missing_value").open_flags == []
    assert line("nre_missing").open_flags == []
    assert [f.code for f in line("does_not_fit").open_flags] == ["does_not_fit"]
    assert [f.code for f in line("invalid_input").open_flags] == ["invalid_input"]


def test_master_nre_with_existing_tooling_is_rejected(example_input: QuoteInput) -> None:
    data = example_input.model_dump()
    data.update(tooling_exists=True, master_nre={"value": 3000.0, "currency": "EUR", "per": "pcs"})
    with pytest.raises(ValueError, match="master_nre"):
        QuoteInput.model_validate(data)


def test_sweep_drops_order_level_overrides(filled_lib: Library, example_input: QuoteInput) -> None:
    inp = example_input.model_copy(
        update={
            "overrides": {
                "price.order_total": Override(value=5000.0, reason="sabit"),
                "labor.serim": Override(value=50.0, reason="ölçüm"),
            }
        }
    )
    rows = quantity_sweep(inp, filled_lib, [1, 20])
    assert all(r.dropped_overrides == ("price.order_total",) for r in rows)
    assert all(r.kept_overrides == ("labor.serim",) for r in rows)
    assert rows[0].order_total != rows[1].order_total  # the fixed total did not leak through
    assert "price.order_total" in sweep_csv(rows, "EUR")


def test_set_excel_survives_odd_part_numbers(
    tmp_path: Path, filled_lib: Library, example_input: QuoteInput
) -> None:
    from openpyxl import load_workbook

    long_a = "X" * 40 + "A"
    long_b = "X" * 40 + "B"
    inp = QuoteSetInput(
        quote_no="S-3",
        parts=[
            _part(example_input, "A/B:1"),
            _part(example_input, long_a),
            _part(example_input, long_b),
            _part(example_input, "(teklif)"),
        ],
    )
    result = compute_quote_set(inp, filled_lib)
    wb = load_workbook(write_set_excel(result, tmp_path / "odd.xlsx"))
    names = wb.sheetnames
    assert len(names) == len(set(names)) == 2 + 4
    assert all(len(n) <= 31 and not set("/\\?*[]:") & set(n) for n in names)


def test_null_min_order_is_a_todo_not_off(
    filled_lib_dir: Path, tmp_path: Path, example_input: QuoteInput
) -> None:
    dst = tmp_path / "lib_null_min"
    shutil.copytree(filled_lib_dir, dst)
    path = dst / "thresholds.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["costing"]["min_order_value"]["value"] = None
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")
    lib = load_library(dst)
    result = compute_quote(example_input, lib)
    assert result.value("price.order_total") is None
    assert result.has_open_review
    assert any(fl.code == "missing_value" for _, fl in result.open_review_items)
    sres = compute_quote_set(_set_input(example_input), lib)
    assert sres.value("set.order_total") is None and sres.has_open_review


def test_finalize_set(tmp_path: Path, filled_lib: Library, example_input: QuoteInput) -> None:
    store = Store(tmp_path / "t.sqlite")
    row = store.save_quote_set(compute_quote_set(_set_input(example_input), filled_lib), filled_lib)
    assert row.status == "draft"
    assert store.finalize_set(row.id).status == "final"


def test_csv_neutralises_formulas_and_avoids_exponents(
    filled_lib: Library, example_input: QuoteInput
) -> None:
    inp = example_input.model_copy(
        update={"overrides": {"labor.serim": Override(value=0.0000001, reason="=1+1")}}
    )
    text_ = breakdown_csv(compute_quote(inp, filled_lib), ".")
    assert "'=1+1" in text_ and ";=1+1;" not in text_
    assert "e-" not in text_.split("labor.serim;", 1)[1].split("\r\n", 1)[0].replace("e-mail", "")


def test_cli_csv_has_single_crlf_and_final_needs_db(tmp_path: Path) -> None:
    example = REPO / "examples" / "ornek_parca.yaml"
    out = tmp_path / "q.csv"
    assert main(["--library", str(DEMO), "quote", str(example), "--csv", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r\r" not in raw and b"\r\n" in raw
    pdf = tmp_path / "q.pdf"
    assert main(["--library", str(DEMO), "quote", str(example), "--final", "--pdf", str(pdf)]) == 1
    assert not pdf.exists()  # refused before any output is written
