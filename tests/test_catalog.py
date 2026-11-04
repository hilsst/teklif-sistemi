from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from teklif.common.models import Source
from teklif.ingest.step_props import StepProps, StepValue, derive_part_values
from teklif.storage.catalog import Catalog, CatalogError, MasterModel, Part, Tool


@pytest.fixture
def cat(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path / "t.db", cad_dir=tmp_path / "cad")


def test_cad_file_is_deduplicated_by_hash(cat: Catalog) -> None:
    a, created_a = cat.add_cad_file(b"STEP-DATA", "kapak.step", "part")
    b, created_b = cat.add_cad_file(b"STEP-DATA", "baska_ad.stp", "part")
    assert created_a and not created_b and a.id == b.id
    assert Path(a.stored_path).read_bytes() == b"STEP-DATA"
    with pytest.raises(CatalogError):
        cat.add_cad_file(b"x", "x.step", "yok")


def test_cad_value_override_needs_reason_and_history_is_kept(cat: Catalog) -> None:
    part = cat.add(Part(part_no="P-1"))
    cat.set_param("part", part.id, "part_area_m2", 0.5, "m2", Source.CAD)
    with pytest.raises(CatalogError):
        cat.set_param("part", part.id, "part_area_m2", 0.45, "m2", Source.USER)
    cat.set_param(
        "part", part.id, "part_area_m2", 0.45, "m2", Source.USER, override_reason="flanş hariç"
    )
    cur = cat.params("part", part.id)["part_area_m2"]
    assert (cur.value, cur.source, cur.override_reason) == (0.45, "user", "flanş hariç")
    assert [h.value for h in cat.param_history("part", part.id, "part_area_m2")] == [0.5, 0.45]
    # user value later edited by the user again: no reason needed
    cat.set_param("part", part.id, "part_area_m2", 0.46, "m2", Source.USER)


def test_record_values_never_clobbers_user_entry(cat: Catalog) -> None:
    part = cat.add(Part(part_no="P-1"))
    cat.set_param("part", part.id, "part_area_m2", 0.3, "m2", Source.USER)
    values = [
        StepValue(key="part_area_m2", value=0.5, unit="m2", source=Source.DERIVED),
        StepValue(key="mean_thickness_mm", value=2.0, unit="mm", source=Source.DERIVED),
    ]
    assert cat.record_values("part", part.id, values) == ["part_area_m2"]
    params = cat.params("part", part.id)
    assert params["part_area_m2"].value == 0.3
    assert params["mean_thickness_mm"].source == "derived"


def test_tool_serves_several_parts_and_cost_is_derived(cat: Catalog) -> None:
    tool = cat.add(Tool(tool_no="K-1", body_parts=2))
    left, right = cat.add(Part(part_no="L")), cat.add(Part(part_no="R"))
    cat.link_tool_part(tool.id, left.id)
    cat.link_tool_part(tool.id, right.id)
    cat.link_tool_part(tool.id, right.id)  # idempotent
    assert {p.part_no for p in (left, right)} == {"L", "R"}
    assert [t.tool_no for t in cat.tools_for_part(right.id)] == ["K-1"]

    model = cat.add(MasterModel(model_no="M-1"))
    rec = cat.add_cost("tool", tool.id, 12000.0, "EUR", material=4500.0, cost_date=date(2026, 1, 5))
    assert rec.other == 7500.0
    assert cat.add_cost("master_model", model.id, 3000.0, "EUR").other is None
    with pytest.raises(CatalogError):
        cat.add_cost("tool", tool.id, 100.0, "EUR", material=150.0)
    with pytest.raises(CatalogError):
        cat.add_cost("part", 1, 100.0, "EUR")


def test_operations_keep_actual_and_estimate_apart(cat: Catalog) -> None:
    part = cat.add(Part(part_no="P-1"))
    cat.add_operation(
        part.id,
        "actual",
        seq=10,
        operation="kesim",
        work_center="KESIM_MASASI",
        run_min_per_part=20,
    )
    cat.add_operation(
        part.id, "estimate", seq=20, operation="serim", work_center="TEMIZ_ODA", run_min_per_part=90
    )
    assert [o.operation for o in cat.operations(part.id, "actual")] == ["kesim"]
    assert len(cat.operations(part.id)) == 2
    with pytest.raises(CatalogError):
        cat.add_operation(part.id, "tahmin", seq=1)


def test_derive_part_values_flags_review() -> None:
    props = StepProps(
        solid_count=1,
        file_length_units=["millimetre"],
        values=[
            StepValue(key="surface_area_m2", value=1.0, unit="m2", source=Source.CAD),
            StepValue(key="volume_m3", value=0.002, unit="m3", source=Source.CAD),
        ],
    )
    area, thick = derive_part_values(props)
    assert area.value == 0.5 and area.review_reason
    assert thick.value == pytest.approx(4.0)  # 0.002 m3 / 0.5 m2 = 4 mm


def test_duplicate_numbers_and_zero_cost_are_rejected(cat: Catalog) -> None:
    cat.add(Part(part_no="P-1", rev="A"))
    with pytest.raises(CatalogError):
        cat.add(Part(part_no="P-1", rev="A"))
    cat.add(Part(part_no="P-1", rev="B"))
    tool = cat.add(Tool(tool_no="K-1"))
    with pytest.raises(CatalogError):
        cat.add_cost("tool", tool.id, 0.0, "EUR")


def test_cad_file_measured_marker(cat: Catalog) -> None:
    row, _ = cat.add_cad_file(b"abc", "a.step", "part")
    assert row.extractor_version is None
    cat.mark_measured(row.id, "v1")
    again, created = cat.add_cad_file(b"abc", "a.step", "part")
    assert not created and again.extractor_version == "v1"
