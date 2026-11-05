from pathlib import Path

from teklif.common.models import Source
from teklif.costing.consumable_suggest import ConsumableInputs, suggest
from teklif.costing.engine import compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.library import load_library

LIB = Path(__file__).resolve().parents[1] / "library"


def test_suggestions_are_geometric_minimums_without_invented_extras() -> None:
    by = {x.material: x for x in suggest(ConsumableInputs(skins=2))}
    assert by["CEKIRDEK"].qty == 1.0
    assert by["FILM_YAPISTIRICI"].qty == 2.0
    assert by["VAKUM_TORBA"].qty == 1.0
    assert by["KALIP_AYIRICI"].qty is None  # needs the user's datasheet figure
    by2 = {x.material: x for x in suggest(ConsumableInputs(skins=2, bag_extra=0.2))}
    assert by2["VAKUM_TORBA"].qty == 1.2


def test_override_replaces_missing_library_qty() -> None:
    lib = load_library(LIB)
    pid = "sandvic_toplu"
    pr = lib.processes[pid]
    base = {
        "quote_no": "t",
        "part_no": "t",
        "customer": "",
        "process": pid,
        "material": lib.materials.prepregs[0].id,
        "cure_cycle": pr.cure_cycles[0].id,
        "autoclave": lib.autoclaves.autoclaves[0].id,
        "complexity": next(iter(lib.thresholds.complexity_factors)),
        "plies": [PlyGroup(name="p", area_m2=1.0, count=1)],
        "geometry": PartGeometry(
            part_area_m2=1.0,
            part_perimeter_m=1.0,
            hole_count=0,
            tool_area_m2=2.0,
            tool_perimeter_m=3.0,
            tool_footprint_x_mm=1000,
            tool_footprint_y_mm=1000,
            source=Source.DERIVED,
        ),
        "quantity": 1,
        "batch_size": 1,
    }
    sug = {x.material: x.param() for x in suggest(ConsumableInputs(skins=2))}
    ov = {k: v for k, v in sug.items() if v is not None}
    r0 = compute_quote(QuoteInput(**base), lib)
    r1 = compute_quote(QuoteInput(**base, consumable_qty=ov), lib)
    paths0 = {fl.path for _, fl in r0.review_items if fl.path}
    paths1 = {fl.path for _, fl in r1.review_items if fl.path}
    assert any(".qty" in (p or "") and "CEKIRDEK" in (p or "") for p in paths0)
    assert not any(".qty" in (p or "") and "CEKIRDEK" in (p or "") for p in paths1)
