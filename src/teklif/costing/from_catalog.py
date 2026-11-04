"""Build a ``QuoteInput`` from the data-entry catalog (part card + tool card).

Nothing is guessed: every missing value is listed and no input is produced. Values that
came from CAD with a review reason (or were derived) are passed on as review notes, so the
quote stays a draft until a person has looked at them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from teklif.common.models import Price, Source
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.library import Library
from teklif.storage.catalog import Catalog, CostRecord, ParamValue, Part, Tool

PART_KEYS = ("part_area_m2", "part_perimeter_m", "hole_count")
TOOL_KEYS = ("tool_area_m2", "tool_perimeter_m", "tool_footprint_x_mm", "tool_footprint_y_mm")
LABELS = {
    "part_area_m2": "parça yüzey alanı",
    "part_perimeter_m": "trim kenarı uzunluğu",
    "hole_count": "delik sayısı",
    "tool_area_m2": "kalıp torbalama alanı",
    "tool_perimeter_m": "kalıp torbalama çevresi",
    "tool_footprint_x_mm": "kalıp izi X",
    "tool_footprint_y_mm": "kalıp izi Y",
}


@dataclass
class CatalogQuote:
    input: QuoteInput | None = None
    missing: list[str] = field(default_factory=list)


def _latest_cost(cat: Catalog, entity: str, entity_id: int | None) -> CostRecord | None:
    if entity_id is None:
        return None
    costs = cat.costs(entity, entity_id)
    return costs[-1] if costs else None


def quote_input_from_catalog(
    cat: Catalog,
    lib: Library,
    part_id: int,
    tool_id: int | None,
    quote_no: str,
    quantity: int,
    batch_size: int,
    wage_factor: float = 1.0,
    wage_factor_reason: str = "",
) -> CatalogQuote:
    with Session(cat.engine) as s:
        part = s.get(Part, part_id)
        tool = s.get(Tool, tool_id) if tool_id is not None else None
    if part is None:
        return CatalogQuote(missing=["parça bulunamadı"])
    missing: list[str] = []
    for label, value in (
        ("proses", part.process),
        ("prepreg", part.prepreg),
        ("kür çevrimi", part.cure_cycle),
        ("karmaşıklık", part.complexity),
    ):
        if not value:
            missing.append(f"parça kartında {label} yok")
    if tool is None:
        missing.append("kalıp seçilmedi")
    plies = cat.plies(part.id)
    if not plies:
        missing.append("ply tablosu boş")

    used: list[ParamValue] = []
    values: dict[str, float] = {}
    for entity, owner, keys in (("part", part, PART_KEYS), ("tool", tool, TOOL_KEYS)):
        if owner is None:
            continue
        params = cat.params(entity, owner.id)
        for key in keys:
            p = params.get(key)
            if p is None or p.value is None or not math.isfinite(p.value):
                missing.append(f"{LABELS[key]} girilmemiş ({entity} kartı)")
            elif key != "hole_count" and p.value <= 0:
                missing.append(f"{LABELS[key]} sıfırdan büyük olmalı ({entity} kartı)")
            elif key == "hole_count" and (p.value < 0 or p.value != int(p.value)):
                missing.append(f"{LABELS[key]} tam sayı olmalı ({entity} kartı)")
            else:
                values[key] = p.value
                used.append(p)
    tool_cost = _latest_cost(cat, "tool", tool.id) if tool is not None else None
    master_cost = _latest_cost(cat, "master_model", tool.master_model_id) if tool else None
    if tool is not None and tool.master_model_id is not None:
        if tool_cost is not None and tool_cost.includes_master:
            master_cost = None  # already inside the tool total: never add it twice
        elif master_cost is None:
            missing.append("master model maliyeti girilmemiş (kalıp toplamı onu içermiyor)")
    if missing or tool is None:
        return CatalogQuote(missing=missing)

    sources = {p.source for p in used}
    source = (
        Source.USER
        if Source.USER.value in sources
        else Source.DERIVED
        if Source.DERIVED.value in sources
        else Source.CAD
    )
    notes = [
        f"{p.key}: {p.review_reason or 'türetilmiş değer, kontrol edin'}"
        for p in used
        if p.review_reason or p.source == Source.DERIVED.value
    ]

    def nre(rec: CostRecord | None) -> Price | None:
        return (
            None
            if rec is None
            else Price(
                value=rec.total,
                currency=rec.currency,
                per="pcs",
                source=Source.USER,
                note=f"katalog maliyet kaydı #{rec.id} ({rec.cost_date or 'tarihsiz'})",
            )
        )

    inp = QuoteInput(
        quote_no=quote_no,
        part_no=part.part_no,
        part_name=part.name,
        customer=part.customer,
        process=part.process,
        material=part.prepreg,
        cure_cycle=part.cure_cycle,
        complexity=part.complexity,
        plies=[
            PlyGroup(name=p.name, material=p.prepreg, area_m2=p.area_m2, count=p.count)
            for p in plies
        ],
        geometry=PartGeometry(
            part_area_m2=values["part_area_m2"],
            part_perimeter_m=values["part_perimeter_m"],
            hole_count=int(values["hole_count"]),
            tool_area_m2=values["tool_area_m2"],
            tool_perimeter_m=values["tool_perimeter_m"],
            tool_footprint_x_mm=values["tool_footprint_x_mm"],
            tool_footprint_y_mm=values["tool_footprint_y_mm"],
            source=source,
        ),
        quantity=quantity,
        batch_size=batch_size,
        tooling_nre=nre(tool_cost),
        master_nre=nre(master_cost),
        wage_factor=wage_factor,
        wage_factor_reason=wage_factor_reason,
        review_notes=notes,
    )
    return CatalogQuote(input=inp)
