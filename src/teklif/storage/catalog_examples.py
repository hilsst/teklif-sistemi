"""Load part/model/tool cost examples from a CSV plus a folder of part STEP files.

CSV columns: ``part_no, step_token, model_cost, tool_cost, currency``. The STEP file is the one
in the folder whose name contains ``step_token`` (missing files are reported, not guessed).
Existing parts are never overwritten. Measured values are derived and flagged for review; costs
are the company's own numbers (source user, kept in cost records).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from teklif.geometry.plate_features import extract_plate_features, plate_step_values
from teklif.ingest.step_props import (
    EXTRACTOR_VERSION,
    StepUnavailableError,
    extract_step_props,
)
from teklif.storage.catalog import Catalog, CatalogError, MasterModel, Part, Tool


@dataclass
class ExampleSummary:
    parts: int = 0
    notes: list[str] = field(default_factory=list)


def _float(text: str) -> float | None:
    text = text.strip()
    return float(text) if text else None


def load_examples(cat: Catalog, csv_path: Path, step_dir: Path) -> ExampleSummary:
    out = ExampleSummary()
    steps = sorted(step_dir.glob("*.st*p")) if step_dir.is_dir() else []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        part_no = r["part_no"].strip()
        with Session(cat.engine) as s:
            if s.scalar(select(Part).where(Part.part_no == part_no, Part.rev == "A")):
                out.notes.append(f"{part_no}: already in catalog, skipped")
                continue
        token = r["step_token"].strip().upper().replace(" ", "_")
        match = next((p for p in steps if token in p.name.upper().replace(" ", "_")), None)
        cad_id: int | None = None
        values = []
        if match is None:
            out.notes.append(f"{part_no}: no STEP file containing '{token}', measurements missing")
        else:
            cad, _ = cat.add_cad_file(match.read_bytes(), match.name, "part")
            cad_id = cad.id
            try:
                props = extract_step_props(match)
                values = props.values + plate_step_values(extract_plate_features(match))
                cat.record_values("cad_file", cad.id, values)
                cat.mark_measured(cad.id, EXTRACTOR_VERSION)
            except StepUnavailableError as exc:
                out.notes.append(f"{part_no}: not measured ({exc})")
        part = cat.add(Part(part_no=part_no, rev="A", name=part_no, cad_file_id=cad_id))
        if values:
            cat.record_values("part", part.id, values)
        currency = r["currency"].strip() or "EUR"
        model_cost = _float(r["model_cost"])
        tool_cost = _float(r["tool_cost"])
        try:
            model = None
            if model_cost is not None:
                model = cat.add(MasterModel(model_no=f"{part_no} MODEL"))
                cat.add_cost("master_model", model.id, model_cost, currency)
            if tool_cost is not None:
                tool = cat.add(
                    Tool(tool_no=f"{part_no} KALIP", master_model_id=model.id if model else None)
                )
                cat.add_cost("tool", tool.id, tool_cost, currency)
                cat.link_tool_part(tool.id, part.id)
        except CatalogError as exc:
            out.notes.append(f"{part_no}: {exc}")
        out.parts += 1
    return out
