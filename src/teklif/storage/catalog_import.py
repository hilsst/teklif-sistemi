"""Load validated history workbooks (``kalip_gecmis`` / ``iscilik_gecmis``) into the catalog.

Rows come from ``history_import.read_workbook`` (already validated). Existing records are
never overwritten; a row that would collide is reported as skipped. All imported values
carry ``source=user``: they are the company's own historical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from teklif.common.models import Source
from teklif.library import Library
from teklif.storage.catalog import Catalog, CatalogError, MasterModel, Part, Tool
from teklif.storage.history_import import Value

Row = dict[str, Value]


@dataclass
class ImportSummary:
    created: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def count(self, what: str) -> None:
        self.created[what] = self.created.get(what, 0) + 1


def _find_part(cat: Catalog, part_no: str) -> Part | None:
    with Session(cat.engine) as s:
        return s.scalars(
            select(Part).where(Part.part_no == part_no).order_by(Part.id.desc())
        ).first()


def _ensure_part(cat: Catalog, row: Row, summary: ImportSummary) -> Part:
    part_no = str(row["parca_no"])
    part = _find_part(cat, part_no)
    if part is None:
        part = cat.add(
            Part(
                part_no=part_no,
                name=str(row.get("parca_adi") or ""),
                process=str(row.get("proses") or ""),
                complexity=str(row.get("karmasiklik") or ""),
            )
        )
        summary.count("parça")
    return part


def _set_if_absent(
    cat: Catalog, entity: str, entity_id: int, key: str, value: Value, unit: str
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    if key in cat.params(entity, entity_id):
        return
    cat.set_param(entity, entity_id, key, float(value), unit, Source.USER)


def import_tool_history(cat: Catalog, rows: list[Row]) -> ImportSummary:
    summary = ImportSummary()
    for row in rows:
        tool_no = str(row["kalip_no"])
        with Session(cat.engine) as s:
            exists = s.scalar(select(Tool).where(Tool.tool_no == tool_no)) is not None
        if exists:
            summary.skipped.append(f"{tool_no}: kalıp zaten kayıtlı")
            continue
        master_kind = str(row.get("master_model") or "yok")
        master_no = f"{tool_no}-M"
        with Session(cat.engine) as s:
            master_taken = s.scalar(select(MasterModel).where(MasterModel.model_no == master_no))
        if master_kind != "yok" and master_taken is not None:
            summary.skipped.append(f"{tool_no}: master model no {master_no} zaten kayıtlı")
            continue
        part = _ensure_part(cat, row, summary)
        master_id: int | None = None
        if master_kind != "yok":
            master = cat.add(
                MasterModel(
                    model_no=master_no,
                    material=master_kind,
                    note="Maliyeti kalıp toplam maliyetine dahildir (geçmiş kalıp tablosu).",
                )
            )
            master_id = master.id
            summary.count("master model")
        tool = cat.add(
            Tool(
                tool_no=tool_no,
                tool_type=str(row.get("kalip_tipi") or ""),
                material_class=str(row.get("kalip_cinsi") or ""),
                material_grade=str(row.get("kalip_malzemesi") or ""),
                body_parts=int(row.get("kalip_parca_sayisi") or 1),  # type: ignore[arg-type]
                loose_cores=int(row.get("maca_sayisi") or 0),  # type: ignore[arg-type]
                undercut_solution=str(row.get("ters_aci_cozumu") or ""),
                target_life=int(row["hedef_kalip_omru"]) if row.get("hedef_kalip_omru") else None,  # type: ignore[arg-type]
                has_substructure=row.get("alt_yapi") == "E",
                master_model_id=master_id,
                note=str(row.get("notlar") or ""),
            )
        )
        summary.count("kalıp")
        cat.link_tool_part(tool.id, part.id)
        for axis in ("x", "y", "z"):
            _set_if_absent(
                cat, "tool", tool.id, f"envelope_{axis}_mm", row.get(f"kalip_zarf_{axis}"), "mm"
            )
        for axis in ("x", "y"):
            _set_if_absent(
                cat,
                "tool",
                tool.id,
                f"tool_footprint_{axis}_mm",
                row.get(f"kalip_zarf_{axis}"),
                "mm",
            )
        _set_if_absent(cat, "part", part.id, "part_area_m2", row.get("parca_alani"), "m2")
        for axis in ("x", "y", "z"):
            _set_if_absent(
                cat, "part", part.id, f"envelope_{axis}_mm", row.get(f"parca_zarf_{axis}"), "mm"
            )
        total = row.get("fiili_toplam_maliyet")
        if isinstance(total, int | float) and total > 0:
            materials = [
                v
                for v in (
                    row.get("kalip_malzeme_maliyeti"),
                    row.get("master_model_malzeme_maliyeti"),
                )
                if isinstance(v, int | float)
            ]
            material = float(sum(materials)) if materials else None
            if material is not None and material > total:
                material = None
                summary.skipped.append(
                    f"{tool_no}: malzeme toplamı maliyetten büyük, kırılım atlandı"
                )
            when = row.get("tarih")
            cat.add_cost(
                "tool",
                tool.id,
                float(total),
                str(row["para_birimi"]),
                material=material,
                cost_date=when if isinstance(when, date) else None,
                note="geçmiş kalıp tablosu; master model dahil",
                includes_master=True,
            )
            summary.count("maliyet kaydı")
    return summary


def import_labor_history(cat: Catalog, rows: list[Row], lib: Library) -> ImportSummary:
    """Each row becomes an ``actual`` operation; times are converted to clock minutes per part."""
    summary = ImportSummary()
    for row in rows:
        part = _ensure_part(cat, row, summary)
        op_id = str(row["operasyon"])
        order_note = f"iş emri {row['is_emri_no']}"
        if any(o.operation == op_id and o.note == order_note for o in cat.operations(part.id)):
            summary.skipped.append(f"{row['is_emri_no']}/{op_id}: zaten aktarılmış")
            continue
        if not row.get("operator_sayisi"):
            summary.skipped.append(
                f"{row['is_emri_no']}/{op_id}: kişi sayısı yok, parça başı dakikaya çevrilemez"
            )
            continue
        persons = float(row["operator_sayisi"])  # type: ignore[arg-type]
        batch = int(row["parti_adedi"])  # type: ignore[arg-type]
        if batch <= 0 or persons <= 0:
            summary.skipped.append(
                f"{row['is_emri_no']}/{op_id}: parti adedi ve kişi sayısı > 0 olmalı"
            )
            continue
        setup_total = float(row.get("hazirlik_suresi") or 0)  # type: ignore[arg-type]
        run_total = float(row["fiili_iscilik"]) - setup_total  # type: ignore[arg-type]
        if run_total <= 0:
            summary.skipped.append(f"{row['is_emri_no']}/{op_id}: hazırlık toplamdan büyük")
            continue
        tpl = lib.processes.get(str(row["proses"]))
        seq = next((o.seq for o in (tpl.operations if tpl else []) if o.id == op_id), 0)
        when = row.get("tarih")
        try:
            cat.add_operation(
                part.id,
                "actual",
                seq=seq,
                operation=op_id,
                work_center=str(row["is_merkezi"]),
                persons=persons,
                setup_min=setup_total / persons,
                run_min_per_part=run_total / persons / batch,
                batch_size=batch,
                done_on=when if isinstance(when, date) else None,
                note=order_note,
            )
        except CatalogError as exc:
            summary.skipped.append(f"{row['is_emri_no']}/{op_id}: {exc}")
            continue
        summary.count("operasyon")
        for key, col, unit in (
            ("part_area_m2", "parca_alani", "m2"),
            ("part_perimeter_m", "trim_cevresi", "m"),
            ("hole_count", "delik_sayisi", "hole"),
        ):
            _set_if_absent(cat, "part", part.id, key, row.get(col), unit)
    return summary
