"""Data-entry catalog: CAD files, master models, tools, parts, operations and costs.

This is the company's own data, entered through the UI (or the Excel history upload) and
used both as quote input and as the reference set for calibrating the library. Every
number lives in ``param_values`` together with its source (``cad`` / ``derived`` / ``user``),
so a CAD-measured value can be overridden, but only with a reason, and the history is kept.

Money: tool and master model prices are the company's *cost* (not a sales price). Only the
total is required; the material share is optional and the remainder is derived.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import ForeignKey, String, UniqueConstraint, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from teklif.common.models import Source
from teklif.ingest.step_props import StepValue

CAD_KINDS = ("part", "tool", "master_model", "assembly")
ENTITIES = ("cad_file", "master_model", "tool", "part")
OPERATION_KINDS = ("actual", "estimate")
# Sources whose value may only be replaced by the user together with a reason.
_PROTECTED_SOURCES = {Source.CAD.value, Source.DERIVED.value}


def _now() -> datetime:
    return datetime.now(UTC)


class CatalogError(ValueError):
    """Invalid catalog entry (missing reason, unknown kind, duplicate id...)."""


class Base(DeclarativeBase):
    pass


class CadFile(Base):
    __tablename__ = "cad_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(256))
    kind: Mapped[str] = mapped_column(String(16))
    stored_path: Mapped[str]
    extractor_version: Mapped[str | None]
    uploaded_at: Mapped[datetime] = mapped_column(default=_now)


class MasterModel(Base):
    __tablename__ = "master_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_no: Mapped[str] = mapped_column(String(64), unique=True)
    material: Mapped[str] = mapped_column(String(64), default="")
    material_grade: Mapped[str] = mapped_column(String(128), default="")
    cad_file_id: Mapped[int | None] = mapped_column(ForeignKey("cad_files.id"))
    note: Mapped[str] = mapped_column(default="")


class Tool(Base):
    __tablename__ = "tools"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_no: Mapped[str] = mapped_column(String(64), unique=True)
    tool_type: Mapped[str] = mapped_column(String(32), default="")
    material_class: Mapped[str] = mapped_column(String(32), default="")
    material_grade: Mapped[str] = mapped_column(String(128), default="")
    body_parts: Mapped[int] = mapped_column(default=1)
    loose_cores: Mapped[int] = mapped_column(default=0)
    undercut_solution: Mapped[str] = mapped_column(String(32), default="")
    target_life: Mapped[int | None]
    has_substructure: Mapped[bool] = mapped_column(default=False)
    cad_file_id: Mapped[int | None] = mapped_column(ForeignKey("cad_files.id"))
    master_model_id: Mapped[int | None] = mapped_column(ForeignKey("master_models.id"))
    note: Mapped[str] = mapped_column(default="")


class Part(Base):
    __tablename__ = "parts"
    __table_args__ = (UniqueConstraint("part_no", "rev"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    part_no: Mapped[str] = mapped_column(String(128))
    rev: Mapped[str] = mapped_column(String(16), default="A")
    name: Mapped[str] = mapped_column(String(256), default="")
    customer: Mapped[str] = mapped_column(String(256), default="")
    process: Mapped[str] = mapped_column(String(64), default="")
    prepreg: Mapped[str] = mapped_column(String(64), default="")
    cure_cycle: Mapped[str] = mapped_column(String(64), default="")
    complexity: Mapped[str] = mapped_column(String(16), default="")
    cad_file_id: Mapped[int | None] = mapped_column(ForeignKey("cad_files.id"))


class ToolPart(Base):
    """A tool can serve several parts (family tool, left/right) and vice versa."""

    __tablename__ = "tool_parts"
    __table_args__ = (UniqueConstraint("tool_id", "part_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tools.id"))
    part_id: Mapped[int] = mapped_column(ForeignKey("parts.id"))


class PartPly(Base):
    __tablename__ = "part_plies"

    id: Mapped[int] = mapped_column(primary_key=True)
    part_id: Mapped[int] = mapped_column(ForeignKey("parts.id"))
    name: Mapped[str] = mapped_column(String(128), default="")
    prepreg: Mapped[str | None] = mapped_column(String(64))
    area_m2: Mapped[float]
    count: Mapped[int] = mapped_column(default=1)


class PartOperation(Base):
    """One operation of a part with its time. ``kind`` is actual (measured) or estimate."""

    __tablename__ = "part_operations"

    id: Mapped[int] = mapped_column(primary_key=True)
    part_id: Mapped[int] = mapped_column(ForeignKey("parts.id"))
    seq: Mapped[int]
    operation: Mapped[str] = mapped_column(String(64))
    work_center: Mapped[str] = mapped_column(String(64))
    persons: Mapped[float] = mapped_column(default=1.0)
    setup_min: Mapped[float] = mapped_column(default=0.0)
    run_min_per_part: Mapped[float]
    batch_size: Mapped[int] = mapped_column(default=1)
    kind: Mapped[str] = mapped_column(String(16))
    done_on: Mapped[date | None]
    note: Mapped[str] = mapped_column(default="")


class CostRecord(Base):
    """Company cost of a tool or master model. Total required, material share optional."""

    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity: Mapped[str] = mapped_column(String(16))
    entity_id: Mapped[int]
    total: Mapped[float]
    material: Mapped[float | None]
    currency: Mapped[str] = mapped_column(String(8))
    cost_date: Mapped[date | None]
    note: Mapped[str] = mapped_column(default="")

    @property
    def other(self) -> float | None:
        """Everything that is not material (labor, CNC, consumables), derived."""
        return None if self.material is None else self.total - self.material


class ParamValue(Base):
    """Append-only numeric value with unit and source; the newest row per key is current."""

    __tablename__ = "param_values"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity: Mapped[str] = mapped_column(String(16), index=True)
    entity_id: Mapped[int] = mapped_column(index=True)
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[float | None]
    unit: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))
    review_reason: Mapped[str | None]
    override_reason: Mapped[str | None]
    user: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class Catalog:
    def __init__(self, path: Path | str, cad_dir: Path | str | None = None) -> None:
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)
        self.cad_dir = Path(cad_dir) if cad_dir else Path(str(path)).parent / "cad"

    def _session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    # --- CAD files -------------------------------------------------------------

    def add_cad_file(self, data: bytes, name: str, kind: str) -> tuple[CadFile, bool]:
        """Store a CAD file by content hash. Returns ``(row, created)``."""
        if kind not in CAD_KINDS:
            raise CatalogError(f"unknown cad kind '{kind}'")
        digest = hashlib.sha256(data).hexdigest()
        with self._session() as s:
            existing = s.scalar(select(CadFile).where(CadFile.sha256 == digest))
            if existing is not None:
                return existing, False
            self.cad_dir.mkdir(parents=True, exist_ok=True)
            stored = self.cad_dir / f"{digest}{Path(name).suffix.lower()}"
            stored.write_bytes(data)
            row = CadFile(sha256=digest, name=name, kind=kind, stored_path=str(stored))
            s.add(row)
            s.commit()
            return row, True

    # --- cards -----------------------------------------------------------------

    def mark_measured(self, cad_file_id: int, version: str) -> None:
        with self._session() as s:
            row = s.get(CadFile, cad_file_id)
            if row is not None:
                row.extractor_version = version
                s.commit()

    def add[T: (MasterModel, Tool, Part)](self, row: T) -> T:
        with self._session() as s:
            s.add(row)
            try:
                s.commit()
            except IntegrityError as exc:
                raise CatalogError("a record with this number already exists") from exc
            return row

    def link_tool_part(self, tool_id: int, part_id: int) -> None:
        with self._session() as s:
            if s.scalar(
                select(ToolPart).where(ToolPart.tool_id == tool_id, ToolPart.part_id == part_id)
            ):
                return
            s.add(ToolPart(tool_id=tool_id, part_id=part_id))
            s.commit()

    def tools_for_part(self, part_id: int) -> list[Tool]:
        with self._session() as s:
            q = select(Tool).join(ToolPart, ToolPart.tool_id == Tool.id)
            return list(s.scalars(q.where(ToolPart.part_id == part_id)))

    def add_ply(
        self, part_id: int, area_m2: float, count: int = 1, **fields: str | None
    ) -> PartPly:
        if area_m2 <= 0 or count < 1:
            raise CatalogError("ply area must be > 0 and count >= 1")
        with self._session() as s:
            row = PartPly(part_id=part_id, area_m2=area_m2, count=count, **fields)
            s.add(row)
            s.commit()
            return row

    def plies(self, part_id: int) -> list[PartPly]:
        with self._session() as s:
            return list(s.scalars(select(PartPly).where(PartPly.part_id == part_id)))

    def add_operation(
        self, part_id: int, kind: str, **fields: str | float | int | date | None
    ) -> PartOperation:
        if kind not in OPERATION_KINDS:
            raise CatalogError(f"operation kind must be one of {OPERATION_KINDS}")
        with self._session() as s:
            row = PartOperation(part_id=part_id, kind=kind, **fields)
            s.add(row)
            s.commit()
            return row

    def operations(self, part_id: int, kind: str | None = None) -> list[PartOperation]:
        with self._session() as s:
            q = select(PartOperation).where(PartOperation.part_id == part_id)
            if kind is not None:
                q = q.where(PartOperation.kind == kind)
            return list(s.scalars(q.order_by(PartOperation.seq)))

    def add_cost(
        self,
        entity: str,
        entity_id: int,
        total: float,
        currency: str,
        material: float | None = None,
        cost_date: date | None = None,
        note: str = "",
    ) -> CostRecord:
        if entity not in ("tool", "master_model"):
            raise CatalogError("costs are recorded for tools and master models")
        if total <= 0 or (material is not None and not 0 <= material <= total):
            raise CatalogError("material must be between 0 and the total cost")
        with self._session() as s:
            row = CostRecord(
                entity=entity,
                entity_id=entity_id,
                total=total,
                material=material,
                currency=currency,
                cost_date=cost_date,
                note=note,
            )
            s.add(row)
            s.commit()
            return row

    def costs(self, entity: str, entity_id: int) -> list[CostRecord]:
        with self._session() as s:
            q = select(CostRecord).where(
                CostRecord.entity == entity, CostRecord.entity_id == entity_id
            )
            return list(s.scalars(q.order_by(CostRecord.id)))

    # --- traceable values ------------------------------------------------------

    def set_param(
        self,
        entity: str,
        entity_id: int,
        key: str,
        value: float | None,
        unit: str,
        source: Source,
        review_reason: str | None = None,
        override_reason: str | None = None,
        user: str | None = None,
    ) -> ParamValue:
        """Record a value. Replacing a CAD/derived value by another source needs a reason."""
        if entity not in ENTITIES:
            raise CatalogError(f"unknown entity '{entity}'")
        current = self.params(entity, entity_id).get(key)
        replaces_protected = (
            current is not None
            and current.source in _PROTECTED_SOURCES
            and source.value != current.source
        )
        if replaces_protected and not (override_reason or "").strip():
            raise CatalogError(
                f"{key}: overriding a {current.source if current else ''} value needs a reason"
            )
        with self._session() as s:
            row = ParamValue(
                entity=entity,
                entity_id=entity_id,
                key=key,
                value=value,
                unit=unit,
                source=source.value,
                review_reason=review_reason,
                override_reason=override_reason if replaces_protected else None,
                user=user,
            )
            s.add(row)
            s.commit()
            return row

    def record_values(self, entity: str, entity_id: int, values: list[StepValue]) -> list[str]:
        """Store CAD/derived values without clobbering a user entry. Returns skipped keys."""
        skipped: list[str] = []
        for v in values:
            current = self.params(entity, entity_id).get(v.key)
            if current is not None and current.source == Source.USER.value:
                skipped.append(v.key)
                continue
            self.set_param(
                entity, entity_id, v.key, v.value, v.unit, v.source, review_reason=v.review_reason
            )
        return skipped

    def params(self, entity: str, entity_id: int) -> dict[str, ParamValue]:
        """Current value per key (the newest row)."""
        with self._session() as s:
            q = select(ParamValue).where(
                ParamValue.entity == entity, ParamValue.entity_id == entity_id
            )
            return {r.key: r for r in s.scalars(q.order_by(ParamValue.id))}

    def param_history(self, entity: str, entity_id: int, key: str) -> list[ParamValue]:
        with self._session() as s:
            q = select(ParamValue).where(
                ParamValue.entity == entity,
                ParamValue.entity_id == entity_id,
                ParamValue.key == key,
            )
            return list(s.scalars(q.order_by(ParamValue.id)))
