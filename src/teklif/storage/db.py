"""SQLite storage: quotes (with full input/result JSON), override log, actuals."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import ForeignKey, String, UniqueConstraint, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from teklif.costing.engine import QuoteResult
from teklif.library.loader import Library, library_hash


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class QuoteRow(Base):
    __tablename__ = "quotes"
    __table_args__ = (UniqueConstraint("quote_no", "revision"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    quote_no: Mapped[str] = mapped_column(String(64), index=True)
    revision: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=_now)
    part_no: Mapped[str] = mapped_column(String(128))
    customer: Mapped[str] = mapped_column(String(256), default="")
    currency: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int]
    unit_cost: Mapped[float | None]
    unit_price: Mapped[float | None]
    order_total: Mapped[float | None]
    manual_review: Mapped[bool]
    library_hash: Mapped[str] = mapped_column(String(64))
    result_json: Mapped[str]

    overrides: Mapped[list[OverrideRow]] = relationship(back_populates="quote")
    actuals: Mapped[list[ActualRow]] = relationship(back_populates="quote")

    def result(self) -> QuoteResult:
        return QuoteResult.model_validate_json(self.result_json)


class OverrideRow(Base):
    __tablename__ = "overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    quote_id: Mapped[int] = mapped_column(ForeignKey("quotes.id"))
    line_key: Mapped[str] = mapped_column(String(128))
    computed: Mapped[float | None]
    value: Mapped[float]
    reason: Mapped[str]
    user: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_now)

    quote: Mapped[QuoteRow] = relationship(back_populates="overrides")


class ActualRow(Base):
    """Realised cost of a quoted job (per part, quote currency)."""

    __tablename__ = "actuals"

    id: Mapped[int] = mapped_column(primary_key=True)
    quote_id: Mapped[int] = mapped_column(ForeignKey("quotes.id"))
    recorded_at: Mapped[datetime] = mapped_column(default=_now)
    unit_cost: Mapped[float]
    material: Mapped[float | None]
    consumable: Mapped[float | None]
    labor: Mapped[float | None]
    labor_hours: Mapped[float | None]
    tooling_nre: Mapped[float | None]
    note: Mapped[str] = mapped_column(default="")

    quote: Mapped[QuoteRow] = relationship(back_populates="actuals")


class Store:
    def __init__(self, path: Path | str) -> None:
        self.engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self.engine)

    def save_quote(self, result: QuoteResult, lib: Library) -> QuoteRow:
        """Store a new revision of the quote, with its override log."""
        inp = result.input
        with Session(self.engine, expire_on_commit=False) as s:
            last = s.scalar(
                select(func.max(QuoteRow.revision)).where(QuoteRow.quote_no == inp.quote_no)
            )
            row = QuoteRow(
                quote_no=inp.quote_no,
                revision=(last or 0) + 1,
                part_no=inp.part_no,
                customer=inp.customer,
                currency=result.currency,
                quantity=inp.quantity,
                unit_cost=result.value("cost.unit"),
                unit_price=result.value("price.unit_final"),
                order_total=result.value("price.order_total"),
                manual_review=result.manual_review,
                library_hash=library_hash(lib),
                result_json=result.model_dump_json(),
            )
            for line in result.lines:
                if line.override is not None:
                    row.overrides.append(
                        OverrideRow(
                            line_key=line.key,
                            computed=line.result,
                            value=line.override.value,
                            reason=line.override.reason,
                            user=line.override.user,
                        )
                    )
            s.add(row)
            s.commit()
            return row

    def get(self, quote_id: int) -> QuoteRow:
        with Session(self.engine, expire_on_commit=False) as s:
            row = s.get(QuoteRow, quote_id)
            if row is None:
                raise KeyError(quote_id)
            return row

    def list_quotes(self) -> list[QuoteRow]:
        with Session(self.engine, expire_on_commit=False) as s:
            return list(s.scalars(select(QuoteRow).order_by(QuoteRow.id.desc())))

    def overrides(self, quote_id: int) -> list[OverrideRow]:
        with Session(self.engine, expire_on_commit=False) as s:
            return list(s.scalars(select(OverrideRow).where(OverrideRow.quote_id == quote_id)))

    def add_actual(
        self, quote_id: int, unit_cost: float, **fields: float | str | None
    ) -> ActualRow:
        with Session(self.engine, expire_on_commit=False) as s:
            if s.get(QuoteRow, quote_id) is None:
                raise KeyError(quote_id)
            row = ActualRow(quote_id=quote_id, unit_cost=unit_cost, **fields)
            s.add(row)
            s.commit()
            return row

    def actuals(self, quote_id: int) -> list[ActualRow]:
        with Session(self.engine, expire_on_commit=False) as s:
            return list(s.scalars(select(ActualRow).where(ActualRow.quote_id == quote_id)))
