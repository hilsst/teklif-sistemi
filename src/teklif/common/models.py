"""Core traceability models shared by every layer.

Every number the system uses is a ``Param`` (or ``Price``) that carries its unit and
source. Every computed cost is a ``CostLine`` that carries its formula, its inputs
and its result. A ``None`` value means "not yet provided by the company" (TODO);
using such a value never produces a silent number, it produces a ``ReviewFlag``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Source(StrEnum):
    """Where a value came from."""

    CAD = "cad"
    DRAWING = "drawing"
    LIBRARY = "library"
    USER = "user"
    DEFAULT = "default"
    TODO = "todo"


class ReviewFlag(BaseModel):
    """A ``manual_review`` marker: the system is not sure and says why."""

    model_config = ConfigDict(frozen=True)

    code: str
    reason: str
    path: str | None = None


class Param(BaseModel):
    """A numeric parameter with unit and provenance. ``value=None`` means TODO."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    unit: str
    source: Source = Source.LIBRARY
    desc: str | None = None
    note: str | None = None

    @property
    def is_todo(self) -> bool:
        return self.value is None


class Price(BaseModel):
    """A price per unit quantity in an explicit currency, e.g. 85 EUR per m2."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    currency: str
    per: str
    source: Source = Source.LIBRARY
    desc: str | None = None
    note: str | None = None

    @property
    def is_todo(self) -> bool:
        return self.value is None

    @property
    def unit(self) -> str:
        return f"{self.currency}/{self.per}"


class CostInput(BaseModel):
    """One input of a cost formula, as shown in the quote breakdown."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float | None
    unit: str
    source: Source
    path: str | None = None


class Override(BaseModel):
    """A user override of a computed result, stored with its justification."""

    model_config = ConfigDict(frozen=True)

    value: float
    reason: str
    user: str | None = None


class CostLine(BaseModel):
    """A single traceable cost item: formula, inputs, sources and result."""

    key: str
    description: str
    formula: str
    inputs: list[CostInput]
    result: float | None
    currency: str
    flags: list[ReviewFlag] = Field(default_factory=list)
    override: Override | None = None

    @property
    def manual_review(self) -> bool:
        return bool(self.flags)

    @property
    def final(self) -> float | None:
        """Overridden value if present, otherwise the computed result."""
        return self.override.value if self.override is not None else self.result


def to_input(name: str, value: Param | Price, path: str | None = None) -> CostInput:
    """Convert a library value into a ``CostInput`` for display and audit."""
    source = Source.TODO if value.is_todo else value.source
    return CostInput(name=name, value=value.value, unit=value.unit, source=source, path=path)


def missing_value_flag(name: str, path: str | None = None) -> ReviewFlag:
    return ReviewFlag(
        code="missing_value",
        reason=f"'{name}' has no company value yet (TODO); result cannot be computed.",
        path=path,
    )


def product_cost_line(
    key: str,
    description: str,
    factors: dict[str, Param | Price],
    currency: str,
    paths: dict[str, str] | None = None,
) -> CostLine:
    """Pure formula ``result = product(factors)``.

    Any TODO factor makes the result ``None`` and adds a ``missing_value`` flag, so
    an unfilled company value can never leak a fabricated number into a quote.
    """
    paths = paths or {}
    inputs = [to_input(name, v, paths.get(name)) for name, v in factors.items()]
    flags = [missing_value_flag(i.name, i.path) for i in inputs if i.value is None]
    result: float | None = None
    if not flags:
        result = 1.0
        for i in inputs:
            assert i.value is not None
            result *= i.value
    return CostLine(
        key=key,
        description=description,
        formula=" × ".join(factors),
        inputs=inputs,
        result=result,
        currency=currency,
        flags=flags,
    )
