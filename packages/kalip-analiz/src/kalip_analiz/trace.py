"""Traceability models: every number carries its formula, inputs, sources and result.

``CostLine`` is the generic traceable line (the name comes from the quoting system this package
was split from; here it holds decision scores and ratios as well as money). ``ReviewFlag`` is a
``manual_review`` marker: the analysis is not sure and says why.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Source(StrEnum):
    """Where a value came from."""

    CAD = "cad"
    DRAWING = "drawing"
    LIBRARY = "library"
    USER = "user"
    DEFAULT = "default"
    DERIVED = "derived"
    TODO = "todo"


class ReviewFlag(BaseModel):
    """A ``manual_review`` marker: the system is not sure and says why."""

    model_config = ConfigDict(frozen=True)

    code: str
    reason: str
    path: str | None = None


class CostInput(BaseModel):
    """One input of a formula, as shown in the breakdown."""

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

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("An override needs a non-empty reason")
        return v.strip()


DECIDED_BY_OVERRIDE = frozenset({"missing_value", "missing_upstream", "nre_missing"})


class CostLine(BaseModel):
    """A single traceable item: formula, inputs, sources and result.

    ``unit`` is a currency code for money lines (``EUR``) or a unit for quantity
    lines such as operation times (``min``) or decision scores (``ratio``).
    """

    key: str
    description: str
    formula: str
    inputs: list[CostInput]
    result: float | None
    unit: str
    category: str = ""
    flags: list[ReviewFlag] = Field(default_factory=list)
    override: Override | None = None

    @property
    def manual_review(self) -> bool:
        return bool(self.flags)

    @property
    def open_flags(self) -> list[ReviewFlag]:
        """Flags still needing a decision.

        An override (with a reason) decides a *missing value* flag. Other flags (does not fit,
        over temperature, invalid input...) describe a real problem and stay open.
        """
        if self.override is None:
            return self.flags
        return [fl for fl in self.flags if fl.code not in DECIDED_BY_OVERRIDE]

    @property
    def final(self) -> float | None:
        """Overridden value if present, otherwise the computed result."""
        return self.override.value if self.override is not None else self.result


TraceLine = CostLine
