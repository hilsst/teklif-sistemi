"""Core traceability models shared by every layer.

Every number the system uses is a ``Param`` (or ``Price``) that carries its unit and
source. Every computed cost is a ``CostLine`` that carries its formula, its inputs
and its result. A ``None`` value means "not yet provided by the company" (TODO);
using such a value never produces a silent number, it produces a ``ReviewFlag``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
    """A single traceable item: formula, inputs, sources and result.

    ``unit`` is a currency code for money lines (``EUR``) or a unit for quantity
    lines such as operation times (``min``).
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
    def final(self) -> float | None:
        """Overridden value if present, otherwise the computed result."""
        return self.override.value if self.override is not None else self.result


LibValue = Param | Price


def derived(value: float | None, unit: str, desc: str | None = None) -> Param:
    """A value computed from other values (e.g. a driver total), as a ``Param``."""
    return Param(value=value, unit=unit, source=Source.DERIVED, desc=desc)


def to_input(name: str, value: LibValue, path: str | None = None) -> CostInput:
    """Convert a library value into a ``CostInput`` for display and audit."""
    source = Source.TODO if value.is_todo else value.source
    return CostInput(name=name, value=value.value, unit=value.unit, source=source, path=path)


def missing_value_flag(name: str, path: str | None = None) -> ReviewFlag:
    return ReviewFlag(
        code="missing_value",
        reason=f"'{name}' has no company value yet (TODO); result cannot be computed.",
        path=path,
    )


def formula_line(
    key: str,
    description: str,
    formula: str,
    inputs: Mapping[str, LibValue],
    compute: Callable[[Mapping[str, float]], float],
    unit: str,
    category: str = "",
    paths: Mapping[str, str] | None = None,
    flags: Sequence[ReviewFlag] = (),
) -> CostLine:
    """Evaluate ``compute`` on the input values and record everything.

    Any TODO input makes the result ``None`` and adds a ``missing_value`` flag, so
    an unfilled company value can never leak a fabricated number into a quote.
    ``flags`` are extra review flags raised by the caller (e.g. a failed check).
    """
    paths = paths or {}
    cost_inputs = [to_input(name, v, paths.get(name)) for name, v in inputs.items()]
    all_flags = list(flags)
    for i in cost_inputs:
        if i.value is None:
            all_flags.append(
                ReviewFlag(code="missing_upstream", reason=f"'{i.name}' could not be computed")
                if i.source is Source.DERIVED
                else missing_value_flag(i.name, i.path)
            )
    result: float | None = None
    if all(i.value is not None for i in cost_inputs):
        values = {i.name: i.value for i in cost_inputs if i.value is not None}
        try:
            result = compute(values)
        except (ZeroDivisionError, ValueError) as exc:
            all_flags.append(ReviewFlag(code="invalid_input", reason=f"{formula}: {exc}"))
        else:
            if not math.isfinite(result):
                all_flags.append(ReviewFlag(code="invalid_input", reason=f"{formula}: {result}"))
                result = None
    return CostLine(
        key=key,
        description=description,
        formula=formula,
        inputs=cost_inputs,
        result=result,
        unit=unit,
        category=category,
        flags=all_flags,
    )


def product_cost_line(
    key: str,
    description: str,
    factors: Mapping[str, LibValue],
    unit: str,
    paths: Mapping[str, str] | None = None,
    category: str = "",
) -> CostLine:
    """Pure formula ``result = product(factors)``."""
    return formula_line(
        key,
        description,
        " × ".join(factors),
        factors,
        lambda v: math.prod(v.values()),
        unit,
        category,
        paths,
    )


def sum_line(
    key: str, description: str, lines: Sequence[CostLine], unit: str, category: str = ""
) -> CostLine:
    """Sum of other lines' final values (overrides respected).

    If any part is missing the sum is ``None`` with a ``missing_upstream`` flag; the
    upstream lines keep their own flags, so the reason stays visible there.
    """
    inputs = [
        CostInput(name=ln.key, value=ln.final, unit=ln.unit, source=Source.DERIVED) for ln in lines
    ]
    missing = [i.name for i in inputs if i.value is None]
    flags = []
    result: float | None = None
    if missing:
        flags.append(ReviewFlag(code="missing_upstream", reason="Missing: " + ", ".join(missing)))
    else:
        result = math.fsum(i.value for i in inputs if i.value is not None)
    return CostLine(
        key=key,
        description=description,
        formula="Σ " + (", ".join(ln.key for ln in lines) if lines else "∅"),
        inputs=inputs,
        result=result,
        unit=unit,
        category=category,
        flags=flags,
    )
