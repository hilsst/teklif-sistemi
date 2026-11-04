"""Rough tool cost from the part's one-side area (linear fit of the company's past tool costs).

The fit is only valid inside the calibration area range; outside it the estimate is still
returned but flagged, never silently extrapolated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "library" / "tool_cost_model.yaml"


class AreaRange(BaseModel):
    min: float
    max: float


class ToolCostModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    base: float
    per_m2: float = Field(gt=0)
    valid_area_m2: AreaRange
    valid_for: str
    samples: int
    r_squared: float
    max_residual_ratio: float
    leave_one_out_max_error: float


class ToolCostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float
    currency: str
    low: float
    high: float
    in_range: bool
    note: str


def load_model(path: Path | str = DEFAULT_PATH) -> ToolCostModel:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ToolCostModel.model_validate(data["tool_cost_model"])


def estimate_tool_cost(part_area_m2: float, model: ToolCostModel | None = None) -> ToolCostEstimate:
    if not part_area_m2 > 0:
        raise ValueError("part area must be greater than zero")
    m = model or load_model()
    value = m.base + m.per_m2 * part_area_m2
    band = value * m.leave_one_out_max_error
    in_range = m.valid_area_m2.min <= part_area_m2 <= m.valid_area_m2.max
    note = (
        f"estimate from {m.samples} past tools ({m.valid_for}), "
        f"leave-one-out error up to {m.leave_one_out_max_error:.0%}"
    )
    if not in_range:
        note += (
            f"; area {part_area_m2:.2f} m2 is outside the calibrated range "
            f"{m.valid_area_m2.min}-{m.valid_area_m2.max} m2, treat as unreliable"
        )
    return ToolCostEstimate(
        value=value,
        currency=m.currency,
        low=value - band,
        high=value + band,
        in_range=in_range,
        note=note,
    )
