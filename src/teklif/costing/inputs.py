"""Quote input: everything the user (or, from Phase 2, the CAD layer) provides."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from teklif.common.models import Override, Price, Source


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlyGroup(_Strict):
    """Plies sharing the same material and flat-pattern area."""

    name: str = ""
    material: str | None = Field(None, description="Prepreg id; defaults to the part material")
    area_m2: float = Field(gt=0, description="Flat-pattern area of one ply, scrap excluded")
    count: int = Field(1, ge=1)


class PartGeometry(_Strict):
    part_area_m2: float = Field(gt=0, description="Part surface area, one side")
    part_perimeter_m: float = Field(ge=0, description="Trim edge length")
    hole_count: int = Field(0, ge=0)
    tool_area_m2: float = Field(gt=0, description="Bagging area incl. flange and run-off")
    tool_perimeter_m: float = Field(gt=0, description="Bagging perimeter (sealant line)")
    tool_footprint_x_mm: float = Field(gt=0, description="Tool footprint in the autoclave, X")
    tool_footprint_y_mm: float = Field(gt=0, description="Tool footprint in the autoclave, Y")
    source: Source = Source.USER


class QuoteInput(_Strict):
    quote_no: str
    part_no: str
    part_name: str = ""
    customer: str = ""
    process: str
    material: str = Field(description="Default prepreg id for ply groups")
    cure_cycle: str
    autoclave: str | None = Field(None, description="Autoclave id; defaults to the first one")
    complexity: str
    plies: list[PlyGroup] = Field(min_length=1)
    geometry: PartGeometry
    quantity: int = Field(ge=1, description="Order quantity")
    batch_size: int = Field(ge=1, description="Parts per production batch (setup split)")
    first_unit_no: int = Field(
        1, ge=1, description="Cumulative unit number of the first part (repeat orders)"
    )
    include_optional: list[str] = Field(default_factory=list)
    exclude_operations: list[str] = Field(default_factory=list)
    tooling_nre: Price | None = Field(None, description="Tool + master model price (Phase 1)")
    tooling_exists: bool = Field(False, description="Tool already exists, no NRE")
    amortize_nre: bool = False
    quote_currency: str | None = None
    overrides: dict[str, Override] = Field(default_factory=dict)
