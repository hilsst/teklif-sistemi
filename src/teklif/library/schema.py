"""Pydantic schemas for the YAML library files under ``library/``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from teklif.common.models import Param, Price


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- currency.yaml ---------------------------------------------------------------


class CurrencyTable(_Strict):
    base_currency: str
    quote_currency: str
    rate_date: str | None = None
    rates: dict[str, Param]


# --- enums.yaml --------------------------------------------------------------------


class EnumItem(_Strict):
    id: str
    label: str


class Enums(_Strict):
    processes: list[EnumItem]
    tool_material_classes: list[EnumItem]
    master_model_materials: list[EnumItem]
    tool_types: list[EnumItem]
    undercut_solutions: list[EnumItem]
    operations: list[EnumItem]
    complexity: list[EnumItem]
    yes_no: list[EnumItem]

    def ids(self, name: str) -> list[str]:
        items: list[EnumItem] = getattr(self, name)
        return [i.id for i in items]


# --- materials.yaml ----------------------------------------------------------------


class Prepreg(_Strict):
    id: str
    name: str
    fiber: Literal["carbon", "glass", "aramid"]
    form: Literal["ud", "woven"]
    resin: str
    cure_temp: Param
    areal_weight: Param
    cured_ply_thickness: Param
    roll_width: Param
    out_life: Param
    price: Price
    nesting_scrap_ratio: Param | None = None


class Consumable(_Strict):
    id: str
    name: str
    max_temp: Param
    price: Price


class ToolingMaterial(_Strict):
    id: str
    name: str
    tool_class: str
    max_service_temp: Param
    cte: Param
    density: Param
    tool_life: Param
    price: Price


class Materials(_Strict):
    prepregs: list[Prepreg]
    consumables: list[Consumable]
    tooling_materials: list[ToolingMaterial]


# --- work_centers.yaml / autoclaves.yaml -------------------------------------------


class WorkCenter(_Strict):
    id: str
    name: str
    labor_rate: Price
    machine_rate: Price
    operators: Param
    efficiency: Param


class WorkCenters(_Strict):
    work_centers: list[WorkCenter]


class Autoclave(_Strict):
    id: str
    name: str
    inner_diameter: Param
    usable_length: Param
    usable_width: Param
    max_temp: Param
    max_pressure: Param
    vacuum_ports: Param
    thermocouples: Param
    hourly_rate: Price
    load_efficiency: Param


class Autoclaves(_Strict):
    autoclaves: list[Autoclave]


# --- thresholds.yaml ---------------------------------------------------------------


class DraftThresholds(_Strict):
    min_ok_deg: Param
    warn_min_deg: Param


class UndercutThresholds(_Strict):
    core_max_area_ratio: Param


class TessellationThresholds(_Strict):
    max_angular_deflection_deg: Param


class CostingDefaults(_Strict):
    nesting_scrap_ratio: Param
    learning_curve_ratio: Param
    learning_reference_unit: Param
    overhead_ratio: Param
    overhead_basis: Literal["direct_cost", "labor"]
    margin_ratio: Param
    margin_basis: Literal["markup", "margin"]
    consumable_waste_ratio: Param
    autoclave_spacing: Param


class AcceptanceThresholds(_Strict):
    cost_tolerance: Param


class Thresholds(_Strict):
    draft: DraftThresholds
    undercut: UndercutThresholds
    tessellation: TessellationThresholds
    costing: CostingDefaults
    acceptance: AcceptanceThresholds
    complexity_factors: dict[str, Param]


# --- processes/*.yaml --------------------------------------------------------------

Driver = Literal[
    "per_part",
    "ply_area_total_m2",
    "ply_count",
    "part_area_m2",
    "part_perimeter_m",
    "hole_count",
    "tool_area_m2",
    "tool_perimeter_m",
    "debulk_count",
]

DRIVER_UNITS: dict[str, str] = {
    "per_part": "part",
    "ply_area_total_m2": "m2",
    "ply_count": "ply",
    "part_area_m2": "m2",
    "part_perimeter_m": "m",
    "hole_count": "hole",
    "tool_area_m2": "m2",
    "tool_perimeter_m": "m",
    "debulk_count": "debulk",
}


class TimeTerm(_Strict):
    driver: Driver
    rate: Param


class ConsumableUse(_Strict):
    material: str
    driver: Driver
    qty: Param


class Operation(_Strict):
    seq: int
    id: str
    name: str
    work_center: str
    optional: bool = False
    apply_complexity: bool = False
    autoclave_cure: bool = False
    learning: bool = True
    setup: Param | None = None
    times: list[TimeTerm] = Field(default_factory=list)
    consumables: list[ConsumableUse] = Field(default_factory=list)


class CureCycle(_Strict):
    id: str
    name: str
    cure_temp: Param
    pressure: Param
    cycle_time: Param


class ProcessTemplate(_Strict):
    id: str
    name: str
    version: int
    parameters: dict[str, Param] = Field(default_factory=dict)
    cure_cycles: list[CureCycle] = Field(default_factory=list)
    operations: list[Operation]
