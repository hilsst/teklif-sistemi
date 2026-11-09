"""Pydantic schemas for the YAML library files under ``library/``."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from teklif.common.models import Param, Price, Source


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


class WageClass(_Strict):
    id: str
    name: str
    multiplier: Param

    @model_validator(mode="after")
    def _multiplier_positive(self) -> WageClass:
        v = self.multiplier.value
        if v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f"wage class {self.id}: multiplier must be a number > 0")
        return self


class WorkCenter(_Strict):
    id: str
    name: str
    wage_class: str | None = None
    labor_rate: Price | None = None
    machine_rate: Price
    operators: Param
    efficiency: Param

    @property
    def labor(self) -> Price:
        """The effective labor rate (derived from the wage class when one is set)."""
        assert self.labor_rate is not None
        return self.labor_rate


class WorkCenters(_Strict):
    """Work centers with wage classes.

    A center with a ``wage_class`` gets ``labor_rate = engineer_rate x multiplier`` (source
    ``derived``); a center without one must give ``labor_rate`` itself.
    """

    engineer_rate: Price | None = None
    wage_classes: list[WageClass] = Field(default_factory=list)
    work_centers: list[WorkCenter]

    @model_validator(mode="after")
    def _derive_labor_rates(self) -> WorkCenters:
        classes = {c.id: c for c in self.wage_classes}
        for wc in self.work_centers:
            if wc.wage_class is None:
                if wc.labor_rate is None:
                    raise ValueError(f"{wc.id}: labor_rate or wage_class required")
                continue
            if wc.wage_class not in classes:
                raise ValueError(f"{wc.id}: unknown wage_class '{wc.wage_class}'")
            if wc.labor_rate is not None:
                raise ValueError(f"{wc.id}: labor_rate is derived from wage_class, remove it")
            if self.engineer_rate is None:
                raise ValueError(f"{wc.id}: wage_class needs engineer_rate")
            cls = classes[wc.wage_class]
            mult = cls.multiplier.value
            base = self.engineer_rate.value
            wc.labor_rate = Price(
                value=None if mult is None or base is None else base * mult,
                currency=self.engineer_rate.currency,
                per=self.engineer_rate.per,
                source=Source.DERIVED,
                desc=f"İşçilik saat ücreti (türetilmiş: mühendis ücreti × {cls.id})",
                note=f"engineer_rate × wage_classes[{cls.id}].multiplier",
            )
        return self


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
    warn_review_ratio: Param = Param(
        value=0.25,
        unit="ratio",
        source=Source.DEFAULT,
        desc="0-2° (uyarı) bölgesi kalıp yüzeyinin bu payını aşarsa çekme yönü manual_review",
    )


class UndercutThresholds(_Strict):
    core_max_area_ratio: Param
    noise_area_ratio: Param = Param(
        value=0.002,
        unit="ratio",
        source=Source.DEFAULT,
        desc="Bu orandan küçük VE noise_draft_deg'den sığ ters açı lekesi gürültü sayılır",
    )
    noise_draft_deg: Param = Param(
        value=0.5,
        unit="deg",
        source=Source.DEFAULT,
        desc="Gürültü sayılabilecek en derin ters açı (0 ile -bu değer arası)",
    )
    opposite_cos: Param = Param(
        value=-0.5,
        unit="ratio",
        source=Source.DEFAULT,
        desc="İki ters açı bölgesinin yanal normal kosinüsü bunun altındaysa karşılıklı",
    )
    max_split_pieces: Param = Param(
        value=3,
        unit="count",
        source=Source.USER,
        desc="Bölünmüş kalıpta önerilecek en fazla parça sayısı (1, 2, 3 parçalı seçenek)",
    )
    threshold_margin_ratio: Param = Param(
        value=0.2,
        unit="ratio",
        source=Source.DEFAULT,
        desc="Bölge alan oranı maça eşiğine bu bağıl pay kadar yakınsa karar manual_review",
    )


class PullSearchSettings(_Strict):
    max_tilt_deg: Param = Param(
        value=90.0,
        unit="deg",
        source=Source.DEFAULT,
        desc="Aday çekme yönleri seçili yüzeyin ortalama iç normalinden en fazla bu kadar sapar",
    )
    samples: Param = Param(
        value=600,
        unit="count",
        source=Source.DEFAULT,
        desc="Küre üzerinde taranan aday yön sayısı (tam küre için; koni dışı atılır)",
    )
    refine_min_step_deg: Param = Param(
        value=0.1,
        unit="deg",
        source=Source.DEFAULT,
        desc="Yerel optimizasyonun en küçük adımı",
    )
    robust_min_quantile: Param = Param(
        value=0.01,
        unit="ratio",
        source=Source.DEFAULT,
        desc="En küçük çekme açısı üçgenlerin bu payı dışarıda bırakılarak okunur (gürültü)",
    )


class TrimRules(_Strict):
    trim_all_openings: Param


class TessellationThresholds(_Strict):
    max_angular_deflection_deg: Param


class CostingDefaults(_Strict):
    nesting_scrap_ratio: Param
    nest_efficiency: Param = Param(
        value=0.8,
        unit="ratio",
        source=Source.USER,
        desc="Kumaş nest verimi (kullanılan alan / harcanan alan); harcanan = 1 / verim",
    )
    fire_ratio: Param = Param(
        value=0.0, unit="ratio", source=Source.DEFAULT, desc="Nest dışı ek fire payı"
    )
    learning_curve_ratio: Param
    learning_reference_unit: Param
    overhead_ratio: Param
    overhead_basis: Literal["direct_cost", "labor"]
    margin_ratio: Param
    margin_basis: Literal["markup", "margin"]
    consumable_waste_ratio: Param
    autoclave_spacing: Param
    min_order_value: Price
    sweep_quantities: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)


class AcceptanceThresholds(_Strict):
    cost_tolerance: Param


class Thresholds(_Strict):
    draft: DraftThresholds
    undercut: UndercutThresholds
    pull_search: PullSearchSettings = Field(default_factory=PullSearchSettings)
    trim: TrimRules
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
    margin: Param | None = Field(
        None, description="Safety factor on the paid time of this operation (ratio, > 0)"
    )
    setup: Param | None = None
    times: list[TimeTerm] = Field(default_factory=list)
    consumables: list[ConsumableUse] = Field(default_factory=list)

    @model_validator(mode="after")
    def _margin_positive(self) -> Operation:
        v = None if self.margin is None else self.margin.value
        if self.margin is not None and v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f"operation {self.id}: margin must be a number > 0")
        return self


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


# --- blueprints.yaml ---------------------------------------------------------------

NreType = Literal["tool", "master", "fixture", "other"]


class BlueprintItem(_Strict):
    """One line item a blueprint creates in a new quote."""

    type: Literal["part", "nre"]
    id: str
    name: str
    nre_type: NreType | None = None
    tooling_exists: bool = False

    @model_validator(mode="after")
    def _check(self) -> BlueprintItem:
        if (self.type == "nre") != (self.nre_type is not None):
            raise ValueError("nre_type is required for (and only for) type 'nre'")
        return self


class Blueprint(_Strict):
    id: str
    name: str
    description: str = ""
    items: list[BlueprintItem] = Field(min_length=1)


class Blueprints(_Strict):
    blueprints: list[Blueprint]
