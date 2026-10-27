"""Quote engine: QuoteInput + Library -> fully traceable list of CostLines.

Order of evaluation: learning factor, operation times, labor/machine, material,
consumables, autoclave, sums, overhead, price, NRE, order total. User overrides are
applied as each line is created, so every downstream line uses the overridden value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from teklif.common.models import (
    CostLine,
    LibValue,
    Param,
    ReviewFlag,
    Source,
    derived,
    formula_line,
    sum_line,
)
from teklif.costing import formulas as f
from teklif.costing.inputs import QuoteInput
from teklif.library.loader import Library
from teklif.library.schema import (
    Autoclave,
    CureCycle,
    Operation,
    Prepreg,
    ProcessTemplate,
    WorkCenter,
)
from teklif.routing.route import DriverValue, active_operations, compute_drivers


class QuoteInputError(ValueError):
    """The input references something the library does not have."""


class QuoteResult(BaseModel):
    input: QuoteInput
    currency: str
    lines: list[CostLine]

    def get(self, key: str) -> CostLine:
        for line in self.lines:
            if line.key == key:
                return line
        raise KeyError(key)

    def value(self, key: str) -> float | None:
        return self.get(key).final

    @property
    def review_items(self) -> list[tuple[str, ReviewFlag]]:
        """Every flag with the line it belongs to (manual_review list)."""
        return [(ln.key, fl) for ln in self.lines for fl in ln.flags]

    @property
    def manual_review(self) -> bool:
        return bool(self.review_items)


class _Lines:
    """Collects lines and applies user overrides by key."""

    def __init__(self, overrides: Mapping[str, Any]) -> None:
        self.lines: list[CostLine] = []
        self._overrides = dict(overrides)
        self._used: set[str] = set()

    def add(self, line: CostLine) -> CostLine:
        if line.key in self._overrides:
            line.override = self._overrides[line.key]
            self._used.add(line.key)
        self.lines.append(line)
        return line

    def unused_overrides(self) -> list[str]:
        return sorted(set(self._overrides) - self._used)


def _p(file: str, path: str) -> str:
    return f"{file}:{path}"


def _user(value: float, unit: str, desc: str | None = None) -> Param:
    return Param(value=value, unit=unit, source=Source.USER, desc=desc)


def fx_param(lib: Library, from_ccy: str, to_ccy: str) -> tuple[Param, str | None]:
    """Multiplier converting ``from_ccy`` amounts into ``to_ccy``."""
    unit = f"{to_ccy}/{from_ccy}"
    if from_ccy == to_ccy:
        return Param(value=1.0, unit=unit, source=Source.DEFAULT, desc="same currency"), None
    rates = lib.currency.rates
    if from_ccy not in rates or to_ccy not in rates:
        raise QuoteInputError(f"Unknown currency pair {from_ccy}->{to_ccy}")
    r_from, r_to = rates[from_ccy].value, rates[to_ccy].value
    value = r_to / r_from if r_from is not None and r_to is not None else None
    todo = from_ccy if r_from is None else to_ccy
    return Param(value=value, unit=unit, source=Source.LIBRARY), _p(
        "currency.yaml", f"rates.{todo}"
    )


def _line(
    key: str,
    description: str,
    formula: str,
    inputs: Mapping[str, tuple[LibValue, str | None] | LibValue],
    compute: Callable[[Mapping[str, float]], float],
    unit: str,
    category: str,
    flags: list[ReviewFlag] | None = None,
) -> CostLine:
    values: dict[str, LibValue] = {}
    paths: dict[str, str] = {}
    for name, item in inputs.items():
        if isinstance(item, tuple):
            values[name], path = item
            if path is not None:
                paths[name] = path
        else:
            values[name] = item
    return formula_line(
        key, description, formula, values, compute, unit, category, paths, flags or []
    )


def _from_line(line: CostLine, unit: str | None = None) -> Param:
    return derived(line.final, unit or line.unit, desc=line.key)


# --- validation --------------------------------------------------------------------


def _resolve(
    inp: QuoteInput, lib: Library
) -> tuple[ProcessTemplate, dict[str, Prepreg], Autoclave]:
    if inp.process not in lib.processes:
        raise QuoteInputError(f"No process template for '{inp.process}'")
    template = lib.processes[inp.process]
    prepregs = {p.id: p for p in lib.materials.prepregs}
    for mat in {inp.material} | {p.material for p in inp.plies if p.material}:
        if mat not in prepregs:
            raise QuoteInputError(f"Unknown prepreg '{mat}'")
    if inp.cure_cycle not in {c.id for c in template.cure_cycles}:
        raise QuoteInputError(f"Unknown cure cycle '{inp.cure_cycle}' for '{template.id}'")
    if inp.complexity not in lib.thresholds.complexity_factors:
        raise QuoteInputError(f"Unknown complexity '{inp.complexity}'")
    autoclaves = {a.id: a for a in lib.autoclaves.autoclaves}
    if not autoclaves:
        raise QuoteInputError("No autoclave in the library")
    ac_id = inp.autoclave or next(iter(autoclaves))
    if ac_id not in autoclaves:
        raise QuoteInputError(f"Unknown autoclave '{ac_id}'")
    ccy = inp.quote_currency or lib.currency.quote_currency
    if ccy not in lib.currency.rates:
        raise QuoteInputError(f"Unknown quote currency '{ccy}'")
    return template, prepregs, autoclaves[ac_id]


# --- engine ------------------------------------------------------------------------


def compute_quote(inp: QuoteInput, lib: Library) -> QuoteResult:
    template, prepregs, autoclave = _resolve(inp, lib)
    ccy = inp.quote_currency or lib.currency.quote_currency
    tfile = f"processes/{template.id}.yaml"
    costing = lib.thresholds.costing
    ops = active_operations(template, inp)
    drivers = compute_drivers(template, inp)
    work_centers = {w.id: w for w in lib.work_centers.work_centers}
    out = _Lines(inp.overrides)

    def fx(from_ccy: str) -> tuple[Param, str | None]:
        return fx_param(lib, from_ccy, ccy)

    # learning factor ----------------------------------------------------------------
    learning: CostLine | None = None
    if any(op.learning and op.times for op in ops):
        learning = out.add(
            _line(
                "learning.factor",
                "Öğrenme eğrisi çarpanı (sipariş ortalaması)",
                "mean((n / ref)^log2(LC)), n = first .. first + qty - 1",
                {
                    "LC": (
                        costing.learning_curve_ratio,
                        _p("thresholds.yaml", "costing.learning_curve_ratio"),
                    ),
                    "ref": (
                        costing.learning_reference_unit,
                        _p("thresholds.yaml", "costing.learning_reference_unit"),
                    ),
                    "first": _user(inp.first_unit_no, "part"),
                    "qty": _user(inp.quantity, "part"),
                },
                lambda v: f.learning_factor(v["LC"], v["ref"], v["first"], v["qty"]),
                "ratio",
                "time",
            )
        )
    cf = lib.thresholds.complexity_factors[inp.complexity]
    cf_path = _p("thresholds.yaml", f"complexity_factors.{inp.complexity}")

    # operation times, labor and machine ----------------------------------------------
    time_lines: list[CostLine] = []
    labor_lines: list[CostLine] = []
    machine_lines: list[CostLine] = []
    for op in ops:
        wc = work_centers[op.work_center]
        tl = out.add(_time_line(op, wc, tfile, inp, drivers, cf, cf_path, learning))
        time_lines.append(tl)
        labor_lines.append(out.add(_labor_line(op, wc, tl, fx(wc.labor_rate.currency))))
        if wc.machine_rate.value != 0:
            machine_lines.append(out.add(_machine_line(op, wc, tl, fx(wc.machine_rate.currency))))
    out.add(sum_line("time.total", "Toplam süre (parça başı)", time_lines, "min", "time"))

    # material -------------------------------------------------------------------------
    material_lines: list[CostLine] = []
    for i, ply in enumerate(inp.plies, start=1):
        mat = prepregs[ply.material or inp.material]
        if mat.nesting_scrap_ratio is not None:
            scrap = (
                mat.nesting_scrap_ratio,
                _p("materials.yaml", f"prepregs[{mat.id}].nesting_scrap_ratio"),
            )
        else:
            scrap = (
                costing.nesting_scrap_ratio,
                _p("thresholds.yaml", "costing.nesting_scrap_ratio"),
            )
        material_lines.append(
            out.add(
                _line(
                    f"material.ply{i}",
                    f"{ply.name or f'Ply grubu {i}'}: {mat.id}",
                    "area × count × (1 + scrap) × price × fx",
                    {
                        "area": _user(ply.area_m2, "m2"),
                        "count": _user(ply.count, "ply"),
                        "scrap": scrap,
                        "price": (mat.price, _p("materials.yaml", f"prepregs[{mat.id}].price")),
                        "fx": fx(mat.price.currency),
                    },
                    lambda v: f.ply_material_cost(
                        v["area"], v["count"], v["scrap"], v["price"], v["fx"]
                    ),
                    ccy,
                    "material",
                )
            )
        )

    # consumables ----------------------------------------------------------------------
    consumables = {c.id: c for c in lib.materials.consumables}
    consumable_lines: list[CostLine] = []
    for op in ops:
        for use in op.consumables:
            mat_c = consumables[use.material]
            d = drivers[use.driver]
            consumable_lines.append(
                out.add(
                    _line(
                        f"consumable.{op.id}.{mat_c.id}",
                        f"{op.name}: {mat_c.name}",
                        "driver × qty × (1 + waste) × price × fx",
                        {
                            "driver": (d.param, d.path),
                            "qty": (
                                use.qty,
                                _p(tfile, f"operations[{op.id}].consumables[{mat_c.id}].qty"),
                            ),
                            "waste": (
                                costing.consumable_waste_ratio,
                                _p("thresholds.yaml", "costing.consumable_waste_ratio"),
                            ),
                            "price": (
                                mat_c.price,
                                _p("materials.yaml", f"consumables[{mat_c.id}].price"),
                            ),
                            "fx": fx(mat_c.price.currency),
                        },
                        lambda v: f.consumable_cost(
                            v["driver"], v["qty"], v["waste"], v["price"], v["fx"]
                        ),
                        ccy,
                        "consumable",
                    )
                )
            )

    # autoclave ------------------------------------------------------------------------
    autoclave_lines: list[CostLine] = []
    if any(op.autoclave_cure for op in ops):
        cycle = next(c for c in template.cure_cycles if c.id == inp.cure_cycle)
        autoclave_lines += _autoclave_lines(out, inp, lib, autoclave, cycle, prepregs, tfile, fx)

    # sums, overhead, price -----------------------------------------------------------
    mat_total = out.add(sum_line("material.total", "Malzeme", material_lines, ccy, "total"))
    cons_total = out.add(sum_line("consumable.total", "Sarf", consumable_lines, ccy, "total"))
    lab_total = out.add(sum_line("labor.total", "İşçilik", labor_lines, ccy, "total"))
    mach_total = out.add(sum_line("machine.total", "Makine", machine_lines, ccy, "total"))
    ac_total = out.add(sum_line("autoclave.total", "Autoclave", autoclave_lines, ccy, "total"))
    direct = out.add(
        sum_line(
            "cost.direct",
            "Direkt maliyet",
            [mat_total, cons_total, lab_total, mach_total, ac_total],
            ccy,
            "total",
        )
    )
    basis_line = direct if costing.overhead_basis == "direct_cost" else lab_total
    oh = out.add(
        _line(
            "cost.overhead",
            f"Genel gider ({costing.overhead_basis})",
            f"{basis_line.key} × overhead_ratio",
            {
                "basis": _from_line(basis_line),
                "overhead_ratio": (
                    costing.overhead_ratio,
                    _p("thresholds.yaml", "costing.overhead_ratio"),
                ),
            },
            lambda v: f.overhead(v["basis"], v["overhead_ratio"]),
            ccy,
            "total",
        )
    )
    unit_cost = out.add(sum_line("cost.unit", "Birim maliyet", [direct, oh], ccy, "total"))
    price_formula = (
        "cost × (1 + margin)" if costing.margin_basis == "markup" else "cost / (1 - margin)"
    )
    unit_price = out.add(
        _line(
            "price.unit",
            f"Birim fiyat ({costing.margin_basis})",
            price_formula,
            {
                "cost": _from_line(unit_cost),
                "margin": (costing.margin_ratio, _p("thresholds.yaml", "costing.margin_ratio")),
            },
            lambda v: f.price_from_cost(v["cost"], v["margin"], costing.margin_basis),
            ccy,
            "price",
        )
    )

    # NRE and order total -------------------------------------------------------------
    nre = _nre_line(out, inp, ccy, fx)
    qty = _user(inp.quantity, "part")
    if inp.amortize_nre and nre is not None:
        final = out.add(
            _line(
                "price.unit_final",
                "Birim fiyat (NRE yedirilmiş)",
                "price.unit + nre / qty",
                {"price": _from_line(unit_price), "nre": _from_line(nre), "qty": qty},
                lambda v: v["price"] + v["nre"] / v["qty"],
                ccy,
                "price",
            )
        )
        total_inputs: dict[str, LibValue] = {"unit": _from_line(final), "qty": qty}
        out.add(
            _line(
                "price.order_total",
                "Sipariş toplamı",
                "price.unit_final × qty",
                total_inputs,
                lambda v: v["unit"] * v["qty"],
                ccy,
                "price",
            )
        )
    else:
        final = out.add(
            _line(
                "price.unit_final",
                "Birim fiyat",
                "price.unit",
                {"price": _from_line(unit_price)},
                lambda v: v["price"],
                ccy,
                "price",
            )
        )
        total_inputs = {"unit": _from_line(final), "qty": qty}
        formula = "price.unit_final × qty"
        if nre is not None:
            total_inputs["nre"] = _from_line(nre)
            formula += " + nre"
        out.add(
            _line(
                "price.order_total",
                "Sipariş toplamı",
                formula,
                total_inputs,
                lambda v: v["unit"] * v["qty"] + v.get("nre", 0.0),
                ccy,
                "price",
            )
        )

    unused = out.unused_overrides()
    if unused:
        raise QuoteInputError(f"Overrides for unknown lines: {unused}")
    return QuoteResult(input=inp, currency=ccy, lines=out.lines)


def _time_line(
    op: Operation,
    wc: WorkCenter,
    tfile: str,
    inp: QuoteInput,
    drivers: Mapping[str, DriverValue],
    cf: Param,
    cf_path: str,
    learning: CostLine | None,
) -> CostLine:
    base = f"operations[{op.id}]"
    inputs: dict[str, tuple[LibValue, str | None] | LibValue] = {}
    parts: list[str] = []
    if op.setup is not None:
        inputs["setup"] = (op.setup, _p(tfile, f"{base}.setup"))
        inputs["batch"] = _user(inp.batch_size, "part")
        parts.append("setup / batch")
    run_terms: list[str] = []
    for t in op.times:
        d = drivers[t.driver]
        inputs[f"rate.{t.driver}"] = (t.rate, _p(tfile, f"{base}.times[{t.driver}].rate"))
        inputs[t.driver] = (d.param, d.path)
        run_terms.append(f"rate.{t.driver} × {t.driver}")
    use_cf = op.apply_complexity and bool(op.times)
    use_lc = op.learning and bool(op.times) and learning is not None
    if use_cf:
        inputs["complexity"] = (cf, cf_path)
    if use_lc:
        assert learning is not None
        inputs["learning"] = _from_line(learning)
    if run_terms:
        run = "(" + " + ".join(run_terms) + ")"
        run += " × complexity" if use_cf else ""
        run += " × learning" if use_lc else ""
        parts.append(run)
    inputs["efficiency"] = (
        wc.efficiency,
        _p("work_centers.yaml", f"work_centers[{wc.id}].efficiency"),
    )
    formula = "(" + " + ".join(parts or ["0"]) + ") / efficiency"
    drivers_used = [t.driver for t in op.times]

    def compute(v: Mapping[str, float]) -> float:
        return f.operation_time(
            v.get("setup", 0.0),
            v.get("batch", 1.0),
            [(v[f"rate.{d}"], v[d]) for d in drivers_used],
            v.get("complexity", 1.0),
            v.get("learning", 1.0),
            v["efficiency"],
        )

    return _line(f"time.{op.id}", op.name, formula, inputs, compute, "min", "time")


def _labor_line(
    op: Operation, wc: WorkCenter, time: CostLine, fx: tuple[Param, str | None]
) -> CostLine:
    wpath = f"work_centers[{wc.id}]"
    return _line(
        f"labor.{op.id}",
        f"{op.name} ({wc.id})",
        "time / 60 × operators × labor_rate × fx",
        {
            "time": _from_line(time),
            "operators": (wc.operators, _p("work_centers.yaml", f"{wpath}.operators")),
            "labor_rate": (wc.labor_rate, _p("work_centers.yaml", f"{wpath}.labor_rate")),
            "fx": fx,
        },
        lambda v: f.labor_cost(v["time"], v["operators"], v["labor_rate"], v["fx"]),
        fx[0].unit.split("/")[0],
        "labor",
    )


def _machine_line(
    op: Operation, wc: WorkCenter, time: CostLine, fx: tuple[Param, str | None]
) -> CostLine:
    return _line(
        f"machine.{op.id}",
        f"{op.name} ({wc.id})",
        "time / 60 × machine_rate × fx",
        {
            "time": _from_line(time),
            "machine_rate": (
                wc.machine_rate,
                _p("work_centers.yaml", f"work_centers[{wc.id}].machine_rate"),
            ),
            "fx": fx,
        },
        lambda v: f.machine_cost(v["time"], v["machine_rate"], v["fx"]),
        fx[0].unit.split("/")[0],
        "machine",
    )


def _autoclave_lines(
    out: _Lines,
    inp: QuoteInput,
    lib: Library,
    ac: Autoclave,
    cycle: CureCycle,
    prepregs: Mapping[str, Prepreg],
    tfile: str,
    fx: Callable[[str], tuple[Param, str | None]],
) -> list[CostLine]:
    g = inp.geometry
    apath = f"autoclaves[{ac.id}]"
    spacing = lib.thresholds.costing.autoclave_spacing
    flags: list[ReviewFlag] = []
    dims = (ac.usable_length.value, ac.usable_width.value, spacing.value)
    if all(x is not None for x in dims):
        length, width, s = (float(x) for x in dims if x is not None)
        if not f.fits_in_autoclave(g.tool_footprint_x_mm, g.tool_footprint_y_mm, s, length, width):
            flags.append(
                ReviewFlag(
                    code="does_not_fit", reason=f"Tool footprint does not fit autoclave {ac.id}"
                )
            )
    cure_temp, max_temp = cycle.cure_temp.value, ac.max_temp.value
    if cure_temp is not None and max_temp is not None and cure_temp > max_temp:
        flags.append(
            ReviewFlag(
                code="over_temperature",
                reason=f"Cycle {cycle.id} exceeds {ac.id} max temperature",
            )
        )
    for mat_id in {p.material or inp.material for p in inp.plies}:
        mat_temp = prepregs[mat_id].cure_temp.value
        if (
            mat_temp is not None
            and cycle.cure_temp.value is not None
            and mat_temp != cycle.cure_temp.value
        ):
            flags.append(
                ReviewFlag(
                    code="cure_mismatch", reason=f"{mat_id} cure {mat_temp} °C ≠ cycle {cycle.id}"
                )
            )
    share = out.add(
        _line(
            "autoclave.share",
            f"Autoclave {ac.id} çevrim payı",
            "(X + 2s)(Y + 2s) / (L × W × load_eff)",
            {
                "X": Param(value=g.tool_footprint_x_mm, unit="mm", source=g.source),
                "Y": Param(value=g.tool_footprint_y_mm, unit="mm", source=g.source),
                "s": (spacing, _p("thresholds.yaml", "costing.autoclave_spacing")),
                "L": (ac.usable_length, _p("autoclaves.yaml", f"{apath}.usable_length")),
                "W": (ac.usable_width, _p("autoclaves.yaml", f"{apath}.usable_width")),
                "load_eff": (ac.load_efficiency, _p("autoclaves.yaml", f"{apath}.load_efficiency")),
            },
            lambda v: f.autoclave_share(v["X"], v["Y"], v["s"], v["L"], v["W"], v["load_eff"]),
            "ratio",
            "autoclave",
            flags,
        )
    )
    cost = out.add(
        _line(
            "autoclave.cost",
            f"Autoclave kürü ({cycle.id})",
            "cycle_time × hourly_rate × fx × share",
            {
                "cycle_time": (cycle.cycle_time, _p(tfile, f"cure_cycles[{cycle.id}].cycle_time")),
                "hourly_rate": (ac.hourly_rate, _p("autoclaves.yaml", f"{apath}.hourly_rate")),
                "fx": fx(ac.hourly_rate.currency),
                "share": _from_line(share),
            },
            lambda v: f.autoclave_cost(v["cycle_time"], v["hourly_rate"], v["fx"], v["share"]),
            fx(ac.hourly_rate.currency)[0].unit.split("/")[0],
            "autoclave",
        )
    )
    return [cost]


def _nre_line(
    out: _Lines, inp: QuoteInput, ccy: str, fx: Callable[[str], tuple[Param, str | None]]
) -> CostLine | None:
    if inp.tooling_exists:
        return None
    if inp.tooling_nre is None:
        return out.add(
            CostLine(
                key="nre.tooling",
                description="Kalıp + master model (NRE)",
                formula="kullanıcı girişi",
                inputs=[],
                result=None,
                unit=ccy,
                category="nre",
                flags=[
                    ReviewFlag(
                        code="nre_missing",
                        reason="Tooling NRE not entered and tool not marked existing",
                    )
                ],
            )
        )
    nre = inp.tooling_nre.model_copy(update={"source": Source.USER})
    return out.add(
        _line(
            "nre.tooling",
            "Kalıp + master model (NRE)",
            "nre × fx",
            {"nre": nre, "fx": fx(nre.currency)},
            lambda v: v["nre"] * v["fx"],
            ccy,
            "nre",
        )
    )
