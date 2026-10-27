"""Load, validate and inspect the YAML library.

``load_library`` parses every file with its pydantic schema and then runs the
cross-file checks (references, units, currencies). Any problem raises
``LibraryError`` with every issue listed, so a broken library never loads halfway.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from teklif.common.models import Param, Price
from teklif.common.units import is_known_unit
from teklif.library.schema import (
    DRIVER_UNITS,
    Autoclaves,
    CurrencyTable,
    Enums,
    Materials,
    ProcessTemplate,
    Thresholds,
    WorkCenters,
)

DEFAULT_LIBRARY_DIR = Path(
    os.environ.get("TEKLIF_LIBRARY", Path(__file__).resolve().parents[3] / "library")
)


class LibraryError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("Library validation failed:\n- " + "\n- ".join(issues))


@dataclass(frozen=True)
class Library:
    root: Path
    currency: CurrencyTable
    enums: Enums
    materials: Materials
    work_centers: WorkCenters
    autoclaves: Autoclaves
    thresholds: Thresholds
    processes: dict[str, ProcessTemplate]

    def files(self) -> dict[str, BaseModel]:
        """Every loaded model keyed by its file path relative to the library root."""
        out: dict[str, BaseModel] = {
            "currency.yaml": self.currency,
            "enums.yaml": self.enums,
            "materials.yaml": self.materials,
            "work_centers.yaml": self.work_centers,
            "autoclaves.yaml": self.autoclaves,
            "thresholds.yaml": self.thresholds,
        }
        for pid, proc in self.processes.items():
            out[f"processes/{pid}.yaml"] = proc
        return out


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse[M: BaseModel](model: type[M], path: Path, issues: list[str]) -> M | None:
    try:
        return model.model_validate(_read_yaml(path))
    except FileNotFoundError:
        issues.append(f"{path.name}: file not found")
    except (ValidationError, yaml.YAMLError) as exc:
        issues.append(f"{path.name}: {exc}")
    return None


def load_library(root: Path | str = DEFAULT_LIBRARY_DIR) -> Library:
    root = Path(root)
    issues: list[str] = []
    currency = _parse(CurrencyTable, root / "currency.yaml", issues)
    enums = _parse(Enums, root / "enums.yaml", issues)
    materials = _parse(Materials, root / "materials.yaml", issues)
    work_centers = _parse(WorkCenters, root / "work_centers.yaml", issues)
    autoclaves = _parse(Autoclaves, root / "autoclaves.yaml", issues)
    thresholds = _parse(Thresholds, root / "thresholds.yaml", issues)
    processes: dict[str, ProcessTemplate] = {}
    for path in sorted((root / "processes").glob("*.yaml")):
        proc = _parse(ProcessTemplate, path, issues)
        if proc is not None:
            if proc.id != path.stem:
                issues.append(f"processes/{path.name}: id '{proc.id}' must match file name")
            processes[proc.id] = proc
    if issues:
        raise LibraryError(issues)
    assert currency and enums and materials and work_centers and autoclaves and thresholds
    lib = Library(
        root=root,
        currency=currency,
        enums=enums,
        materials=materials,
        work_centers=work_centers,
        autoclaves=autoclaves,
        thresholds=thresholds,
        processes=processes,
    )
    issues = validate_references(lib)
    if issues:
        raise LibraryError(issues)
    return lib


# --- cross-file validation ---------------------------------------------------------


def _duplicates(ids: list[str]) -> set[str]:
    return {i for i in ids if ids.count(i) > 1}


def iter_values(obj: Any, path: str = "") -> Iterator[tuple[str, Param | Price]]:
    """Yield every ``Param``/``Price`` in a model tree with a dotted path."""
    if isinstance(obj, Param | Price):
        yield path, obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from iter_values(getattr(obj, name), f"{path}.{name}" if path else name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_values(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = (
                getattr(v, "id", None) or getattr(v, "material", None) or getattr(v, "driver", None)
            )
            label = f"{key}" if key is not None else str(i)
            yield from iter_values(v, f"{path}[{label}]")


def validate_references(lib: Library) -> list[str]:
    issues: list[str] = []
    currencies = frozenset(lib.currency.rates)

    if lib.currency.base_currency not in currencies:
        issues.append("currency.yaml: base_currency missing from rates")
    elif lib.currency.rates[lib.currency.base_currency].value != 1.0:
        issues.append("currency.yaml: base currency rate must be 1.0")
    if lib.currency.quote_currency not in currencies:
        issues.append("currency.yaml: quote_currency missing from rates")

    for fname, model in lib.files().items():
        for path, v in iter_values(model):
            if isinstance(v, Price):
                if v.currency not in currencies:
                    issues.append(f"{fname}: {path}: unknown currency '{v.currency}'")
                if not is_known_unit(v.per):
                    issues.append(f"{fname}: {path}: unknown unit '{v.per}'")
            elif not is_known_unit(v.unit, currencies):
                issues.append(f"{fname}: {path}: unknown unit '{v.unit}'")

    for name in type(lib.enums).model_fields:
        for dup in _duplicates(lib.enums.ids(name)):
            issues.append(f"enums.yaml: {name}: duplicate id '{dup}'")

    mat = lib.materials
    all_material_ids = (
        [m.id for m in mat.prepregs]
        + [m.id for m in mat.consumables]
        + [m.id for m in mat.tooling_materials]
    )
    for dup in _duplicates(all_material_ids):
        issues.append(f"materials.yaml: duplicate id '{dup}'")
    tool_classes = set(lib.enums.ids("tool_material_classes"))
    for tm in mat.tooling_materials:
        if tm.tool_class not in tool_classes:
            issues.append(f"materials.yaml: {tm.id}: unknown tool_class '{tm.tool_class}'")

    wc_ids = [w.id for w in lib.work_centers.work_centers]
    for dup in _duplicates(wc_ids):
        issues.append(f"work_centers.yaml: duplicate id '{dup}'")
    for dup in _duplicates([a.id for a in lib.autoclaves.autoclaves]):
        issues.append(f"autoclaves.yaml: duplicate id '{dup}'")

    complexity_ids = set(lib.enums.ids("complexity"))
    if set(lib.thresholds.complexity_factors) != complexity_ids:
        issues.append("thresholds.yaml: complexity_factors keys must match enums.complexity ids")

    consumables = {c.id: c for c in mat.consumables}
    process_ids = set(lib.enums.ids("processes"))
    operation_ids = set(lib.enums.ids("operations"))
    for pid, proc in lib.processes.items():
        where = f"processes/{pid}.yaml"
        if pid not in process_ids:
            issues.append(f"{where}: process id not in enums.processes")
        for dup in _duplicates([o.id for o in proc.operations]):
            issues.append(f"{where}: duplicate operation id '{dup}'")
        seqs = [o.seq for o in proc.operations]
        if seqs != sorted(seqs):
            issues.append(f"{where}: operations must be in ascending seq order")
        for op in proc.operations:
            if op.id not in operation_ids:
                issues.append(f"{where}: {op.id}: operation id not in enums.operations")
            if op.work_center not in wc_ids:
                issues.append(f"{where}: {op.id}: unknown work_center '{op.work_center}'")
            if op.setup is not None and op.setup.unit != "min/batch":
                issues.append(f"{where}: {op.id}: setup unit must be 'min/batch'")
            for t in op.times:
                expected = f"min/{DRIVER_UNITS[t.driver]}"
                if t.rate.unit != expected:
                    issues.append(f"{where}: {op.id}: rate for {t.driver} must be '{expected}'")
            for c in op.consumables:
                if c.material not in consumables:
                    issues.append(f"{where}: {op.id}: unknown consumable '{c.material}'")
                    continue
                expected = f"{consumables[c.material].price.per}/{DRIVER_UNITS[c.driver]}"
                if c.qty.unit != expected:
                    issues.append(f"{where}: {op.id}: qty for {c.material} must be '{expected}'")
        if any(o.autoclave_cure for o in proc.operations) and not proc.cure_cycles:
            issues.append(f"{where}: autoclave_cure operation requires cure_cycles")
    return issues


# --- TODO report -------------------------------------------------------------------


@dataclass(frozen=True)
class TodoItem:
    file: str
    path: str
    unit: str
    desc: str | None


def list_todos(lib: Library) -> list[TodoItem]:
    """Every library value still waiting for company data."""
    out: list[TodoItem] = []
    for fname, model in lib.files().items():
        for path, v in iter_values(model):
            if v.is_todo:
                out.append(TodoItem(file=fname, path=path, unit=v.unit, desc=v.desc))
    return out
