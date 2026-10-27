from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from teklif.costing.inputs import QuoteInput
from teklif.library import Library, load_library
from teklif.library.loader import DEFAULT_LIBRARY_DIR

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_INPUT = REPO / "examples" / "ornek_parca.yaml"

# Test-only numbers used to fill every TODO value. They are NOT company data.
FILL_BY_KEY: dict[str, float] = {
    "efficiency": 0.8,
    "load_efficiency": 0.6,
    "learning_curve_ratio": 0.9,
    "orta": 1.2,
    "yuksek": 1.5,
    "TRY": 40.0,
    "USD": 1.1,
    "usable_length": 6000.0,
    "usable_width": 2500.0,
    "inner_diameter": 3000.0,
    "autoclave_spacing": 100.0,
    "cycle_time": 6.0,
    "labor_rate": 40.0,
    "machine_rate": 20.0,
    "hourly_rate": 150.0,
    "operators": 1.0,
    "debulk_interval_plies": 4.0,
    "overhead_ratio": 0.25,
    "margin_ratio": 0.15,
    "nesting_scrap_ratio": 0.3,
    "consumable_waste_ratio": 0.1,
    "max_temp": 200.0,
}
FILL_BY_UNIT: dict[str, float] = {"min/batch": 30.0, "ratio": 0.1, "degC": 200.0}


def _fill(node: Any, key: str = "") -> None:
    if isinstance(node, dict):
        if "value" in node and node["value"] is None:
            if key in FILL_BY_KEY:
                node["value"] = FILL_BY_KEY[key]
            elif "currency" in node:
                node["value"] = 60.0 if key == "price" else 10.0
            elif node.get("unit") in FILL_BY_UNIT:
                node["value"] = FILL_BY_UNIT[node["unit"]]
            elif str(node.get("unit", "")).startswith("min/"):
                node["value"] = 2.0
            else:
                node["value"] = 1.0
        for k, v in node.items():
            _fill(v, str(k))
    elif isinstance(node, list):
        for item in node:
            _fill(item, key if not isinstance(item, dict) else "")


def make_filled_library(dst: Path) -> Path:
    shutil.copytree(DEFAULT_LIBRARY_DIR, dst)
    for path in dst.rglob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _fill(data)
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")
    return dst


@pytest.fixture(scope="session")
def lib() -> Library:
    return load_library(DEFAULT_LIBRARY_DIR)


@pytest.fixture(scope="session")
def filled_lib_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_filled_library(tmp_path_factory.mktemp("filled") / "library")


@pytest.fixture(scope="session")
def filled_lib(filled_lib_dir: Path) -> Library:
    return load_library(filled_lib_dir)


@pytest.fixture
def lib_copy(tmp_path: Path) -> Path:
    """A writable copy of the library for tests that break it on purpose."""
    dst = tmp_path / "library"
    shutil.copytree(DEFAULT_LIBRARY_DIR, dst)
    return dst


@pytest.fixture
def example_input() -> QuoteInput:
    return QuoteInput.model_validate(yaml.safe_load(EXAMPLE_INPUT.read_text(encoding="utf-8")))
