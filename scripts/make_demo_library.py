"""Build ``examples/demo_library``: the real library with every TODO filled by DUMMY values.

The numbers below are plausible but invented, for demonstrating the system only.
They are NOT company data and must never be used for a real quote.

Usage: uv run python scripts/make_demo_library.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "library"
DST = REPO / "examples" / "demo_library"
NOTE = "DEMO / dummy değer - gerçek firma verisi değil"

# "file": {"path": value}. "[*]" applies to every list item.
DEMO: dict[str, dict[str, float]] = {
    "currency.yaml": {"rates.TRY": 45.0, "rates.USD": 1.08},
    "materials.yaml": {
        "prepregs[*].out_life": 720,
        "prepregs[*].roll_width": 1250,
        "prepregs[PP_CF_UD_120].areal_weight": 150,
        "prepregs[PP_CF_UD_120].cured_ply_thickness": 0.15,
        "prepregs[PP_CF_UD_120].price": 55,
        "prepregs[PP_CF_WOVEN_120].areal_weight": 200,
        "prepregs[PP_CF_WOVEN_120].cured_ply_thickness": 0.22,
        "prepregs[PP_CF_WOVEN_120].price": 65,
        "prepregs[PP_CF_UD_180].areal_weight": 150,
        "prepregs[PP_CF_UD_180].cured_ply_thickness": 0.15,
        "prepregs[PP_CF_UD_180].price": 75,
        "prepregs[PP_CF_WOVEN_180].areal_weight": 200,
        "prepregs[PP_CF_WOVEN_180].cured_ply_thickness": 0.22,
        "prepregs[PP_CF_WOVEN_180].price": 85,
        "prepregs[PP_GF_WOVEN_120].areal_weight": 300,
        "prepregs[PP_GF_WOVEN_120].cured_ply_thickness": 0.25,
        "prepregs[PP_GF_WOVEN_120].price": 18,
        "consumables[*].max_temp": 200,
        "consumables[VAKUM_TORBA].price": 2.5,
        "consumables[BREATHER].price": 2.0,
        "consumables[AYIRICI_FILM].price": 3.0,
        "consumables[PEEL_PLY].price": 2.5,
        "consumables[SIZDIRMAZLIK_BANDI].price": 1.2,
        "consumables[KALIP_AYIRICI].price": 60,
        "consumables[BOYA_ASTAR].price": 45,
        "tooling_materials[*].tool_life": 300,
        "tooling_materials[TB_120].max_service_temp": 135,
        "tooling_materials[TB_120].cte": 50,
        "tooling_materials[TB_120].density": 700,
        "tooling_materials[TB_120].price": 3500,
        "tooling_materials[TB_180].max_service_temp": 200,
        "tooling_materials[TB_180].cte": 35,
        "tooling_materials[TB_180].density": 750,
        "tooling_materials[TB_180].price": 6000,
        "tooling_materials[CFRP_TOOLING_PREPREG].max_service_temp": 200,
        "tooling_materials[CFRP_TOOLING_PREPREG].cte": 3,
        "tooling_materials[CFRP_TOOLING_PREPREG].density": 1550,
        "tooling_materials[CFRP_TOOLING_PREPREG].price": 120,
        "tooling_materials[GFRP_TOOLING].max_service_temp": 150,
        "tooling_materials[GFRP_TOOLING].cte": 14,
        "tooling_materials[GFRP_TOOLING].density": 1800,
        "tooling_materials[GFRP_TOOLING].price": 30,
        "tooling_materials[INVAR36].max_service_temp": 250,
        "tooling_materials[INVAR36].price": 35,
        "tooling_materials[AL_PLAKA].max_service_temp": 200,
        "tooling_materials[AL_PLAKA].price": 6,
        "tooling_materials[CELIK].max_service_temp": 300,
        "tooling_materials[CELIK].price": 3,
    },
    "work_centers.yaml": {
        "work_centers[*].labor_rate": 25,
        "work_centers[*].operators": 1,
        "work_centers[*].efficiency": 0.85,
        "work_centers[KESIM_MASASI].machine_rate": 30,
        "work_centers[TEMIZ_ODA].machine_rate": 8,
        "work_centers[TEMIZ_ODA].operators": 2,
        "work_centers[AUTOCLAVE_YUKLEME].machine_rate": 0,
        "work_centers[KALIPTAN_CIKARMA].machine_rate": 0,
        "work_centers[CNC_TRIM].machine_rate": 45,
        "work_centers[EL_ISCILIGI].machine_rate": 0,
        "work_centers[NDT_UT].machine_rate": 40,
        "work_centers[OLCUM].machine_rate": 35,
        "work_centers[BOYA].machine_rate": 15,
        "work_centers[MONTAJ].machine_rate": 0,
        "work_centers[KALIP_CNC].machine_rate": 60,
        "work_centers[KALIP_ATOLYE].machine_rate": 5,
    },
    "autoclaves.yaml": {
        "autoclaves[AC1].inner_diameter": 3000,
        "autoclaves[AC1].usable_length": 8000,
        "autoclaves[AC1].usable_width": 2400,
        "autoclaves[AC1].max_temp": 200,
        "autoclaves[AC1].max_pressure": 7,
        "autoclaves[AC1].vacuum_ports": 20,
        "autoclaves[AC1].thermocouples": 24,
        "autoclaves[AC1].hourly_rate": 180,
        "autoclaves[AC1].load_efficiency": 0.55,
    },
    "thresholds.yaml": {
        "costing.nesting_scrap_ratio": 0.35,
        "costing.learning_curve_ratio": 0.88,
        "costing.overhead_ratio": 0.30,
        "costing.margin_ratio": 0.15,
        "costing.consumable_waste_ratio": 0.15,
        "costing.autoclave_spacing": 150,
        "complexity_factors.orta": 1.25,
        "complexity_factors.yuksek": 1.6,
    },
    "processes/autoclave_prepreg.yaml": {
        "parameters.debulk_interval_plies": 4,
        "cure_cycles[cure_120].pressure": 6,
        "cure_cycles[cure_120].cycle_time": 6,
        "cure_cycles[cure_180].pressure": 7,
        "cure_cycles[cure_180].cycle_time": 9,
        "operations[kesim].setup": 20,
        "operations[kesim].times[ply_area_total_m2].rate": 1.2,
        "operations[kesim].times[ply_count].rate": 0.5,
        "operations[serim].setup": 30,
        "operations[serim].times[ply_area_total_m2].rate": 12,
        "operations[serim].times[ply_count].rate": 1.5,
        "operations[serim].consumables[KALIP_AYIRICI].qty": 0.03,
        "operations[debulk].times[debulk_count].rate": 20,
        "operations[torbalama].times[tool_area_m2].rate": 15,
        "operations[torbalama].times[tool_perimeter_m].rate": 3,
        "operations[torbalama].consumables[PEEL_PLY].qty": 1.1,
        "operations[torbalama].consumables[AYIRICI_FILM].qty": 1.2,
        "operations[torbalama].consumables[BREATHER].qty": 1.3,
        "operations[torbalama].consumables[VAKUM_TORBA].qty": 1.6,
        "operations[torbalama].consumables[SIZDIRMAZLIK_BANDI].qty": 1.05,
        "operations[kur].setup": 45,
        "operations[kur].times[per_part].rate": 10,
        "operations[kaliptan_cikarma].times[part_area_m2].rate": 8,
        "operations[kaliptan_cikarma].times[tool_area_m2].rate": 6,
        "operations[trim].setup": 40,
        "operations[trim].times[part_perimeter_m].rate": 2.5,
        "operations[trim].times[hole_count].rate": 1.5,
        "operations[trim].times[per_part].rate": 5,
        "operations[el_isciligi].times[part_perimeter_m].rate": 6,
        "operations[ndt].setup": 30,
        "operations[ndt].times[part_area_m2].rate": 20,
        "operations[ndt].times[per_part].rate": 15,
        "operations[olcum].setup": 30,
        "operations[olcum].times[per_part].rate": 20,
        "operations[olcum].times[hole_count].rate": 0.5,
        "operations[boya].setup": 30,
        "operations[boya].times[part_area_m2].rate": 25,
        "operations[boya].consumables[BOYA_ASTAR].qty": 0.15,
        "operations[montaj].times[per_part].rate": 30,
    },
}


def _targets(node: Any, segments: list[str]) -> list[dict[str, Any]]:
    """Resolve a path like ``operations[kesim].times[ply_count].rate`` to value dicts."""
    if not segments:
        return [node]
    seg, rest = segments[0], segments[1:]
    if "[" in seg:
        name, key = seg[:-1].split("[")
        items = node[name]
        hits = [
            it
            for it in items
            if key == "*" or key in (it.get("id"), it.get("material"), it.get("driver"))
        ]
        if not hits:
            raise KeyError(seg)
        return [t for it in hits for t in _targets(it, rest)]
    return _targets(node[seg], rest)


def _remaining(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        if "value" in node and node["value"] is None:
            return [path]
        return [p for k, v in node.items() for p in _remaining(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _remaining(v, f"{path}[{i}]")]
    return []


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    missing: list[str] = []
    for file, values in DEMO.items():
        path = DST / file
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Specific paths first, so "[*]" only fills what they left empty.
        for dotted, value in sorted(values.items(), key=lambda kv: "[*]" in kv[0]):
            for target in _targets(data, dotted.split(".")):
                if target.get("value") is None:
                    target["value"] = value
                    target["source"] = "default"
                    target["note"] = NOTE
        missing += [f"{file}:{p}" for p in _remaining(data)]
        header = f"# {NOTE}. scripts/make_demo_library.py ile üretildi; elle düzenlemeyin.\n"
        path.write_text(
            header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    if missing:
        print("Unfilled TODO values:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    print(f"Wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
