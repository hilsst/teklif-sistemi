"""Every threshold of the analysis, read from YAML (never written in code).

The packaged ``thresholds.yaml`` holds the defaults; a user file (or an already parsed mapping,
or an object such as ``teklif.library.schema.Thresholds``) overrides any subset of its keys.
Each value remembers where it came from (``Source``) so the decision trace can show it.

Value nodes are ``{value: 2.0, unit: deg, source: user, desc: ...}`` or a bare number. A node
whose value is ``null`` (a TODO in the user's file) keeps the packaged default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from kalip_analiz.trace import Source

DEFAULTS_FILE = Path(__file__).with_name("thresholds.yaml")
LIB = "thresholds.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


_DEFAULT_TREE = _read_yaml(DEFAULTS_FILE)


def _node(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if node is None:
            return None
        node = node.get(part) if isinstance(node, Mapping) else getattr(node, part, None)
    return node


def _value_source(node: Any) -> tuple[Any, Source] | None:
    """(value, source) of a value node, None when absent or TODO."""
    if node is None:
        return None
    if isinstance(node, Mapping):
        v, src = node.get("value"), node.get("source")
    elif isinstance(node, int | float):
        v, src = node, None
    else:
        v, src = getattr(node, "value", None), getattr(node, "source", None)
    if v is None:
        return None
    return v, Source(src) if src is not None else Source.LIBRARY


def default(path: str) -> float:
    """Packaged default of ``path`` (e.g. ``"draft.min_ok_deg"``)."""
    vs = _value_source(_node(_DEFAULT_TREE, path))
    if vs is None:
        raise KeyError(f"{DEFAULTS_FILE.name} has no value for {path}")
    return float(vs[0])


def _d(path: str) -> Any:
    return field(default=default(path), metadata={"path": path})


def _i(path: str) -> Any:
    return field(default=int(default(path)), metadata={"path": path})


@dataclass(frozen=True)
class _Group:
    sources: dict[str, Source] = field(default_factory=dict, compare=False)

    def source(self, name: str) -> Source:
        return self.sources.get(name, Source.DEFAULT)


@dataclass(frozen=True)
class Settings(_Group):
    """Thresholds of the pull-direction search and the tooling decision."""

    min_ok_deg: float = _d("draft.min_ok_deg")
    warn_min_deg: float = _d("draft.warn_min_deg")
    warn_review_ratio: float = _d("draft.warn_review_ratio")
    core_max_area_ratio: float = _d("undercut.core_max_area_ratio")
    noise_area_ratio: float = _d("undercut.noise_area_ratio")
    noise_draft_deg: float = _d("undercut.noise_draft_deg")
    opposite_cos: float = _d("undercut.opposite_cos")
    max_split_pieces: int = _i("undercut.max_split_pieces")
    threshold_margin_ratio: float = _d("undercut.threshold_margin_ratio")
    core_cost_weight: float = _d("undercut.core_cost_weight")
    core_area_cost_weight: float = _d("undercut.core_area_cost_weight")
    split_cost_weight: float = _d("undercut.split_cost_weight")
    unresolved_cost_weight: float = _d("undercut.unresolved_cost_weight")
    warn_cost_weight: float = _d("undercut.warn_cost_weight")
    sandwich_tol_deg: float = _d("undercut.sandwich_tol_deg")
    max_tilt_deg: float = _d("pull_search.max_tilt_deg")
    piece_max_tilt_deg: float = _d("pull_search.piece_max_tilt_deg")
    samples: int = _i("pull_search.samples")
    refine_min_step_deg: float = _d("pull_search.refine_min_step_deg")
    robust_min_quantile: float = _d("pull_search.robust_min_quantile")
    plane_normals: int = _i("pull_search.plane_normals")
    exact_candidates: int = _i("pull_search.exact_candidates")
    axis_splits: int = _i("pull_search.axis_splits")
    axis_mid_plane: int = _i("pull_search.axis_mid_plane")


@dataclass(frozen=True)
class MeshParams(_Group):
    """STEP tessellation: fine mesh for the analysis, coarse mesh for the 3D view."""

    linear_deflection_mm: float = _d("mesh.linear_deflection_mm")
    angular_deflection_deg: float = _d("mesh.angular_deflection_deg")
    view_deflection_mm: float = _d("mesh.view_deflection_mm")
    view_angle_deg: float = _d("mesh.view_angle_deg")


@dataclass(frozen=True)
class ViewerParams(_Group):
    """3D viewer: selection pause and the size of the drawn markers."""

    select_pause_s: float = _d("viewer.select_pause_s")
    parting_wall_ratio: float = _d("viewer.parting_wall_ratio")
    core_marker_ratio: float = _d("viewer.core_marker_ratio")
    core_leader_ratio: float = _d("viewer.core_leader_ratio")
    piece_arrow_ratio: float = _d("viewer.piece_arrow_ratio")


def field_path(cls: type[_Group], name: str) -> str:
    """YAML path of a settings field, e.g. ``draft.min_ok_deg``."""
    return str(next(f.metadata["path"] for f in fields(cls) if f.name == name))


def _build[G: _Group](cls: type[G], root: Any) -> G:
    vals: dict[str, Any] = {}
    sources: dict[str, Source] = {}
    for f in fields(cls):
        if f.name == "sources":
            continue
        vs = _value_source(_node(root, field_path(cls, f.name))) if root is not None else None
        if vs is None:
            continue
        v, src = vs
        vals[f.name] = int(v) if f.type in (int, "int") else float(v)
        sources[f.name] = src
    return cls(**vals, sources=sources)


@dataclass(frozen=True)
class AnalysisSettings:
    search: Settings
    mesh: MeshParams
    viewer: ViewerParams
    file: str = LIB  # name shown in the trace


ThresholdSource = str | Path | Mapping[str, Any] | AnalysisSettings | object | None


def load_settings(src: ThresholdSource = None) -> AnalysisSettings:
    """Settings from a YAML file, a parsed mapping, an attribute tree (``Thresholds`` model) or
    ``None`` (packaged defaults only). Keys missing from ``src`` keep the packaged default."""
    if isinstance(src, AnalysisSettings):
        return src
    name = LIB
    root: Any = src
    if isinstance(src, str | Path):
        p = Path(src)
        root, name = _read_yaml(p), p.name
    return AnalysisSettings(
        search=_build(Settings, root),
        mesh=_build(MeshParams, root),
        viewer=_build(ViewerParams, root),
        file=name,
    )


def settings_from_thresholds(thr: Any) -> Settings:
    """Search/decision settings from a thresholds tree; a TODO (None) value keeps the packaged
    default and is reported by the caller's TODO list, never invented here."""
    return load_settings(thr).search
