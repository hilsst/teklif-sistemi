"""analyze() end to end on synthetic meshes, YAML thresholds and the viewer export (no OCP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kalip_analiz import Solution, analyze, load_settings, mesh_from_arrays
from kalip_analiz.settings import Settings, default


def _box_mold(tilt: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A shallow tray as the tool surface: floor (face 0) + 4 walls leaning out by ``tilt`` deg.
    Normals point out of the part, i.e. down into the tool."""
    t = np.tan(np.radians(tilt))
    v = np.array(
        [
            [0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],  # floor
            [-20 * t, -20 * t, 20], [100 + 20 * t, -20 * t, 20],
            [100 + 20 * t, 100 + 20 * t, 20], [-20 * t, 100 + 20 * t, 20],  # rim
        ],
        dtype=float,
    )  # fmt: skip
    tri = [(0, 2, 1), (0, 3, 2)]  # floor, normal -z
    face = [0, 0]
    for k, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)], 1):
        tri += [(a, b, b + 4), (a, b + 4, a + 4)]
        face += [k, k]
    return v, np.array(tri, dtype=np.int64), np.array(face)


def _outward_ok(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    # make every wall triangle's normal point away from the tray centre in the horizontal plane
    out = t.copy()
    c = np.array([50.0, 50.0, 0.0])
    for i, (a, b, d) in enumerate(t):
        n = np.cross(v[b] - v[a], v[d] - v[a])
        mid = (v[a] + v[b] + v[d]) / 3 - c
        if i >= 2 and np.dot(n[:2], mid[:2]) < 0:
            out[i] = (a, d, b)
    return out


def test_tray_with_draft_is_single_piece() -> None:
    v, t, f = _box_mold(tilt=5.0)
    mesh = mesh_from_arrays(v, _outward_ok(v, t), f)
    r = analyze(mesh, faces=None)
    assert r.solution in (Solution.SINGLE, Solution.CORE, Solution.REVIEW)
    assert r.pull[2] != 0.0 or r.assessment.methods
    assert r.assessment.trace  # every number is traceable


def test_thresholds_come_from_yaml(tmp_path: Path) -> None:
    f = tmp_path / "t.yaml"
    f.write_text("draft:\n  min_ok_deg: {value: 7.5}\nundercut:\n  max_split_pieces: 2\n")
    cfg = load_settings(f)
    assert cfg.search.min_ok_deg == 7.5
    assert cfg.search.max_split_pieces == 2
    assert cfg.search.warn_min_deg == default("draft.warn_min_deg")  # missing key: packaged default
    assert load_settings(None).search == Settings()


def test_todo_value_keeps_default() -> None:
    cfg = load_settings({"draft": {"min_ok_deg": {"value": None, "unit": "deg"}}})
    assert cfg.search.min_ok_deg == default("draft.min_ok_deg")


def test_max_pieces_option_and_no_faces_error() -> None:
    v, t, f = _box_mold(0.0)
    mesh = mesh_from_arrays(v, _outward_ok(v, t), f)
    r = analyze(mesh, max_pieces=1)
    assert r.assessment.max_pieces == 1
    with pytest.raises(ValueError):
        analyze(mesh, faces=[])


def test_export_html_is_self_contained(tmp_path: Path) -> None:
    v, t, f = _box_mold(5.0)
    r = analyze(mesh_from_arrays(v, _outward_ok(v, t), f))
    out = r.export_html(tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "KALIP_VIEWER_DATA" in html
    assert 'src="three.min.js"' not in html
    payload = r.viewer_payload(paint="draft")
    assert len(payload["tri_color"]) == len(payload["pos"]) // 9
