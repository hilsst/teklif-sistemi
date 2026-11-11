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


def test_to_dict_is_json_and_edges_optional() -> None:
    import json

    v, t, f = _box_mold(5.0)
    mesh = mesh_from_arrays(v, _outward_ok(v, t), f, with_edges=False)
    assert all(not face.edges for face in mesh.faces)
    d = analyze(mesh).to_dict()
    json.dumps(d)
    assert d["methods"] and d["trace"] and "solution" in d


def test_plane_search_limits_come_from_yaml() -> None:
    cfg = load_settings(
        {"pull_search": {"exact_candidates": {"value": 3}, "axis_mid_plane": {"value": 0}}}
    ).search
    assert (cfg.exact_candidates, cfg.axis_mid_plane) == (3, 0)
    d = load_settings(None).search
    assert (d.plane_normals, d.exact_candidates, d.axis_mid_plane, d.axis_splits) == (12, 8, 1, 4)


def test_mid_plane_survives_tight_candidate_limit() -> None:
    """With only one exact candidate the mid-part axis plane must still be evaluated."""
    v, t, f = _box_mold(0.0)
    mesh = mesh_from_arrays(v, _outward_ok(v, t), f, with_edges=False)
    loose = {"pull_search": {"exact_candidates": {"value": 1}, "axis_mid_plane": {"value": 1}}}
    tight = {"pull_search": {"exact_candidates": {"value": 1}, "axis_mid_plane": {"value": 0}}}
    a = analyze(mesh, thresholds=loose, pull=(0, 0, 1)).assessment.methods
    b = analyze(mesh, thresholds=tight, pull=(0, 0, 1)).assessment.methods
    assert sum(m.kind == "plane" for m in a) >= sum(m.kind == "plane" for m in b)


def test_cut_mesh_at_plane_leaves_no_straddling_triangle() -> None:
    import numpy as np

    from kalip_analiz.mesh import cut_mesh_at_planes, mesh_from_arrays, split_triangles_at_plane

    v = np.array([[-100.0, 0, 0], [300.0, 0, 0], [0, 50.0, 0], [10.0, -50.0, 0]])
    t = np.array([[0, 1, 2], [0, 3, 1]])
    n = np.array([1.0, 0.0, 0.0])
    nv, nt, parent = split_triangles_at_plane(v, t, n, 20.0)
    pr = nv[nt] @ n
    assert not ((pr.min(axis=1) < 20.0 - 1e-6) & (pr.max(axis=1) > 20.0 + 1e-6)).any()

    # area and winding survive the cut
    def area(vv: np.ndarray, tt: np.ndarray) -> np.ndarray:
        c = np.cross(vv[tt[:, 1]] - vv[tt[:, 0]], vv[tt[:, 2]] - vv[tt[:, 0]])
        return c[:, 2] / 2

    assert np.isclose(area(nv, nt).sum(), area(v, t).sum())
    assert (np.sign(area(nv, nt)) == np.sign(area(v, t))[parent]).all()
    mesh = mesh_from_arrays(v, t, np.array([0, 0]), with_edges=False)
    cut = cut_mesh_at_planes(mesh, [(n, 20.0)])
    assert len(cut.faces[0].triangles) == len(nt)
    assert np.isclose(cut.faces[0].tri_areas.sum(), mesh.faces[0].tri_areas.sum())
