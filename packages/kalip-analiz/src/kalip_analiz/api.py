"""Single entry point: ``analyze(step_file, faces)`` -> pull direction, split and core decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from kalip_analiz import pull_search as ps
from kalip_analiz import viewer as vw
from kalip_analiz.mesh import FaceMesh, faces_facing, load_face_mesh, mesh_from_arrays
from kalip_analiz.settings import AnalysisSettings, ThresholdSource, load_settings

AXES: dict[str, tuple[float, float, float]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the analysis found. ``assessment`` holds the decision, the compared methods
    (``assessment.methods``, cheapest first) and the traceable ``assessment.trace``."""

    mesh: FaceMesh
    faces: list[int]
    tris: ps.TriSet
    pull: tuple[float, float, float]  # single-piece pull direction that was judged
    directions: list[ps.DirectionScore]  # optimised single-piece candidates, best first
    assessment: ps.ToolingAssessment
    settings: AnalysisSettings

    @property
    def solution(self) -> ps.Solution:
        return self.assessment.solution

    @property
    def manual_review(self) -> bool:
        return self.assessment.manual_review

    @property
    def method(self) -> ps.Method | None:
        """The shown method (the system's recommendation unless ``method=`` picked another)."""
        return self.assessment.chosen

    @property
    def cores(self) -> list[dict[str, Any]]:
        """Numbered cores (M1..) and unresolved regions (Ç1..) of the shown method."""
        m = self.method
        return vw.core_rows(m) if m is not None else []

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready result for a web host: decision, reasons, flags, compared methods, pieces,
        cores and the traceable calculation lines (formula, inputs, sources)."""
        a = self.assessment
        m = self.method
        return {
            "solution": a.solution.value,
            "label": a.label,
            "manual_review": a.manual_review,
            "pull": list(self.pull),
            "reasons": list(a.reasons),
            "alternatives": list(a.alternatives),
            "flags": [fl.model_dump() for fl in a.flags],
            "max_pieces": a.max_pieces,
            "shown": a.shown,
            "recommended": 0,
            "methods": [
                {
                    "label": x.label,
                    "kind": x.kind,
                    "pieces": x.n_pieces,
                    "cores": x.n_cores,
                    "unresolved": x.n_unresolved,
                    "core_share": x.core_share,
                    "cost": x.cost,
                }
                for x in a.methods
            ],
            "pieces": [
                {
                    "area_ratio": p.area_ratio,
                    "pull": list(p.pull.direction),
                    "min_draft_deg": p.pull.min_draft_deg,
                    "cores": len(p.cores),
                    "unresolved": len(p.unresolved),
                }
                for p in (m.pieces if m is not None else [])
            ],
            "planes": [
                {"normal": list(pl.normal), "offset": pl.offset}
                for pl in (m.planes if m is not None else [])
            ],
            "cores": self.cores,
            "legend_html": vw.piece_legend(m) if m is not None else "",
            "trace": [ln.model_dump(mode="json") for ln in a.trace],
        }

    def summary(self) -> str:
        a = self.assessment
        return "\n".join([a.label, *(f"- {r}" for r in a.reasons)])

    def viewer_payload(self, paint: str = "pieces", view: FaceMesh | None = None) -> dict[str, Any]:
        """Payload for the three.js viewer. ``paint``: ``"pieces"`` (mold pieces, cores, parting
        walls) or ``"draft"`` (draft map for the single-piece pull). ``view``: a coarser mesh of
        the same STEP for the screen (default: the analysis mesh)."""
        v = view or self.mesh
        sp, ss = self.settings.search, self.settings.viewer
        m = self.method
        extra = None
        if paint == "pieces" and m is not None:
            colors: list[Any] = vw.piece_colors(v, self.faces, self.tris, m)
            extra = vw.analysis_extra(self.tris, m, ss)
        else:
            colors = vw.triangle_colors(v, self.faces, self.pull, sp.min_ok_deg, sp.warn_min_deg)
        return vw.viewer_payload(
            v,
            colors,
            "kalip",
            self.pull,
            extra,
            selected=self.faces,
            sync_ms=round(ss.select_pause_s * 1000),
        )

    def side_html(self) -> str:
        m = self.method
        parts = [f"<b>{self.assessment.label}</b>"]
        parts += [f"<div>• {r}</div>" for r in self.assessment.reasons]
        if m is not None:
            parts.append(f"<p>{vw.piece_legend(m)}</p>")
            parts.append(vw.rows_html(self.cores))
        return "".join(parts)

    def export_html(self, path: str | Path, paint: str = "pieces") -> Path:
        """One self-contained HTML file: 3D map (draft / pieces / parting walls / numbered cores)
        with the decision and the core table beside it. Opens in any browser."""
        return vw.export_html(self.viewer_payload(paint), path, self.side_html())


def analyze(
    step_file: str | Path | FaceMesh,
    faces: Sequence[int] | str | tuple[float, float, float] | None = None,
    *,
    thresholds: ThresholdSource = None,
    pull: Sequence[float] | None = None,
    max_pieces: int | None = None,
    method: int = 0,
    facing_tol_deg: float = 30.0,
) -> AnalysisResult:
    """Find the best pull direction for the mold surface and decide single piece / cores / split.

    ``step_file``: a STEP path (needs ``kalip-analiz[cad]``) or a ready ``FaceMesh``
    (``mesh_from_arrays`` for any other mesh source).
    ``faces``: B-rep face ids that form the mold surface (the side of the customer's part surface
    that sits on the mold); or an axis name such as ``"+Z"`` / a vector to take every face
    looking that way (within ``facing_tol_deg``); ``None``: all faces.
    ``thresholds``: path to a YAML file, a parsed dict, or None. Only the keys you give replace
    the packaged ``thresholds.yaml``; nothing is hard-coded.
    ``pull``: judge this single-piece direction instead of the optimised best.
    ``max_pieces``: most mold pieces allowed (default ``undercut.max_split_pieces``).
    ``method``: index into ``result.assessment.methods`` to describe instead of the cheapest.
    """
    cfg = load_settings(thresholds)
    mesh = (
        step_file if isinstance(step_file, FaceMesh) else load_face_mesh(step_file, with_edges=True)
    )
    if faces is None:
        ids = [f.id for f in mesh.faces if len(f.triangles)]
    elif isinstance(faces, str):
        ids = faces_facing(mesh, np.array(AXES[faces]), facing_tol_deg)
    elif len(faces) == 3 and all(isinstance(x, float) for x in faces):
        ids = faces_facing(mesh, np.asarray(faces, dtype=float), facing_tol_deg)
    else:
        ids = [int(i) for i in faces]
    ids = sorted(set(ids))
    if not ids:
        raise ValueError("no faces selected")
    s = cfg.search
    if max_pieces is not None:
        s = replace(s, max_split_pieces=max(1, int(max_pieces)))
    tris = ps.triset_from_mesh(mesh, ids)
    found = ps.search_directions(tris.normals, tris.areas, s)
    if pull is None:
        d = found[0].direction
    else:
        v = np.asarray(pull, dtype=float)
        u = v / np.linalg.norm(v)
        d = (float(u[0]), float(u[1]), float(u[2]))
    a = ps.assess(tris, np.array(d), s)
    if method:
        a = a.view(method)
    return AnalysisResult(mesh, ids, tris, d, found, a, replace(cfg, search=s))


__all__ = ["AXES", "AnalysisResult", "analyze", "mesh_from_arrays"]
