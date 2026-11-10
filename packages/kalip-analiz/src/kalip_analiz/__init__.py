"""kalip-analiz: automatic pull direction, split-mold and core (maça) decision for a mold surface.

    from kalip_analiz import analyze
    r = analyze("part.step", faces=[3, 4, 7])        # face ids of the mold surface
    print(r.summary())
    r.export_html("sonuc.html")                       # 3D map with numbered cores

Thresholds come from YAML (``thresholds.yaml``), never from code.
"""

from kalip_analiz.api import AnalysisResult, analyze
from kalip_analiz.mesh import FaceMesh, load_face_mesh, mesh_from_arrays
from kalip_analiz.pull_search import Solution
from kalip_analiz.settings import load_settings

__all__ = [
    "AnalysisResult",
    "FaceMesh",
    "Solution",
    "analyze",
    "load_face_mesh",
    "load_settings",
    "mesh_from_arrays",
]
__version__ = "0.1.0"
