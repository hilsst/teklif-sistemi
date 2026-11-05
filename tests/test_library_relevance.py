from pathlib import Path

from teklif.app.library_relevance import is_needed, needed_paths
from teklif.app.surface_panel import mold_submesh
from teklif.library import load_library
from teklif.library.editor import editable_values

LIB = Path(__file__).resolve().parents[1] / "library"


def test_needed_is_a_strict_subset_of_todo() -> None:
    lib = load_library(LIB)
    needed = needed_paths(lib, "sandvic_toplu")
    todo = [v for v in editable_values(lib) if v.value is None]
    shown = [v for v in todo if is_needed(v, needed)]
    assert shown
    assert len(shown) < len(todo)
    assert any(v.path.endswith("hourly_rate") for v in shown)


def test_mold_submesh_renumbers() -> None:
    import numpy as np

    from teklif.geometry.surface_select import FaceMesh, MeshFace

    def face(i: int) -> MeshFace:
        return MeshFace(
            i,
            1.0,
            True,
            np.zeros((3, 3)),
            np.array([[0, 1, 2]]),
            np.array([[0, 0, 1.0]]),
            np.array([0.5]),
        )

    sub = mold_submesh(FaceMesh([face(0), face(1), face(2)]), [2])
    assert [f.id for f in sub.faces] == [0]
