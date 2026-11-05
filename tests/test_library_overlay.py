import shutil
from pathlib import Path

from teklif.app.library_relevance import is_needed, needed_paths
from teklif.library import Library, load_library
from teklif.library.editor import Change, EditableValue, editable_values, set_values
from teklif.library.overlay import record, restore

SRC = Path(__file__).resolve().parents[1] / "library"


def _first_todo(lib: Library) -> EditableValue:
    return next(v for v in editable_values(lib) if v.value is None and v.currency is None)


def test_saved_value_persists_and_survives_library_replacement(tmp_path: Path) -> None:
    root = tmp_path / "library"
    shutil.copytree(SRC, root)
    lib = load_library(root)
    v = _first_todo(lib)
    ch = Change(v.file, v.path, 12.5)
    set_values(root, [ch], "deneme")
    record(tmp_path / "data" / "values.json", [ch])
    # a fresh load of the same folder sees the value
    after = {(x.file, x.path): x.value for x in editable_values(load_library(root))}
    assert after[(v.file, v.path)] == 12.5
    # the program folder is replaced by a fresh copy (zip update): value is gone ...
    shutil.rmtree(root)
    shutil.copytree(SRC, root)
    fresh = load_library(root)
    assert {(x.file, x.path): x.value for x in editable_values(fresh)}[(v.file, v.path)] is None
    # ... and comes back from data/
    back = restore(root, tmp_path / "data" / "values.json", fresh)
    assert {(x.file, x.path): x.value for x in editable_values(back)}[(v.file, v.path)] == 12.5


def test_filled_needed_value_stays_in_needed_view() -> None:
    lib = load_library(SRC)
    needed = needed_paths(lib, "sandvic_toplu")
    v = next(x for x in editable_values(lib) if is_needed(x, needed) and x.value is None)
    assert is_needed(v, needed)  # relevance does not depend on the value being empty


def test_needed_paths_unchanged_after_filling(tmp_path: Path) -> None:
    root = tmp_path / "library"
    shutil.copytree(SRC, root)
    lib = load_library(root)
    before = needed_paths(lib, "sandvic_toplu")
    v = next(x for x in editable_values(lib) if x.value is None and f"{x.file}:{x.path}" in before)
    set_values(root, [Change(v.file, v.path, 3.0, v.currency)], "deneme")
    assert f"{v.file}:{v.path}" in needed_paths(load_library(root), "sandvic_toplu")
