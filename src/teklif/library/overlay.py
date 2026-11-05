"""Keep the user's library values outside the program folder.

The library lives next to the program, and Kemal updates by replacing that folder (only ``data``
is carried over). Every saved value is therefore also written to ``data/library_values.json``;
at start-up the values that the library no longer holds are written back.
"""

from __future__ import annotations

import json
from pathlib import Path

from teklif.library.editor import Change, editable_values, set_values
from teklif.library.loader import Library, load_library

RESTORE_REASON = "güncelleme sonrası kayıtlı değerler geri yüklendi"


def _read(path: Path) -> dict[str, dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record(path: Path, changes: list[Change]) -> None:
    """Remember saved changes (latest wins); an emptied value is forgotten."""
    data = _read(path)
    for c in changes:
        key = f"{c.file}|{c.path}"
        if c.value is None:
            data.pop(key, None)
        else:
            data[key] = {"value": c.value, "currency": c.currency}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def restore(root: Path | str, path: Path, lib: Library) -> Library:
    """Write back remembered values that the library does not hold (after an update)."""
    saved = _read(path)
    if not saved:
        return lib
    current = {f"{v.file}|{v.path}": v for v in editable_values(lib)}
    changes = []
    for key, entry in saved.items():
        cur = current.get(key)
        val = entry.get("value")
        if cur is None or not isinstance(val, int | float) or cur.value == val:
            continue  # path no longer exists, or already restored
        ccy = entry.get("currency")
        file, _, p = key.partition("|")
        changes.append(Change(file, p, float(val), ccy if isinstance(ccy, str) else None))
    if not changes:
        return lib
    try:
        set_values(root, changes, RESTORE_REASON)
    except (ValueError, KeyError, OSError):
        return lib
    return load_library(root)
