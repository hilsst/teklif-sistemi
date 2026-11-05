"""Keep the user's library values outside the program folder.

The library lives next to the program, and Kemal updates by replacing that folder (only ``data``
is carried over). Every saved value is therefore also written to ``data/library_values.json``;
at start-up the values that the library no longer holds are written back.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from teklif.library.editor import CHOICES, Change, current_choice, editable_values, set_values
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


def save_values(
    root: Path | str, changes: list[Change], reason: str, overlay: Path | None
) -> Library:
    """``set_values`` + remember the change (a failed remember never hides the save)."""
    lib = set_values(root, changes, reason)
    if overlay is not None:
        with contextlib.suppress(OSError):
            record(overlay, changes)
    return lib


def restore(root: Path | str, path: Path, lib: Library) -> Library:
    """Write back remembered values that the library does not hold (after an update).

    Each entry is applied on its own, so one that no longer validates is skipped."""
    saved = _read(path)
    if not saved:
        return lib
    current = {f"{v.file}|{v.path}": v for v in editable_values(lib)}
    done = False
    for key, entry in saved.items():
        file, _, p = key.partition("|")
        val = entry.get("value")
        ccy = entry.get("currency")
        ccy = ccy if isinstance(ccy, str) else None
        try:
            if (file, p) in CHOICES:
                if not isinstance(val, str) or current_choice(lib, file, p) == val:
                    continue
                change = Change(file, p, val)
            else:
                cur = current.get(key)
                if cur is None or not isinstance(val, int | float):
                    continue
                if cur.value == val and (ccy is None or cur.currency == ccy):
                    continue
                change = Change(file, p, float(val), ccy)
            set_values(root, [change], RESTORE_REASON)
            done = True
        except (ValueError, KeyError, OSError):
            continue
    return load_library(root) if done else lib
