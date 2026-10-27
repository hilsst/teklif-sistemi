"""Edit library values in place (comments preserved) with validation and a change log.

Paths use the same notation as the TODO report: ``costing.margin_ratio``,
``work_centers[TEMIZ_ODA].labor_rate``, ``operations[kesim].times[ply_count].rate``.
Every write is validated by reloading the whole library; on failure all touched
files are restored. Each change is appended to ``changelog.csv`` with its reason.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from teklif.common.models import Price
from teklif.library.loader import Library, LibraryError, iter_values, load_library

CHANGELOG = "changelog.csv"

# Non-numeric settings the user may switch, with their allowed values.
CHOICES: dict[tuple[str, str], tuple[str, ...]] = {
    ("thresholds.yaml", "costing.margin_basis"): ("markup", "margin"),
    ("thresholds.yaml", "costing.overhead_basis"): ("direct_cost", "labor"),
}


@dataclass(frozen=True)
class EditableValue:
    file: str
    path: str
    desc: str
    unit: str
    value: float | None
    currency: str | None = None


@dataclass(frozen=True)
class Change:
    file: str
    path: str
    value: float | str | None
    currency: str | None = None


def editable_values(lib: Library) -> list[EditableValue]:
    out: list[EditableValue] = []
    for fname, model in lib.files().items():
        if fname == "enums.yaml":
            continue
        for path, v in iter_values(model):
            out.append(
                EditableValue(
                    file=fname,
                    path=path,
                    desc=v.desc or "",
                    unit=v.per if isinstance(v, Price) else v.unit,
                    value=v.value,
                    currency=v.currency if isinstance(v, Price) else None,
                )
            )
    return out


def current_choice(lib: Library, file: str, path: str) -> str:
    if (file, path) not in CHOICES:
        raise KeyError(path)
    node: Any = lib.files()[file]
    for part in path.split("."):
        node = getattr(node, part)
    return str(node)


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _parent_and_key(data: Any, path: str) -> tuple[Any, str]:
    """Walk ``a.b[id].c`` and return (container, final key)."""
    node = data
    parts = path.split(".")
    for seg in parts[:-1]:
        node = _step(node, seg)
    return node, parts[-1]


def _step(node: Any, seg: str) -> Any:
    if "[" not in seg:
        return node[seg]
    name, key = seg[:-1].split("[")
    for item in node[name]:
        if key in (item.get("id"), item.get("material"), item.get("driver")):
            return item
    raise KeyError(seg)


def _apply(data: Any, change: Change) -> tuple[str, str]:
    """Apply one change; return (old, new) as text for the log."""
    parent, key = _parent_and_key(data, change.path)
    if (change.file, change.path) in CHOICES:
        allowed = CHOICES[(change.file, change.path)]
        if change.value not in allowed:
            raise ValueError(f"{change.path}: must be one of {allowed}")
        old = str(parent[key])
        parent[key] = change.value
        return old, str(change.value)
    target = parent[key]
    if not hasattr(target, "get") or "value" not in target:
        raise KeyError(f"{change.path} is not an editable value")
    if change.value is not None and not isinstance(change.value, int | float):
        raise ValueError(f"{change.path}: numeric value expected")
    old = f"{target['value']} {target.get('currency', '')}".strip()
    target["value"] = change.value
    if change.currency is not None and "currency" in target:
        target["currency"] = change.currency
    target["source"] = "user" if change.value is not None else "library"
    if change.value is not None and target.get("note") == "TODO":
        del target["note"]
    new = f"{change.value} {target.get('currency', '')}".strip()
    return old, new


def set_values(root: Path | str, changes: list[Change], reason: str, user: str = "") -> Library:
    """Apply changes, validate the library, log them; restore everything on error."""
    if not reason.strip():
        raise ValueError("A reason is required for library changes")
    root = Path(root)
    y = _yaml()
    originals: dict[Path, bytes] = {}
    log_rows: list[list[str]] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        for file in sorted({c.file for c in changes}):
            path = root / file
            originals[path] = path.read_bytes()
            data = y.load(originals[path].decode("utf-8"))
            for change in (c for c in changes if c.file == file):
                old, new = _apply(data, change)
                if old != new:
                    log_rows.append([now, user, file, change.path, old, new, reason.strip()])
            buf = io.StringIO()
            y.dump(data, buf)
            path.write_text(buf.getvalue(), encoding="utf-8")
        lib = load_library(root)
    except (LibraryError, KeyError, ValueError):
        for path, content in originals.items():
            path.write_bytes(content)
        raise
    log = root / CHANGELOG
    new_file = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["zaman", "kullanici", "dosya", "yol", "eski", "yeni", "gerekce"])
        writer.writerows(log_rows)
    return lib
