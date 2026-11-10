"""STEP reading with OCP (``cadquery-ocp``, optional: ``pip install kalip-analiz[cad]``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class StepUnavailableError(RuntimeError):
    """OCP is not installed or the file cannot be read."""


def load_step_shape(path: Path | str) -> tuple[Any, list[str]]:
    """Read a STEP file; return the OCCT shape and the file's length units."""
    try:
        from OCP.collections import Sequence_TCollection_AsciiString
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise StepUnavailableError(f"cannot read STEP file {path}")
    length, angle, solid_angle = (Sequence_TCollection_AsciiString() for _ in range(3))
    reader.FileUnits(length, angle, solid_angle)
    units = [length.Value(i + 1).ToCString() for i in range(length.Length())]
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise StepUnavailableError("STEP file contains no shape")
    return shape, units
