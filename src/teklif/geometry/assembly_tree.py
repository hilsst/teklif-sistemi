"""Product tree of an assembly STEP file: parts, how many times they occur, which solids they are.

The tree comes from the STEP product structure (names and nesting). Each leaf part is matched to
the solid indices that ``load_step_shape`` / the surface mesh use, so choosing a part in the tree
selects exactly its solids everywhere else (3D view, measurements, learning data, quote).
Standard parts (fasteners and the like) are grouped with a count; special-manufacture parts stay
separate so they can be priced one by one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from teklif.ingest.step_props import StepUnavailableError, load_step_shape, solid_list

# product number prefix used for standard parts at Kemal's company (ST0003216), plus common words
STANDARD_PREFIX = re.compile(r"^(ST|STD)\d", re.IGNORECASE)
STANDARD_WORDS = re.compile(
    r"civata|cıvata|somun|rondela|pul\b|vida|perçin|percin|pim\b|segman|conta|"
    r"bolt|nut\b|washer|screw|rivet|\bpin\b|\bdin\b|\biso\b|\bm\d+x",
    re.IGNORECASE,
)


@dataclass
class Instance:
    path: str  # "TP0007903 / TP0103679.2 / ST0003216.3"
    solids: list[int]


@dataclass
class PartGroup:
    """All occurrences of one part (same product) in the assembly."""

    product: str
    name: str
    instances: list[Instance] = field(default_factory=list)
    volume_mm3: float = 0.0  # one occurrence
    standard: bool = False

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def solids(self) -> list[int]:
        """Solids of one representative occurrence."""
        return self.instances[0].solids if self.instances else []


@dataclass
class AssemblyTree:
    parts: list[PartGroup]
    n_solids: int
    lines: list[str]  # indented text outline of the product tree
    unmatched: int = 0  # leaf solids that could not be matched to a solid index

    @property
    def is_assembly(self) -> bool:
        return self.n_solids > 1


def is_standard(product: str, name: str) -> bool:
    return bool(
        STANDARD_PREFIX.match(product.strip())
        or STANDARD_WORDS.search(name)
        or STANDARD_WORDS.search(product)
    )


def _signature(shape: Any) -> tuple[float, tuple[float, float, float]]:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vp)
    c = vp.CentreOfMass()
    return vp.Mass(), (c.X(), c.Y(), c.Z())


class _Matcher:
    """Match located solids of the product tree to the indices of the plain STEP reader."""

    def __init__(self, shape: Any) -> None:
        self.sigs = [_signature(s) for s in solid_list(shape)]
        self.used: set[int] = set()

    def match(self, shape: Any) -> list[int]:
        out: list[int] = []
        for s in solid_list(shape):
            vol, com = _signature(s)
            best, best_d = -1, 1e30
            for i, (v, c) in enumerate(self.sigs):
                if i in self.used or abs(v - vol) > 1e-3 * max(abs(vol), 1.0):
                    continue
                d = sum((a - b) ** 2 for a, b in zip(c, com, strict=True)) ** 0.5
                if d < best_d:
                    best, best_d = i, d
            if best >= 0 and best_d < 0.5 + 1e-3 * max(abs(vol) ** (1 / 3), 1.0):
                self.used.add(best)
                out.append(best)
        return out


def read_assembly(path: Path | str) -> AssemblyTree:
    """Read the product structure of a STEP file (needs OCP)."""
    try:
        from OCP.collections import Sequence_TDF_Label
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDF import TDF_Label
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopLoc import TopLoc_Location
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
    except ImportError as exc:
        raise StepUnavailableError("OCP (cadquery-ocp) is not installed") from exc

    plain, _ = load_step_shape(path)
    matcher = _Matcher(plain)
    n_solids = len(matcher.sigs)

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise StepUnavailableError(f"cannot read STEP file {path}")
    reader.Transfer(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def name_of(label: Any) -> str:
        n = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), n):
            return str(n.Get().ToExtString()).strip()
        return ""

    groups: dict[str, PartGroup] = {}
    lines: list[str] = []

    def walk(label: Any, loc: Any, path: list[str], depth: int) -> None:
        target = TDF_Label()
        comp_loc = TopLoc_Location()
        if XCAFDoc_ShapeTool.IsReference_s(label):
            XCAFDoc_ShapeTool.GetReferredShape_s(label, target)
            comp_loc = XCAFDoc_ShapeTool.GetLocation_s(label)
        else:
            target = label
        total = loc.Multiplied(comp_loc)
        product = name_of(target) or name_of(label) or "?"
        inst_name = name_of(label) or product
        children = Sequence_TDF_Label()
        XCAFDoc_ShapeTool.GetComponents_s(target, children)
        is_asm = XCAFDoc_ShapeTool.IsAssembly_s(target) and children.Length() > 0
        lines.append(
            "  " * depth + (inst_name if inst_name == product else f"{inst_name} ({product})")
        )
        if is_asm:
            for i in range(children.Length()):
                walk(children.Value(i + 1), total, [*path, inst_name], depth + 1)
            return
        shape = XCAFDoc_ShapeTool.GetShape_s(target).Moved(total)
        idx = matcher.match(shape)
        g = groups.setdefault(product, PartGroup(product=product, name=product))
        if inst_name != product and g.name == product and not inst_name.startswith(product):
            g.name = inst_name.rsplit(".", 1)[0] if inst_name[-2:-1] == "." else inst_name
        if not g.instances and idx:
            g.volume_mm3 = sum(_signature(s)[0] for s in solid_list(shape))
        g.instances.append(Instance(" / ".join([*path, inst_name]), idx))

    roots = Sequence_TDF_Label()
    tool.GetFreeShapes(roots)
    for i in range(roots.Length()):
        walk(roots.Value(i + 1), TopLoc_Location(), [], 0)

    parts = [g for g in groups.values() if any(inst.solids for inst in g.instances)]
    for g in parts:
        g.standard = is_standard(g.product, g.name)
    unmatched = n_solids - len(matcher.used)
    parts.sort(key=lambda g: (g.standard, -g.volume_mm3))
    return AssemblyTree(parts=parts, n_solids=n_solids, lines=lines, unmatched=unmatched)
