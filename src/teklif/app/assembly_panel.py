"""Product tree of an assembly STEP and the choice of which part to price or learn from.

Standard parts (fasteners and the like) are grouped with their count and are not priced;
special-manufacture parts (metal, composite) stay as rows that can be priced one by one. The user
can flip a part between the two, leave it out, and give parts a tool group (parts of one group
share one tool, decided by the user).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from teklif.geometry.assembly_tree import AssemblyTree, read_assembly
from teklif.ingest.step_props import StepUnavailableError


@dataclass(frozen=True)
class PartChoice:
    product: str
    name: str
    solids: tuple[int, ...]
    count: int  # occurrences in the assembly
    tool_group: str
    rows: list[dict[str, object]]  # the whole assembly table, for the summary

    @property
    def label(self) -> str:
        return self.product if self.name == self.product else f"{self.product} ({self.name})"


def _read(data: bytes) -> AssemblyTree | None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "asm.step"
        path.write_bytes(data)
        try:
            return read_assembly(path)
        except StepUnavailableError:
            return None


def _tree(data: bytes) -> AssemblyTree | None:
    key = "asm_tree_" + hashlib.sha256(data).hexdigest()
    if key not in st.session_state:
        st.session_state[key] = _read(data)
    return st.session_state[key]  # type: ignore[no-any-return]


def render_assembly_panel(data: bytes, key: str) -> PartChoice | None:
    """Show the part table of an assembly; return the part chosen for pricing/learning.

    Returns None for a single-part file or when no special part is left to choose.
    """
    tree = _tree(data)
    if tree is None or not tree.is_assembly or not tree.parts:
        return None
    key = f"{key}_{hashlib.sha256(data).hexdigest()[:12]}"
    st.markdown(f"**Montaj ürün ağacı:** {tree.n_solids} gövde, {len(tree.parts)} farklı parça")
    seed = pd.DataFrame(
        [
            {
                "parça no": g.product,
                "ad": g.name,
                "adet": g.count,
                "hacim [cm³]": round(g.volume_mm3 / 1000, 1),
                "tür": "standart" if g.standard else "özel imalat",
                "kalıp grubu": "",
                "teklife dahil": not g.standard,
            }
            for g in tree.parts
        ]
    )
    table = st.data_editor(
        seed,
        hide_index=True,
        key=f"{key}_table",
        disabled=["parça no", "ad", "adet", "hacim [cm³]"],
        column_config={
            "tür": st.column_config.SelectboxColumn(options=["özel imalat", "standart"]),
            "teklife dahil": st.column_config.CheckboxColumn(),
        },
    )
    std = table[table["tür"] == "standart"]
    if len(std):
        st.caption(
            "Standart parçalar (fiyatlanmaz, adetle sayılır): "
            + ", ".join(f"{r['parça no']} ×{r['adet']}" for _, r in std.iterrows())
        )
    with st.expander("Ürün ağacı (alt montajlarla)"):
        st.code("\n".join(tree.lines), language=None)
    if tree.unmatched:
        st.caption(f"{tree.unmatched} gövde ürün ağacıyla eşleşmedi (listede görünmez).")
    pickable = table[(table["tür"] == "özel imalat") & table["teklife dahil"]]
    if pickable.empty:
        st.info("Fiyatlanacak özel imalat parça yok; tabloda bir parçayı 'özel imalat' yapın.")
        return None
    by_no = {g.product: g for g in tree.parts}
    labels = {
        r["parça no"]: f"{r['parça no']} · {r['ad']} (×{r['adet']})" for _, r in pickable.iterrows()
    }
    chosen = st.selectbox(
        "Fiyatlanacak / öğrenme verisi girilecek parça",
        list(labels),
        format_func=lambda p: labels[p],
        key=f"{key}_pick",
    )
    g = by_no[chosen]
    row = table[table["parça no"] == chosen].iloc[0]
    return PartChoice(
        product=g.product,
        name=g.name,
        solids=tuple(g.solids),
        count=g.count,
        tool_group=str(row["kalıp grubu"] or "").strip(),
        rows=[{str(k): v for k, v in r.items()} for r in table.to_dict("records")],
    )
