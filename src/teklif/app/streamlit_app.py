"""Streamlit UI (Phase 1): manual geometry input, full breakdown, overrides, save, export.

Run with ``teklif app`` (or ``streamlit run src/teklif/app/streamlit_app.py``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from teklif.common.models import Override, Price
from teklif.costing.engine import QuoteInputError, QuoteResult, compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.library import Library, LibraryError, list_todos, load_library
from teklif.library.loader import DEFAULT_LIBRARY_DIR
from teklif.reporting.quote_report import (
    CATEGORY_LABELS,
    flags_text,
    inputs_text,
    render_html,
    summary_rows,
    write_excel,
)
from teklif.storage.db import Store

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "examples" / "ornek_parca.yaml"
DEFAULT_DB = Path(os.environ.get("TEKLIF_DB", REPO / "data" / "teklif.sqlite"))


def breakdown_frame(result: QuoteResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "kalem": ln.key,
                "kategori": CATEGORY_LABELS.get(ln.category, ln.category),
                "açıklama": ln.description,
                "formül": ln.formula,
                "girdiler [kaynak]": inputs_text(ln),
                "hesaplanan": ln.result,
                "ezilen": ln.override.value if ln.override else None,
                "gerekçe": ln.override.reason if ln.override else "",
                "sonuç": ln.final,
                "birim": ln.unit,
                "inceleme": flags_text(ln),
            }
            for ln in result.lines
        ]
    )


def excel_bytes(result: QuoteResult) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_excel(result, Path(tmp) / "teklif.xlsx")
        return path.read_bytes()


def pdf_bytes(result: QuoteResult) -> bytes | None:
    try:
        from weasyprint import HTML  # type: ignore[import-untyped]
    except ImportError:
        return None
    data: bytes = HTML(string=render_html(result)).write_pdf()
    return data


def _example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    return data


def _input_form(lib: Library, base: dict[str, Any]) -> QuoteInput | None:
    procs = list(lib.processes)
    prepregs = [p.id for p in lib.materials.prepregs]
    complexity = list(lib.thresholds.complexity_factors)
    autoclaves = [a.id for a in lib.autoclaves.autoclaves]
    currencies = list(lib.currency.rates)

    c1, c2, c3 = st.columns(3)
    quote_no = c1.text_input("Teklif no", base.get("quote_no", ""))
    part_no = c2.text_input("Parça no", base.get("part_no", ""))
    part_name = c3.text_input("Parça adı", base.get("part_name", ""))
    customer = c1.text_input("Müşteri", base.get("customer", ""))
    process = c2.selectbox(
        "Proses", procs, index=procs.index(base["process"]) if base.get("process") in procs else 0
    )
    template = lib.processes[process]
    cycles = [c.id for c in template.cure_cycles]
    cure = c3.selectbox(
        "Kür çevrimi",
        cycles,
        index=cycles.index(base["cure_cycle"]) if base.get("cure_cycle") in cycles else 0,
    )
    material = c1.selectbox(
        "Ana prepreg",
        prepregs,
        index=prepregs.index(base["material"]) if base.get("material") in prepregs else 0,
    )
    autoclave = c2.selectbox(
        "Autoclave",
        autoclaves,
        index=autoclaves.index(base["autoclave"]) if base.get("autoclave") in autoclaves else 0,
    )
    cx = c3.selectbox(
        "Karmaşıklık",
        complexity,
        index=complexity.index(base["complexity"]) if base.get("complexity") in complexity else 0,
    )

    st.markdown("**Ply grupları** (bir ply'ın düz desen alanı, fire hariç)")
    plies_df = pd.DataFrame(
        base.get("plies") or [{"name": "", "material": None, "area_m2": 1.0, "count": 1}]
    ).reindex(columns=["name", "material", "area_m2", "count"])
    plies_df = st.data_editor(
        plies_df,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("ad"),
            "material": st.column_config.SelectboxColumn("prepreg (boş = ana)", options=prepregs),
            "area_m2": st.column_config.NumberColumn("alan [m2]", min_value=0.0, format="%.4f"),
            "count": st.column_config.NumberColumn("adet", min_value=1, step=1),
        },
        key="plies",
    )

    g = base.get("geometry", {})
    st.markdown("**Geometri** (elle giriş)")
    g1, g2, g3, g4 = st.columns(4)
    geo = {
        "part_area_m2": g1.number_input(
            "Parça alanı [m2]", 0.0, value=float(g.get("part_area_m2", 1.0)), format="%.4f"
        ),
        "part_perimeter_m": g2.number_input(
            "Trim çevresi [m]", 0.0, value=float(g.get("part_perimeter_m", 0.0))
        ),
        "hole_count": g3.number_input("Delik sayısı", 0, value=int(g.get("hole_count", 0))),
        "tool_area_m2": g4.number_input(
            "Kalıp alanı [m2]", 0.0, value=float(g.get("tool_area_m2", 1.0)), format="%.4f"
        ),
        "tool_perimeter_m": g1.number_input(
            "Kalıp çevresi [m]", 0.0, value=float(g.get("tool_perimeter_m", 1.0))
        ),
        "tool_footprint_x_mm": g2.number_input(
            "Kalıp izi X [mm]", 0.0, value=float(g.get("tool_footprint_x_mm", 1000.0))
        ),
        "tool_footprint_y_mm": g3.number_input(
            "Kalıp izi Y [mm]", 0.0, value=float(g.get("tool_footprint_y_mm", 1000.0))
        ),
    }

    st.markdown("**Adet ve opsiyonlar**")
    q1, q2, q3 = st.columns(3)
    quantity = q1.number_input("Sipariş adedi", 1, value=int(base.get("quantity", 1)))
    batch = q2.number_input("Parti adedi", 1, value=int(base.get("batch_size", 1)))
    first = q3.number_input("İlk parça sıra no", 1, value=int(base.get("first_unit_no", 1)))
    optional = [op.id for op in template.operations if op.optional]
    all_ops = [op.id for op in template.operations]
    include = q1.multiselect(
        "Opsiyonel operasyonlar",
        optional,
        default=[o for o in base.get("include_optional", []) if o in optional],
    )
    exclude = q2.multiselect(
        "Hariç tutulan operasyonlar",
        all_ops,
        default=[o for o in base.get("exclude_operations", []) if o in all_ops],
    )

    n1, n2, n3, n4 = st.columns(4)
    nre_base = base.get("tooling_nre") or {}
    tooling_exists = n1.checkbox("Kalıp mevcut (NRE yok)", bool(base.get("tooling_exists")))
    nre_value = n2.number_input(
        "Kalıp + master model NRE", 0.0, value=float(nre_base.get("value") or 0.0)
    )
    nre_ccy = n3.selectbox(
        "NRE para birimi",
        currencies,
        index=currencies.index(nre_base.get("currency", currencies[0]))
        if nre_base.get("currency") in currencies
        else 0,
    )
    amortize = n4.checkbox("NRE birim fiyata yedirilsin", bool(base.get("amortize_nre")))

    try:
        plies = [
            PlyGroup(
                name=str(r["name"] or ""),
                material=r["material"]
                if isinstance(r["material"], str) and r["material"]
                else None,
                area_m2=float(r["area_m2"]),
                count=int(r["count"]),
            )
            for r in plies_df.to_dict("records")
            if pd.notna(r.get("area_m2"))
        ]
        return QuoteInput(
            quote_no=quote_no,
            part_no=part_no,
            part_name=part_name,
            customer=customer,
            process=process,
            material=material,
            cure_cycle=cure,
            autoclave=autoclave,
            complexity=cx,
            plies=plies,
            geometry=PartGeometry(**geo),
            quantity=int(quantity),
            batch_size=int(batch),
            first_unit_no=int(first),
            include_optional=include,
            exclude_operations=exclude,
            tooling_nre=None
            if tooling_exists or nre_value <= 0
            else Price(value=nre_value, currency=nre_ccy, per="pcs"),
            tooling_exists=tooling_exists,
            amortize_nre=amortize,
            overrides=st.session_state.get("overrides", {}),
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Girdi hatası: {exc}")
        return None


def _show_result(result: QuoteResult, lib: Library, store: Store) -> None:
    cols = st.columns(4)
    for col, key, label in zip(
        cols,
        ["cost.unit", "price.unit_final", "nre.tooling", "price.order_total"],
        ["Birim maliyet", "Birim fiyat", "NRE", "Sipariş toplamı"],
        strict=True,
    ):
        try:
            v = result.value(key)
        except KeyError:
            v = None
        col.metric(f"{label} [{result.currency}]", "—" if v is None else f"{v:,.2f}")

    if result.manual_review:
        with st.expander(f"Manuel inceleme: {len(result.review_items)} bayrak", expanded=True):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"kalem": k, "kod": fl.code, "neden": fl.reason, "kütüphane yolu": fl.path}
                        for k, fl in result.review_items
                    ]
                ),
                hide_index=True,
            )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "kalem": r.label,
                    "tutar": r.value,
                    "birim": r.unit,
                    "inceleme": "manual_review" if r.review else "",
                }
                for r in summary_rows(result)
            ]
        ),
        hide_index=True,
    )
    st.dataframe(breakdown_frame(result), hide_index=True, height=480)

    st.markdown("**Ezme** (her ezme gerekçesiyle kaydedilir)")
    keys = [ln.key for ln in result.lines]
    o1, o2, o3, o4 = st.columns([3, 2, 4, 1])
    key = o1.selectbox("Kalem", keys, key="ov_key")
    value = o2.number_input("Yeni değer", value=float(result.value(key) or 0.0), key="ov_val")
    reason = o3.text_input("Gerekçe", key="ov_reason")
    if o4.button("Ez"):
        if not reason.strip():
            st.warning("Gerekçe zorunlu.")
        else:
            st.session_state.setdefault("overrides", {})[key] = Override(
                value=value, reason=reason.strip()
            )
            st.rerun()
    if st.session_state.get("overrides"):
        st.write({k: f"{o.value} ({o.reason})" for k, o in st.session_state["overrides"].items()})
        if st.button("Ezmeleri temizle"):
            st.session_state["overrides"] = {}
            st.rerun()

    d1, d2, d3 = st.columns(3)
    if d1.button("Teklifi kaydet"):
        row = store.save_quote(result, lib)
        st.success(f"Kaydedildi: {row.quote_no} rev {row.revision}")
    d2.download_button("Excel indir", excel_bytes(result), f"{result.input.quote_no}.xlsx")
    pdf = pdf_bytes(result)
    if pdf is None:
        d3.caption("PDF için: uv sync --extra pdf")
    else:
        d3.download_button("PDF indir", pdf, f"{result.input.quote_no}.pdf")


def _saved_tab(store: Store, lib: Library) -> None:
    rows = store.list_quotes()
    if not rows:
        st.info("Kayıtlı teklif yok.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "id": r.id,
                    "teklif": r.quote_no,
                    "rev": r.revision,
                    "parça": r.part_no,
                    "adet": r.quantity,
                    "birim maliyet": r.unit_cost,
                    "birim fiyat": r.unit_price,
                    "toplam": r.order_total,
                    "inceleme": r.manual_review,
                    "tarih": r.created_at,
                }
                for r in rows
            ]
        ),
        hide_index=True,
    )
    qid = st.selectbox(
        "Teklif",
        [r.id for r in rows],
        format_func=lambda i: next(f"{r.quote_no} rev {r.revision}" for r in rows if r.id == i),
    )
    row = store.get(qid)
    result = row.result()
    st.dataframe(breakdown_frame(result), hide_index=True, height=360)
    ovs = store.overrides(qid)
    if ovs:
        st.markdown("**Ezme kaydı**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "kalem": o.line_key,
                        "hesaplanan": o.computed,
                        "ezilen": o.value,
                        "gerekçe": o.reason,
                    }
                    for o in ovs
                ]
            ),
            hide_index=True,
        )

    st.markdown("**Gerçekleşme** (parça başı, teklif para birimi)")
    a1, a2, a3 = st.columns(3)
    actual = a1.number_input("Fiili birim maliyet", 0.0, key="act_cost")
    hours = a2.number_input("Fiili işçilik [adam-saat/parça]", 0.0, key="act_h")
    note = a3.text_input("Not", key="act_note")
    if st.button("Gerçekleşmeyi kaydet") and actual > 0:
        store.add_actual(qid, actual, labor_hours=hours or None, note=note)
    tol = lib.thresholds.acceptance.cost_tolerance.value
    for a in store.actuals(qid):
        dev = None if row.unit_cost is None else (row.unit_cost - a.unit_cost) / a.unit_cost
        ok = "" if dev is None or tol is None else ("✓" if abs(dev) <= tol else "✗")
        st.write(
            f"{a.recorded_at:%Y-%m-%d}: fiili {a.unit_cost:,.2f}, "
            f"sapma {'—' if dev is None else f'{dev:+.1%}'} {ok} {a.note}"
        )


def main() -> None:
    st.set_page_config(page_title="Teklif sistemi", layout="wide")
    st.title("Kompozit parça teklifi")
    lib_dir = st.sidebar.text_input("Kütüphane klasörü", str(DEFAULT_LIBRARY_DIR))
    db_path = Path(st.sidebar.text_input("Veritabanı", str(DEFAULT_DB)))
    try:
        lib = load_library(lib_dir)
    except LibraryError as exc:
        st.error(str(exc))
        return
    st.sidebar.caption(f"Kütüphanede {len(list_todos(lib))} TODO değer var.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)

    new_tab, saved_tab = st.tabs(["Yeni teklif", "Kayıtlı teklifler"])
    with new_tab:
        base = _example() if st.toggle("Örnek parçayla doldur", value=True) else {}
        inp = _input_form(lib, base)
        if inp is not None:
            try:
                result = compute_quote(inp, lib)
            except (QuoteInputError, ValueError) as exc:
                st.error(f"Girdi hatası: {exc}")
            else:
                _show_result(result, lib, store)
    with saved_tab:
        _saved_tab(store, lib)


if __name__ == "__main__":
    main()
