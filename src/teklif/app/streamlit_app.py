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

from teklif.app import data_entry, learning_tab, part_quote
from teklif.app.library_relevance import is_needed, needed_paths
from teklif.common.models import Override, Price, Source
from teklif.costing.engine import QuoteInputError, QuoteResult, compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.costing.quote_set import (
    QuoteSetInput,
    QuoteSetResult,
    blueprint_skeleton,
    compute_quote_set,
)
from teklif.costing.sweep import quantity_sweep
from teklif.library import Library, LibraryError, list_todos, load_library
from teklif.library.editor import CHOICES, Change, current_choice, editable_values, set_values
from teklif.library.loader import DEFAULT_LIBRARY_DIR
from teklif.reporting.csv_export import breakdown_csv, set_breakdown_csv, sweep_csv
from teklif.reporting.quote_report import (
    CATEGORY_LABELS,
    flags_text,
    inputs_text,
    pdf_bytes,
    set_item_rows,
    set_pdf_bytes,
    set_total_rows,
    summary_rows,
    write_excel,
    write_set_excel,
)
from teklif.storage.db import NotFinalizableError, Store

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "examples" / "ornek_parca.yaml"
DEFAULT_DB = Path(os.environ.get("TEKLIF_DB", REPO / "data" / "teklif.sqlite"))


NUMBER_COLUMNS = {
    c: st.column_config.NumberColumn(format="%.2f") for c in ("hesaplanan", "ezilen", "sonuç")
}


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


def _example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    return data


def _nre_price(value: float, currency: str, base: dict[str, Any]) -> Price:
    """Keep the catalog price (with its source note) while untouched; an edit makes it USER."""
    if (
        base
        and base.get("currency") == currency
        and abs(float(base.get("value") or 0) - value) < 1e-9
    ):
        return Price.model_validate(base)
    return Price(value=value, currency=currency, per="pcs", source=Source.USER)


def _geometry_source(base_geo: dict[str, Any], geo: dict[str, Any]) -> Source:
    """Keep the catalog's source while the values are untouched; any edit makes it USER."""
    untouched = "source" in base_geo and all(
        abs(float(base_geo.get(k, float("nan"))) - float(v)) < 1e-9 for k, v in geo.items()
    )
    return Source(base_geo["source"]) if untouched else Source.USER


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
    w1, w2 = st.columns([1, 2])
    wage_factor = w1.number_input(
        "Ücret çarpanı (tüm işçilik)",
        min_value=0.01,
        value=float(base.get("wage_factor", 1.0)),
        step=0.05,
        help="1,00 = kütüphane ücretleri. Rekabet veya stratejik önem için değiştirin.",
    )
    wage_reason = w2.text_input(
        "Ücret çarpanı gerekçesi (çarpan 1 değilse zorunlu)",
        value=str(base.get("wage_factor_reason", "")),
    )
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
    master_value = n2.number_input(
        "Master model NRE (ayrı satır; 0 = yukarıdaki tutara dahil)",
        0.0,
        value=float((base.get("master_nre") or {}).get("value") or 0.0),
    )
    master_base = base.get("master_nre") or {}
    master_ccy = n3.selectbox(
        "Master model para birimi",
        currencies,
        index=currencies.index(master_base["currency"])
        if master_base.get("currency") in currencies
        else 0,
    )

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
            geometry=PartGeometry(**geo, source=_geometry_source(g, geo)),
            quantity=int(quantity),
            batch_size=int(batch),
            first_unit_no=int(first),
            review_notes=list(base.get("review_notes", [])),
            wage_factor=float(wage_factor),
            wage_factor_reason=wage_reason,
            include_optional=include,
            exclude_operations=exclude,
            tooling_nre=None
            if tooling_exists or nre_value <= 0
            else _nre_price(nre_value, nre_ccy, nre_base),
            master_nre=None
            if tooling_exists or master_value <= 0
            else _nre_price(master_value, master_ccy, master_base),
            tooling_exists=tooling_exists,
            amortize_nre=amortize,
            overrides=st.session_state.get("overrides", {}),
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Girdi hatası: {exc}")
        return None


def _draft_banner(manual_review: bool, n_flags: int) -> None:
    if manual_review:
        st.error(
            f"TASLAK: {n_flags} manuel inceleme bayrağı açık. Bayraklar çözülmeden "
            "(eksik değer girilmeden veya gerekçeyle ezilmeden) teklif kesinleştirilemez; "
            "PDF'ler TASLAK filigranıyla çıkar."
        )


def _save_buttons(save_draft: Any, save_final: Any, blocked: bool, scope: str = "") -> None:
    """Draft/final save buttons; ``save_*`` are callables that store and return a row."""
    b1, b2 = st.columns(2)
    if b1.button("Taslak olarak kaydet", key=f"{scope}save_draft"):
        row = save_draft()
        st.success(f"Taslak kaydedildi: {row.quote_no} rev {row.revision}")
    if b2.button(
        "Kesinleştir ve kaydet",
        disabled=blocked,
        help="Açık bayrak varken kapalı",
        key=f"{scope}save_final",
    ):
        try:
            row = save_final()
        except NotFinalizableError as exc:
            st.error(str(exc))
        else:
            st.success(f"Kesin teklif kaydedildi: {row.quote_no} rev {row.revision}")


def _sweep_section(result: QuoteResult, lib: Library, scope: str = "") -> None:
    with st.expander("Adet taraması (birim fiyat adetle nasıl değişir)"):
        default = ",".join(str(q) for q in lib.thresholds.costing.sweep_quantities)
        text = st.text_input("Adetler (virgülle)", default, key=f"{scope}sweep_qtys")
        try:
            qtys = [int(q) for q in text.split(",") if q.strip()]
            rows = quantity_sweep(result.input, lib, qtys)
        except ValueError as exc:
            st.warning(f"Geçersiz adet listesi: {exc}")
            return
        frame = pd.DataFrame(
            [
                {
                    "adet": r.quantity,
                    "parti": r.batch_size,
                    "birim maliyet": r.unit_cost,
                    "birim fiyat": r.unit_price,
                    "sipariş toplamı": r.order_total,
                    "etkin birim fiyat": r.effective_unit_price,
                    "inceleme": "manual_review" if r.manual_review else "",
                }
                for r in rows
            ]
        )
        st.dataframe(
            frame,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.2f")
                for c in ("birim maliyet", "birim fiyat", "sipariş toplamı", "etkin birim fiyat")
            },
        )
        chart = frame.set_index("adet")[["birim maliyet", "birim fiyat", "etkin birim fiyat"]]
        if chart.notna().any().any():
            st.line_chart(chart)
        st.download_button(
            "Adet taraması CSV",
            sweep_csv(rows, result.currency).encode("utf-8"),
            f"{result.input.quote_no}_adet_taramasi.csv",
            key=f"{scope}sweep_csv",
        )


def _show_result(result: QuoteResult, lib: Library, store: Store, scope: str = "new") -> None:
    """Result view; ``scope`` keeps widget keys unique when several tabs show a result."""
    sc = f"{scope}_"
    cols = st.columns(4)
    nre_key = "nre.total" if any(ln.key == "nre.total" for ln in result.lines) else "nre.tooling"
    for col, key, label in zip(
        cols,
        ["cost.unit", "price.unit_final", nre_key, "price.order_total"],
        ["Birim maliyet", "Birim fiyat", "NRE", "Sipariş toplamı"],
        strict=True,
    ):
        try:
            v = result.value(key)
        except KeyError:
            v = None
        col.metric(f"{label} [{result.currency}]", "—" if v is None else f"{v:,.2f}")

    _draft_banner(result.has_open_review, len(result.open_review_items))
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
        column_config={"tutar": st.column_config.NumberColumn(format="%.2f")},
    )
    st.dataframe(breakdown_frame(result), hide_index=True, height=480, column_config=NUMBER_COLUMNS)

    if scope == "new":
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
            st.write(
                {k: f"{o.value} ({o.reason})" for k, o in st.session_state["overrides"].items()}
            )
            if st.button("Ezmeleri temizle"):
                st.session_state["overrides"] = {}
                st.rerun()

    else:
        st.caption("Ezme (override) yalnız 'Yeni teklif' sekmesinde yapılır.")

    _sweep_section(result, lib, sc)
    _save_buttons(
        lambda: store.save_quote(result, lib),
        lambda: store.save_quote(result, lib, final=True),
        result.has_open_review,
        sc,
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Excel indir", excel_bytes(result), f"{result.input.quote_no}.xlsx", key=f"{sc}xlsx"
    )
    d2.download_button(
        "PDF indir", pdf_bytes(result), f"{result.input.quote_no}.pdf", key=f"{sc}pdf"
    )
    d3.download_button(
        "CSV indir (kırılım)",
        breakdown_csv(result).encode("utf-8"),
        f"{result.input.quote_no}.csv",
        key=f"{sc}csv",
    )


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
                    "durum": "KESİN" if r.status == "final" else "taslak",
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
    st.dataframe(breakdown_frame(result), hide_index=True, height=360, column_config=NUMBER_COLUMNS)
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


def _set_tab(lib: Library, store: Store) -> None:
    st.caption(
        "Birden çok parça ve tek seferlik (NRE) kalem içeren teklif. Şablon seçin, iskelet YAML'ı "
        "indirip doldurun, sonra yükleyin. Parça kalemleri examples/ornek_parca.yaml biçimindedir."
    )
    bps = {b.id: b for b in lib.blueprints.blueprints}
    c1, c2 = st.columns([2, 1])
    bp_id = c1.selectbox(
        "Teklif şablonu", list(bps), format_func=lambda i: f"{bps[i].name} — {bps[i].description}"
    )
    no = c2.text_input("Teklif no", "T-2026-100")
    skeleton = yaml.safe_dump(
        blueprint_skeleton(lib, bp_id, no), allow_unicode=True, sort_keys=False
    )
    st.download_button("İskelet YAML indir", skeleton.encode("utf-8"), f"{no}_iskelet.yaml")
    up = st.file_uploader("Doldurulmuş teklif YAML", type=["yaml", "yml"])
    if up is None:
        _saved_sets(store)
        return
    try:
        inp = QuoteSetInput.model_validate(yaml.safe_load(up.getvalue().decode("utf-8")))
        result = compute_quote_set(inp, lib)
    except (QuoteInputError, ValueError, yaml.YAMLError) as exc:
        st.error(f"Girdi hatası: {exc}")
        return
    _show_set_result(result, lib, store)
    _saved_sets(store)


def _show_set_result(result: QuoteSetResult, lib: Library, store: Store) -> None:
    total = result.value("set.order_total")
    st.metric(f"Teklif toplamı [{result.currency}]", "—" if total is None else f"{total:,.2f}")
    _draft_banner(result.has_open_review, len(result.open_review_items))
    if result.manual_review:
        with st.expander(f"Manuel inceleme: {len(result.review_items)} bayrak"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"kalem": k, "kod": fl.code, "neden": fl.reason, "kütüphane yolu": fl.path}
                        for k, fl in result.review_items
                    ]
                ),
                hide_index=True,
            )
    for heading, rows in (("Kalemler", set_item_rows(result)), ("Toplam", set_total_rows(result))):
        st.markdown(f"**{heading}**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "kalem": r.label,
                        "tutar": r.value,
                        "birim": r.unit,
                        "inceleme": "manual_review" if r.review else "",
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            column_config={"tutar": st.column_config.NumberColumn(format="%.2f")},
        )
    for part in result.parts:
        with st.expander(f"Kırılım: {part.input.part_no} {part.input.part_name}"):
            st.dataframe(
                breakdown_frame(part), hide_index=True, height=360, column_config=NUMBER_COLUMNS
            )
    _save_buttons(
        lambda: store.save_quote_set(result, lib),
        lambda: store.save_quote_set(result, lib, final=True),
        result.has_open_review,
    )
    qn = result.input.quote_no
    d1, d2, d3 = st.columns(3)
    with tempfile.TemporaryDirectory() as tmp:
        d1.download_button(
            "Excel indir", write_set_excel(result, Path(tmp) / "t.xlsx").read_bytes(), f"{qn}.xlsx"
        )
    d2.download_button("PDF indir", set_pdf_bytes(result), f"{qn}.pdf")
    d3.download_button("CSV indir", set_breakdown_csv(result).encode("utf-8"), f"{qn}.csv")


def _saved_sets(store: Store) -> None:
    rows = store.list_quote_sets()
    if not rows:
        return
    st.markdown("**Kayıtlı çok kalemli teklifler**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "id": r.id,
                    "teklif": r.quote_no,
                    "rev": r.revision,
                    "müşteri": r.customer,
                    "kalem": r.item_count,
                    "toplam": r.order_total,
                    "para birimi": r.currency,
                    "durum": "KESİN" if r.status == "final" else "taslak",
                    "inceleme": r.manual_review,
                    "tarih": r.created_at,
                }
                for r in rows
            ]
        ),
        hide_index=True,
    )


CHOICE_LABELS = {
    "markup": "markup: fiyat = maliyet × (1 + oran)",
    "margin": "margin: fiyat = maliyet / (1 − oran)",
    "direct_cost": "direkt maliyet üzerine",
    "labor": "yalnız işçilik üzerine",
}


def _library_tab(lib: Library, lib_dir: str) -> None:
    st.subheader("Kâr ve genel gider")
    st.caption(
        "Değişiklikler thresholds.yaml dosyasına yazılır, changelog.csv'ye gerekçesiyle kaydedilir."
    )
    costing = lib.thresholds.costing
    c1, c2 = st.columns(2)
    picks: dict[str, str] = {}
    for col, (file, path) in zip((c1, c2), CHOICES, strict=True):
        options = list(CHOICES[(file, path)])
        cur = current_choice(lib, file, path)
        picks[path] = col.selectbox(
            path.split(".")[-1],
            options,
            index=options.index(cur),
            format_func=lambda o: CHOICE_LABELS.get(o, o),
            key=f"ch_{path}",
        )
    margin = c1.number_input(
        "margin_ratio (oran)",
        0.0,
        0.95,
        value=float(costing.margin_ratio.value or 0.0),
        step=0.01,
        key="margin_ratio",
    )
    overhead = c2.number_input(
        "overhead_ratio (oran)",
        0.0,
        5.0,
        value=float(costing.overhead_ratio.value or 0.0),
        step=0.01,
        key="overhead_ratio",
    )
    reason = st.text_input("Gerekçe", key="settings_reason")
    if st.button("Ayarları kaydet"):
        changes = [Change("thresholds.yaml", p, v) for p, v in picks.items()]
        changes += [
            Change("thresholds.yaml", "costing.margin_ratio", margin),
            Change("thresholds.yaml", "costing.overhead_ratio", overhead),
        ]
        _save(lib_dir, changes, reason)

    st.subheader("Kütüphane değerleri")
    modes = {
        "need": "Seçili proses için gereken eksikler",
        "todo": "Tüm eksik (TODO) değerler",
        "all": "Tüm değerler",
    }
    m1, m2 = st.columns([2, 1])
    mode = m1.radio("Göster", list(modes), format_func=modes.get, horizontal=True, key="lib_mode")
    procs = list(lib.processes)
    process = m2.selectbox(
        "Proses",
        procs,
        index=procs.index("sandvic_toplu") if "sandvic_toplu" in procs else 0,
        key="lib_proc",
        disabled=mode != "need",
    )
    all_values = editable_values(lib)
    if mode == "need":
        needed = needed_paths(lib, process)
        rows = [v for v in all_values if v.value is None and is_needed(v, needed)]
        st.caption(
            f"{process} teklifi için gereken ve boş olan {len(rows)} değer gösteriliyor; "
            f"diğer {sum(v.value is None for v in all_values) - len(rows)} eksik değer bu proseste "
            "kullanılmadığı için gizli. Kalıp/model maliyeti, maça ve nest ayarları teklif "
            "ekranından girilir."
        )
    elif mode == "todo":
        rows = [v for v in all_values if v.value is None]
    else:
        rows = all_values
    only_todo = mode != "all"
    base = pd.DataFrame(
        [
            {
                "dosya": v.file,
                "yol": v.path,
                "açıklama": v.desc,
                "birim": v.unit,
                "para birimi": v.currency,
                "değer": v.value,
            }
            for v in rows
        ],
        columns=["dosya", "yol", "açıklama", "birim", "para birimi", "değer"],
    )
    if base.empty:
        st.info("Eksik değer yok." if only_todo else "Kütüphanede değer yok.")
        return
    edited = st.data_editor(
        base,
        hide_index=True,
        disabled=["dosya", "yol", "açıklama", "birim"],
        column_config={
            "para birimi": st.column_config.SelectboxColumn(options=list(lib.currency.rates)),
            "değer": st.column_config.NumberColumn(format="%.4f"),
        },
        key=f"lib_editor_{mode}_{process}",
        height=420,
    )
    reason_all = st.text_input("Gerekçe (ör. 2026 fiyat listesi)", key="values_reason")
    if st.button("Değerleri kaydet"):
        changes = []
        for (_, old), (_, new) in zip(base.iterrows(), edited.iterrows(), strict=True):
            val = None if pd.isna(new["değer"]) else float(new["değer"])
            ccy = new["para birimi"] if isinstance(new["para birimi"], str) else None
            old_val = None if pd.isna(old["değer"]) else float(old["değer"])
            if val != old_val or ccy != (
                old["para birimi"] if isinstance(old["para birimi"], str) else None
            ):
                changes.append(Change(str(new["dosya"]), str(new["yol"]), val, ccy))
        if changes:
            _save(lib_dir, changes, reason_all)
        else:
            st.info("Değişiklik yok.")


def _save(lib_dir: str, changes: list[Change], reason: str) -> None:
    if not reason.strip():
        st.warning("Gerekçe zorunlu.")
        return
    try:
        set_values(lib_dir, changes, reason)
    except (LibraryError, KeyError, ValueError) as exc:
        st.error(f"Kaydedilmedi, kütüphane değişmedi: {exc}")
        return
    st.success(f"{len(changes)} değer kaydedildi.")
    st.rerun()


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

    part_tab, learn_tab, new_tab, set_tab, saved_tab, lib_tab, data_tab = st.tabs(
        [
            "Parçadan teklif",
            "Öğrenme verisi",
            "Yeni teklif",
            "Çok kalemli teklif",
            "Kayıtlı teklifler",
            "Kütüphane",
            "Veri girişi",
        ]
    )
    with part_tab:
        part_quote.render(lib, store, lambda r, lb, st_: _show_result(r, lb, st_, scope="part"))
    with learn_tab:
        learning_tab.render(lib, db_path.with_name("katalog.sqlite"))
    with new_tab:
        catalog_base = st.session_state.get("catalog_base")
        if catalog_base:
            st.info(
                "Girdi katalogdaki parça ve kalıp kartından dolduruldu "
                f"({catalog_base.get('part_no')}). Katalogda bayraklı değerler taslak olarak kalır."
            )
            if st.button("Katalog girdisini temizle"):
                del st.session_state["catalog_base"]
                st.rerun()
            base = catalog_base
        else:
            base = _example() if st.toggle("Örnek parçayla doldur", value=True) else {}
        inp = _input_form(lib, base)
        if inp is not None:
            try:
                result = compute_quote(inp, lib)
            except (QuoteInputError, ValueError) as exc:
                st.error(f"Girdi hatası: {exc}")
            else:
                _show_result(result, lib, store)
    with set_tab:
        _set_tab(lib, store)
    with saved_tab:
        _saved_tab(store, lib)
    with lib_tab:
        _library_tab(lib, lib_dir)
    with data_tab:
        data_entry.render(lib, lib_dir, db_path.with_name("katalog.sqlite"))


if __name__ == "__main__":
    main()
