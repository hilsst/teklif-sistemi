"""'Parçadan teklif' tab: start a quote from the part STEP alone (tool cost is estimated)."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from teklif.common.models import Price, Source
from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.geometry.plate_features import PlateFeatures, extract_plate_features
from teklif.ingest.step_props import StepProps, StepUnavailableError, extract_step_props
from teklif.library import Library
from teklif.storage.db import Store
from teklif.tooling_rules.cost_estimate import ToolCostEstimate, estimate_tool_cost

FLANGE_MM = 150.0  # typical flange length (Kemal, 2026-11-04); becomes a library value in Faz 3
FLANGE_SOURCE = "flanş payı 150 mm (tipik değer, kütüphaneye taşınacak)"


@dataclass(frozen=True)
class Measured:
    name: str
    props: StepProps
    features: PlateFeatures


def measure(data: bytes, name: str) -> Measured:
    """Measure an uploaded STEP file (needs OCP); raises StepUnavailableError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "part.step"
        path.write_bytes(data)
        return Measured(name, extract_step_props(path), extract_plate_features(path))


def tool_footprint_mm(props: StepProps) -> tuple[float, float]:
    """Two largest envelope dimensions plus the flange on both sides (rough, flagged)."""
    dims = sorted(
        (props.get(f"envelope_{a}_mm").value or 0.0 for a in ("x", "y", "z")), reverse=True
    )
    return dims[0] + 2 * FLANGE_MM, dims[1] + 2 * FLANGE_MM


def _cached(upload: Any) -> Measured | None:
    data = upload.getvalue()
    key = "pq_measured_" + hashlib.sha256(data).hexdigest()
    if key not in st.session_state:
        try:
            st.session_state[key] = measure(data, upload.name)
        except StepUnavailableError as exc:
            st.error(f"{upload.name}: ölçülemedi ({exc})")
            return None
    return st.session_state[key]  # type: ignore[no-any-return]


def _estimate_card(est: ToolCostEstimate) -> None:
    st.metric(
        "Tahmini kalıp maliyeti",
        f"{est.value:,.0f} {est.currency}",
        help=f"Aralık {est.low:,.0f} - {est.high:,.0f} {est.currency}. {est.note}",
    )
    (st.caption if est.in_range else st.warning)(
        f"Aralık {est.low:,.0f} - {est.high:,.0f} {est.currency}. {est.note}"
    )


def render(lib: Library, store: Store, show_result: Any) -> None:
    st.markdown(
        "Yalnız **parça STEP'ini** yükleyin. Alan, kalınlık ve delik sayısı ölçülür; kalıp "
        "maliyeti geçmiş kalıplardan tahmin edilir. Teklif onaylanana kadar kalıp ve model "
        "üretilmez, bu yüzden bu satırlar **tahmin** olarak işaretlenir."
    )
    upload = st.file_uploader("Parça STEP dosyası", type=["step", "stp"], key="pq_upload")
    if upload is None:
        return
    m = _cached(upload)
    if m is None:
        return
    f = m.features
    c = st.columns(4)
    c[0].metric("Parça alanı (tek yüz)", f"{f.one_side_area_m2:.2f} m²")
    c[1].metric("Ort. kalınlık", f"{f.mean_thickness_mm:.1f} mm")
    c[2].metric("Delik sayısı", f.hole_count)
    c[3].metric("Eğrisel alan", f"{f.curved_area_fraction:.0%}")
    st.caption(
        "Bu değerler STEP'ten türetilmiş tahminlerdir; teklifte 'incelenmeli' işaretiyle taşınır. "
        "Trim çevresi STEP'ten güvenilir çıkmadığı için aşağıya elle girilir."
    )
    est = estimate_tool_cost(f.one_side_area_m2)
    _estimate_card(est)
    fx, fy = tool_footprint_mm(m.props)
    st.caption(f"Kalıp izi tahmini: {fx:.0f} × {fy:.0f} mm ({FLANGE_SOURCE}).")

    procs = list(lib.processes)
    prepregs = [p.id for p in lib.materials.prepregs]
    complexity = list(lib.thresholds.complexity_factors)
    autoclaves = [a.id for a in lib.autoclaves.autoclaves]
    c1, c2, c3 = st.columns(3)
    stem = Path(m.name).stem
    quote_no = c1.text_input("Teklif no", f"TASLAK-{stem[:24]}", key="pq_qno")
    part_no = c2.text_input("Parça no", stem, key="pq_pno")
    customer = c3.text_input("Müşteri", "", key="pq_cust")
    process = c1.selectbox(
        "Proses",
        procs,
        index=procs.index("sandvic_autoclave") if "sandvic_autoclave" in procs else 0,
        key="pq_proc",
    )
    cycles = [x.id for x in lib.processes[process].cure_cycles]
    cure = c2.selectbox("Kür çevrimi", cycles, key="pq_cure")
    material = c3.selectbox("Ana prepreg", prepregs, key="pq_mat")
    autoclave = c1.selectbox("Autoclave", autoclaves, key="pq_ac")
    cx = c2.selectbox("Karmaşıklık", complexity, key="pq_cx")
    perimeter = c3.number_input("Trim çevresi [m] (elle)", 0.0, value=0.0, key="pq_per")
    q1, q2, q3 = st.columns(3)
    quantity = q1.number_input("Sipariş adedi", 1, value=1, key="pq_qty")
    batch = q2.number_input("Parti adedi", 1, value=1, key="pq_batch")
    wage = q3.number_input("Ücret çarpanı", 0.01, value=1.0, step=0.05, key="pq_wage")
    wage_reason = st.text_input("Ücret çarpanı gerekçesi (çarpan 1 değilse zorunlu)", key="pq_wr")
    template = lib.processes[process]
    o1, o2 = st.columns(2)
    include = o1.multiselect(
        "Opsiyonel operasyonlar (boya, montaj vb.)",
        [op.id for op in template.operations if op.optional],
        key="pq_inc",
    )
    exclude = o2.multiselect(
        "Bu parçada yapılmayan operasyonlar",
        [op.id for op in template.operations],
        key="pq_exc",
    )
    st.markdown("**Ply grupları** (düz desen alanı, fire hariç; varsayılan: parça alanı × 1)")
    plies_df = st.data_editor(
        pd.DataFrame([{"name": "ply", "area_m2": round(f.one_side_area_m2, 4), "count": 1}]),
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("ad"),
            "area_m2": st.column_config.NumberColumn("alan [m2]", min_value=0.0, format="%.4f"),
            "count": st.column_config.NumberColumn("adet", min_value=1, step=1),
        },
        key="pq_plies",
    )
    if perimeter <= 0:
        st.info("Teklifi hesaplamak için trim çevresini girin.")
        return
    flange_m = FLANGE_MM / 1000
    notes = [
        "part_area_m2, hole_count: STEP'ten türetilmiş tahmin (levha özellikleri)",
        f"kalıp maliyeti: {est.note}",
        f"kalıp izi/alanı: {FLANGE_SOURCE}",
    ]
    if not est.in_range:
        notes.append("kalıp maliyeti tahmini kalibrasyon aralığı dışında: güvenilmez")
    try:
        inp = QuoteInput(
            quote_no=quote_no,
            part_no=part_no,
            customer=customer,
            process=process,
            material=material,
            cure_cycle=cure,
            autoclave=autoclave,
            complexity=cx,
            plies=[
                PlyGroup(
                    name=str(r["name"] or ""), area_m2=float(r["area_m2"]), count=int(r["count"])
                )
                for r in plies_df.to_dict("records")
                if pd.notna(r.get("area_m2"))
            ],
            geometry=PartGeometry(
                part_area_m2=f.one_side_area_m2,
                part_perimeter_m=perimeter,
                hole_count=f.hole_count,
                tool_area_m2=f.one_side_area_m2 + flange_m * perimeter,
                tool_perimeter_m=perimeter + 2 * 3.14159265 * flange_m,
                tool_footprint_x_mm=fx,
                tool_footprint_y_mm=fy,
                source=Source.DERIVED,
            ),
            quantity=int(quantity),
            batch_size=int(batch),
            include_optional=include,
            exclude_operations=exclude,
            wage_factor=float(wage),
            wage_factor_reason=wage_reason,
            tooling_nre=Price(
                value=round(est.value, 2),
                currency=est.currency,
                per="pcs",
                source=Source.DERIVED,
                note=est.note,
            ),
            review_notes=notes,
        )
        result = compute_quote(inp, lib)
    except (QuoteInputError, ValueError) as exc:
        st.error(f"Girdi hatası: {exc}")
        return
    show_result(result, lib, store)
