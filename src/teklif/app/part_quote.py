"""'Parçadan teklif' tab: start a quote from the part STEP alone (tool cost is estimated)."""

from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from teklif.app.surface_panel import render_surface_panel
from teklif.common.models import Price, Source
from teklif.costing.engine import QuoteInputError, QuoteResult, compute_quote
from teklif.costing.inputs import PartGeometry, PlyGroup, QuoteInput
from teklif.costing.partial import fill_missing
from teklif.geometry.plate_features import PlateFeatures, extract_plate_features
from teklif.ingest.step_props import StepProps, StepUnavailableError, extract_step_props
from teklif.library import Library
from teklif.storage.db import Store
from teklif.tooling_rules.cost_estimate import (
    ToolCostEstimate,
    estimate_model_cost,
    estimate_tool_cost,
)

TOPLU_AREA_RANGE = (2.4, 4.7)  # area range of the 7 panels behind process sandvic_toplu
TRIM_RULE_OUTER = "varsayım: yalnız dış kenar trimlenir (kütüphane: trim.trim_all_openings = 0)"
TRIM_RULE_ALL = (
    "varsayım: parça üzerindeki tüm boşluklar, delikler ve kenarlar trimlenir "
    "(trim çevresi = kalıp yüzeyinin tüm sınırı)"
)
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


def _price_breakdown_table(result: QuoteResult) -> None:
    """Per-unit price split: part cost + model share + tool share + profit = unit price."""
    rows = [
        ("price.part_cost", "Parça maliyeti"),
        ("price.share_master", "Model payı"),
        ("price.share_tooling", "Kalıp payı"),
        ("price.profit", "Kâr"),
        ("price.unit_final", "Müşteriye ürün başı fiyat"),
    ]
    keys = {ln.key for ln in result.lines}
    frame = pd.DataFrame(
        [
            {"kalem": label, f"[{result.currency}] ürün başına": result.value(k)}
            for k, label in rows
            if k in keys
        ]
    )
    st.markdown("**Ürün başı fiyat kırılımı**")
    st.dataframe(
        frame,
        hide_index=True,
        column_config={frame.columns[1]: st.column_config.NumberColumn(format="%.2f")},
    )
    if "price.share_tooling" not in keys:
        _separate_nre_summary(result, keys)


def _separate_nre_summary(result: QuoteResult, keys: set[str]) -> None:
    """NRE as its own cost: first order pays it, repeat orders only the unit price."""
    parts = []
    for key, label in (("nre.tooling", "kalıp"), ("nre.master", "model")):
        if key in keys:
            parts.append(f"{label} {result.value(key):,.2f}")
    nre_key = "price.nre_priced" if "price.nre_priced" in keys else "nre.total"
    nre_key = nre_key if nre_key in keys else "nre.tooling"
    nre_total = result.value(nre_key)
    unit = result.value("price.unit_final")
    first = result.value("price.order_total")
    if nre_total is None or unit is None or first is None:
        return
    cost_txt = " + ".join(parts)
    st.markdown(
        f"**NRE (tek seferlik): {nre_total:,.2f} {result.currency}** "
        f"(maliyet: {cost_txt}; kâr eklendiyse satış bedeli). Tekrar siparişlerde kalıp ve model "
        f"hazır olduğu için yalnız ürün başı fiyat ({unit:,.2f} {result.currency}) geçerlidir. "
        f"İlk sipariş toplamı: **{first:,.2f} {result.currency}**."
    )


def _partial_view(result: QuoteResult, filled: list[str]) -> None:
    """Price from the entered factors only; missing library values count as 0."""
    st.error(
        "KISMİ FİYAT (teklif değildir): kütüphanede değeri girilmemiş kalemler 0 sayılarak "
        "hesaplandı, gerçek fiyat bundan farklı olacaktır "
        "(genellikle daha yüksek; kur eksikse 1:1 alınır). Eksikleri girdikçe fiyat tamamlanır."
    )
    cols = st.columns(4)
    for col, key, label in zip(
        cols,
        ["cost.unit", "price.unit_final", "nre.total", "price.order_total"],
        ["Kısmi birim maliyet", "Kısmi birim fiyat", "NRE", "Kısmi sipariş toplamı"],
        strict=True,
    ):
        try:
            v = result.value(key)
        except KeyError:
            v = None
        col.metric(f"{label} [{result.currency}]", "—" if v is None else f"{v:,.2f}")
    _price_breakdown_table(result)
    with st.expander(f"Eksik (0 sayılan) değerler: {len(filled)}"):
        st.dataframe(pd.DataFrame({"kütüphane değeri": filled}), hide_index=True)


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
        "Trim çevresi, kalıp yüzeyini seçince otomatik hesaplanır; seçmezseniz aşağıya elle girin."
    )
    with st.expander("3B inceleme ve kalıp yüzeyi seçimi", expanded=True):
        sel = render_surface_panel(upload.getvalue(), lib.thresholds, "pq_surf", FLANGE_MM)
    trim_all = (lib.thresholds.trim.trim_all_openings.value or 0) >= 0.5
    trim_sel = sel.trim_m(trim_all) if sel else 0.0
    rule = "tüm dış kenar, boşluk ve delikler" if trim_all else "yalnız dış kenar"
    tool_part_area = sel.area_m2 if sel else f.one_side_area_m2
    if sel:
        st.success(
            f"Kalıp yüzeyi: {sel.area_m2:.2f} m², trim çevresi {trim_sel:.2f} m ({rule}), "
            f"çekme yönü {sel.pull_label}."
        )
    est = estimate_tool_cost(tool_part_area)
    _estimate_card(est)
    core_choice: str | None = "yok"
    core_cost = 0.0
    if sel and sel.undercut_ratio > 0:
        st.warning(
            f"Seçilen çekme yönünde ters açı var (yüzeyin %{sel.undercut_ratio * 100:.1f}'i). "
            "Bu bölge tek parça kalıptan çıkmaz."
        )
        core_choice = st.radio(
            "Ters açı için maça (veya bölünmüş kalıp) kullanacak mısınız?",
            [
                "Evet, maça kullanacağım",
                "Evet, bölünmüş kalıp kullanacağım",
                "Hayır, ters açıyı başka şekilde çözeceğim",
            ],
            key="pq_core",
            index=None,
        )
        if core_choice and not core_choice.startswith("Hayır"):
            core_cost = st.number_input(
                f"Maça / bölünme ek maliyeti [{est.currency}] (bize maliyet)",
                0.0,
                step=100.0,
                key="pq_core_cost",
                help="Kalıp maliyetine eklenir. Bilmiyorsanız 0 bırakın; teklif taslak kalır.",
            )
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
        index=procs.index("sandvic_toplu") if "sandvic_toplu" in procs else 0,
        key="pq_proc",
    )
    cycles = [x.id for x in lib.processes[process].cure_cycles]
    cure = c2.selectbox("Kür çevrimi", cycles, key="pq_cure")
    material = c3.selectbox("Ana prepreg", prepregs, key="pq_mat")
    autoclave = c1.selectbox("Autoclave", autoclaves, key="pq_ac")
    cx = c2.selectbox("Karmaşıklık", complexity, key="pq_cx")
    auto_per = round(trim_sel, 3) if sel else 0.0
    perimeter = c3.number_input(
        "Trim çevresi [m]" + (" (seçimden; değiştirirseniz gerekçe düşer)" if sel else " (elle)"),
        0.0,
        value=auto_per,
        key=f"pq_per_{auto_per}",
    )
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
    st.markdown("**Kumaş, kat sayısı ve nest**")
    mat_cpt = next(p for p in lib.materials.prepregs if p.id == material).cured_ply_thickness
    sandwich = "sandvic" in process
    k1, k2, k3, k4 = st.columns(4)
    cpt = k1.number_input(
        "Cured ply thickness [mm]",
        0.0,
        value=float(mat_cpt.value or 0.0),
        step=0.01,
        format="%.3f",
        help="Seçilen kumaşın kütüphane değeri; değiştirebilirsiniz."
        + (" Kütüphanede boş, elle girin." if mat_cpt.value is None else ""),
        key=f"pq_cpt_{material}",
    )
    lam_default = 0.0 if sandwich else round(f.mean_thickness_mm, 2)
    lam_t = k2.number_input(
        "Laminat kalınlığı [mm]" + (" (sandviçte bir yüzey kabuğu)" if sandwich else ""),
        0.0,
        value=lam_default,
        step=0.1,
        help="Sandviçte ölçülen kalınlık çekirdeği de içerir, bu yüzden kabuk kalınlığını girin."
        if sandwich
        else "Varsayılan: STEP'ten ölçülen ortalama kalınlık.",
        key=f"pq_lamt_{process}_{upload.name}_{upload.size}",
    )
    skins = int(
        k3.number_input(
            "Yüzey (kabuk) sayısı", 1, value=2 if sandwich else 1, key=f"pq_skin_{process}"
        )
    )
    plies_each = math.ceil(lam_t / cpt - 1e-9) if cpt > 0 and lam_t > 0 else 0
    auto_plies = plies_each * skins
    k4.metric("Otomatik kat sayısı", auto_plies if auto_plies else "—")
    if auto_plies:
        st.caption(
            f"Kat sayısı = ⌈{lam_t:g} ÷ {cpt:g}⌉ = {plies_each} kat × {skins} yüzey = "
            f"{auto_plies}; yukarı yuvarlanır. Aşağıdaki tabloda değiştirebilirsiniz."
        )
    else:
        st.info(
            "Kat sayısı için cured ply thickness ve laminat kalınlığını girin (şimdilik 1 kat)."
        )
    lib_eff = lib.thresholds.costing.nest_efficiency.value or 0.8
    lib_fire = lib.thresholds.costing.fire_ratio.value or 0.0
    e1, e2 = st.columns(2)
    nest_eff = (
        e1.number_input(
            "Nest verimi [%]",
            1.0,
            100.0,
            round(lib_eff * 100, 1),
            step=1.0,
            help="Kumaşı kalıba %100 yerleştiremezsiniz. Harcanan kumaş = kullanılan ÷ verim "
            "(%80 verimde 1 m için 1,25 m).",
            key="pq_nest",
        )
        / 100
    )
    fire = (
        e2.number_input(
            "Ek fire payı [%]",
            0.0,
            100.0,
            round(lib_fire * 100, 1),
            step=1.0,
            help="Nest dışında ek fire (kesim artığı, hatalı kesim). Harcanan × (1 + fire).",
            key="pq_fire",
        )
        / 100
    )
    nest_changed = abs(nest_eff - lib_eff) > 1e-9 or abs(fire - lib_fire) > 1e-9
    st.caption(
        f"Kumaş tüketim çarpanı: {(1 / nest_eff) * (1 + fire):.3f} "
        f"(1 ÷ {nest_eff:.2f} × (1 + {fire:.2f}))"
    )
    st.markdown(
        "**Ply grupları** (düz desen alanı, fire hariç; varsayılan: parça alanı × kat sayısı)"
    )
    plies_df = st.data_editor(
        pd.DataFrame(
            [
                {
                    "name": "ply",
                    "area_m2": round(f.one_side_area_m2, 4),
                    "count": max(auto_plies, 1),
                }
            ]
        ),
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("ad"),
            "area_m2": st.column_config.NumberColumn("alan [m2]", min_value=0.0, format="%.4f"),
            "count": st.column_config.NumberColumn("adet", min_value=1, step=1),
        },
        key=f"pq_plies_{auto_plies}",
    )
    st.markdown("**Kalıp ve model (teklif onaylanınca üretilir; maliyet ürün başına dağıtılır)**")
    mest = estimate_model_cost()
    n1, n2, n3 = st.columns(3)
    tool_cost = n1.number_input(
        f"Kalıp maliyeti [{est.currency}] (tahmin; gerçek teklifiniz varsa değiştirin)",
        0.0,
        value=round(est.value, 2),
        key=f"pq_tool_{round(est.value)}",
    )
    model_cost = n2.number_input(
        f"Model maliyeti [{mest.currency}] (ortalama; gerçek teklifiniz varsa değiştirin)",
        0.0,
        value=round(mest.value, 2),
        key="pq_model",
    )
    nre_qty = n3.number_input(
        "NRE'nin dağıtılacağı adet",
        1,
        value=int(quantity),
        help="Kalıp ve model maliyeti kaç ürüne bölünsün (varsayılan sipariş adedi).",
        key=f"pq_nreqty_{int(quantity)}",
    )
    a1, a2 = st.columns(2)
    amortize = a1.checkbox(
        "Kalıp ve modeli ürün fiyatına yedir (varsayılan: ayrı NRE)",
        False,
        key="pq_amort",
        help="Kapalıysa kalıp ve model ayrı kalem (NRE) olarak teklif edilir.",
    )
    nre_margin = a2.checkbox("Kalıp ve model maliyetine de kâr ekle", True, key="pq_nremargin")
    if perimeter <= 0:
        st.warning("Trim çevresi girilmedi: trim süresi 0 hesaplanıyor (kalıp yüzeyi seçin).")
    flange_m = FLANGE_MM / 1000
    outer = sel.outer_m if sel else perimeter  # flange runs along the outer rim only
    notes = [
        "part_area_m2, hole_count: STEP'ten türetilmiş tahmin (levha özellikleri)",
        f"kalıp maliyeti: {est.note}",
        f"kalıp izi/alanı: {FLANGE_SOURCE}",
        TRIM_RULE_ALL if trim_all else TRIM_RULE_OUTER,
    ]
    if perimeter <= 0:
        notes.append("trim çevresi girilmedi: trim süresi ve flanş alanı 0 hesaplandı")
    if sel:
        notes.append(sel.note())
        if abs(perimeter - trim_sel) > 0.001:
            notes.append(
                f"trim çevresi seçimden ({trim_sel:.2f} m) farklı girildi: {perimeter:.2f} m"
            )
        if sel.undercut_ratio > 0:
            ratio = f"%{sel.undercut_ratio * 100:.1f}"
            if core_choice is None:
                notes.append(f"ters açı var ({ratio}): maça/bölünmüş kalıp kararı verilmedi")
            elif core_choice.startswith("Hayır"):
                notes.append(
                    f"ters açı var ({ratio}): maça/bölünmüş kalıp kullanılmayacak, incelenmeli"
                )
            elif core_cost <= 0:
                notes.append(
                    f"ters açı var ({ratio}): {core_choice.lower()} ama maliyeti girilmedi"
                )
            else:
                notes.append(
                    f"ters açı ({ratio}): {core_choice.lower()}, ek maliyet {core_cost:,.0f}"
                )
    if (
        process == "sandvic_toplu"
        and not TOPLU_AREA_RANGE[0] <= f.one_side_area_m2 <= TOPLU_AREA_RANGE[1]
    ):
        msg = (
            f"toplu serim süresi {TOPLU_AREA_RANGE[0]}-{TOPLU_AREA_RANGE[1]} m2 "
            f"aralığındaki 7 panelden çıkarıldı; {f.one_side_area_m2:.2f} m2 "
            "bu aralığın dışında, güvenilmez"
        )
        notes.append(msg)
        st.warning(msg)
    if not est.in_range:
        notes.append("kalıp maliyeti tahmini kalibrasyon aralığı dışında: güvenilmez")
    if abs(model_cost - mest.value) < 0.01:
        notes.append(f"model maliyeti: {mest.note}")
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
                tool_area_m2=tool_part_area + flange_m * outer,
                tool_perimeter_m=outer + 2 * 3.14159265 * flange_m,
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
                value=tool_cost + core_cost,
                currency=est.currency,
                per="pcs",
                source=Source.DERIVED
                if abs(tool_cost - est.value) < 0.01 and core_cost == 0
                else Source.USER,
                note=(est.note if abs(tool_cost - est.value) < 0.01 else "kullanıcı girişi")
                + (f" + maça/bölünme {core_cost:,.0f}" if core_cost else ""),
            ),
            master_nre=Price(
                value=model_cost,
                currency=mest.currency,
                per="pcs",
                source=Source.DERIVED if abs(model_cost - mest.value) < 0.01 else Source.USER,
                note=mest.note if abs(model_cost - mest.value) < 0.01 else "kullanıcı girişi",
            ),
            amortize_nre=amortize,
            nre_margin=nre_margin,
            nre_quantity=int(nre_qty),
            nest_efficiency=nest_eff if nest_changed else None,
            fire_ratio=fire,
            review_notes=notes,
        )
        result = compute_quote(inp, lib)
        partial = None
        if result.value("price.unit_final") is None:
            flib, filled = fill_missing(lib)
            try:
                partial = (compute_quote(inp, flib), filled)
            except (ArithmeticError, QuoteInputError) as exc:
                st.warning(f"Kısmi fiyat hesaplanamadı ({exc}); eksik değerleri girin.")
    except (QuoteInputError, ValueError) as exc:
        st.error(f"Girdi hatası: {exc}")
        return
    if partial is not None:
        _partial_view(*partial)
    else:
        _price_breakdown_table(result)
    show_result(result, lib, store)
