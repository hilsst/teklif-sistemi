"""'Öğrenme verisi' screen: one place to teach the system from real jobs.

Part STEP + the hours each operation really took + tool and master-model costs (with their STEP
files when they exist). Saved cases refit the rough tool-cost formula, the model-cost range and
labor-hour fits; the refit is shown against the current numbers and written only on request.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from teklif.costing.engine import QuoteInputError, fx_param
from teklif.library import Library
from teklif.storage.catalog import Catalog, CatalogError
from teklif.storage.learning import (
    CaseInput,
    OpEntry,
    collect_model_costs,
    collect_tool_samples,
    fit_labor,
    fit_linear,
    fit_model_cost,
    labor_rows,
    save_case,
    write_cost_models,
)
from teklif.tooling_rules.cost_estimate import DEFAULT_PATH, load_model, load_model_cost


def _to_eur(lib: Library):  # type: ignore[no-untyped-def]
    def conv(amount: float, currency: str) -> float | None:
        if currency == "EUR":
            return amount
        try:
            param, _ = fx_param(lib, currency, "EUR")
        except QuoteInputError:
            return None
        return None if param.value is None else amount * param.value

    return conv


def _case_form(cat: Catalog, lib: Library) -> None:
    gen = st.session_state.setdefault("lr_gen", 0)  # bumped after a save to reset the form
    if flash := st.session_state.pop("lr_flash", None):
        st.success(flash)
    st.markdown("### 1. Parça")
    part_file = st.file_uploader("Parça STEP dosyası", type=["step", "stp"], key=f"lr_{gen}_part")
    c1, c2, c3 = st.columns(3)
    part_no = c1.text_input(
        "Parça no",
        Path(part_file.name).stem if part_file else "",
        key=f"lr_{gen}_pn_{part_file.name if part_file else ''}",
    )
    customer = c2.text_input("Müşteri", key=f"lr_{gen}_cust")
    procs = list(lib.processes)
    process = c3.selectbox(
        "Üretim prosesi",
        procs,
        index=procs.index("sandvic_toplu") if "sandvic_toplu" in procs else 0,
        key=f"lr_{gen}_proc",
    )

    st.markdown("### 2. Bilinen işçilik saatleri")
    st.caption(
        "Her operasyon için **toplam adam-saat** (parça başına). Bilmediğiniz operasyonları boş "
        "bırakın (0). Gerçek süre değil tahminse 'tahmin' seçin."
    )
    tpl = lib.processes[process]
    seed = pd.DataFrame(
        [
            {
                "operasyon": op.id,
                "iş merkezi": op.work_center,
                "süre [adam-saat]": 0.0,
                "tür": "gerçekleşen",
            }
            for op in tpl.operations
        ]
    )
    ops_df = st.data_editor(
        seed,
        num_rows="dynamic",
        hide_index=True,
        key=f"lr_{gen}_ops_{process}",
        column_config={
            "tür": st.column_config.SelectboxColumn(options=["gerçekleşen", "tahmin"]),
            "süre [adam-saat]": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
        },
    )

    st.markdown("### 3. Model ve kalıp")
    has_step = st.radio(
        "Bu parça için model/kalıp STEP dosyanız var mı?",
        ["Yok, yalnız maliyetleri gireceğim", "Var, STEP dosyalarını da yükleyeceğim"],
        key=f"lr_{gen}_hasstep",
        horizontal=True,
    ).startswith("Var")
    tool_step = model_step = None
    if has_step:
        u1, u2 = st.columns(2)
        tool_step = u1.file_uploader("Kalıp STEP", type=["step", "stp"], key=f"lr_{gen}_tool_step")
        model_step = u2.file_uploader(
            "Model STEP", type=["step", "stp"], key=f"lr_{gen}_model_step"
        )
    k1, k2, k3 = st.columns(3)
    currency = k1.selectbox("Para birimi", list(lib.currency.rates), key=f"lr_{gen}_ccy")
    tool_cost = k2.number_input(
        "Kalıp maliyeti (bize maliyet)", 0.0, step=100.0, key=f"lr_{gen}_tc"
    )
    model_cost = k3.number_input(
        "Model maliyeti (bize maliyet)", 0.0, step=100.0, key=f"lr_{gen}_mc"
    )
    includes_master = st.checkbox(
        "Girdiğim kalıp maliyeti modeli de içeriyor", False, key=f"lr_{gen}_incl"
    )
    done_on = st.date_input("Maliyet tarihi", date.today(), key=f"lr_{gen}_date")

    if st.button("Bu örneği kaydet", type="primary", key=f"lr_{gen}_save"):
        if not part_no.strip():
            st.error("Parça no gerekli.")
            return
        if part_file is None:
            st.warning("Parça STEP'i yüklenmedi: alan ölçülemez, bu örnek formüllere girmez.")
        ops = [
            OpEntry(
                str(r["operasyon"]),
                float(r["süre [adam-saat]"] or 0.0),
                str(r["iş merkezi"] or ""),
                actual=r["tür"] != "tahmin",
            )
            for _, r in ops_df.iterrows()
            if pd.notna(r["operasyon"]) and str(r["operasyon"]).strip()
        ]
        case = CaseInput(
            part_no=part_no,
            part_step=(part_file.getvalue(), part_file.name) if part_file else None,
            customer=customer,
            process=process,
            ops=ops,
            tool_cost=tool_cost or None,
            model_cost=model_cost or None,
            currency=currency,
            tool_step=(tool_step.getvalue(), tool_step.name) if tool_step else None,
            model_step=(model_step.getvalue(), model_step.name) if model_step else None,
            includes_master=includes_master,
            done_on=done_on,
        )
        try:
            with st.spinner("Kaydediliyor ve STEP ölçülüyor..."):
                result = save_case(cat, case)
        except CatalogError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Kaydedilemedi: {exc}")
            return
        st.session_state["lr_flash"] = f"{part_no} kaydedildi." + "".join(
            f" Not: {n}" for n in result.notes
        )
        st.session_state["lr_gen"] = gen + 1
        st.rerun()


def _cases_table(cat: Catalog) -> None:
    rows = labor_rows(cat)
    with_hours = {r.part_no: r for r in rows}
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from teklif.storage.catalog import Part

    with Session(cat.engine) as s:
        parts = list(s.scalars(select(Part).order_by(Part.id)))
    frame = []
    for p in parts:
        area = cat.params("part", p.id).get("part_area_m2")
        tools = cat.tools_for_part(p.id)
        tool_costs = [c for t in tools for c in cat.costs("tool", t.id)]
        hours = with_hours[p.part_no].hours if p.part_no in with_hours else {}
        frame.append(
            {
                "parça": p.part_no,
                "alan [m²]": area.value if area is not None else None,
                "kalıp maliyeti": tool_costs[-1].total if tool_costs else None,
                "para": tool_costs[-1].currency if tool_costs else "",
                "işçilik [adam-saat]": round(sum(hours.values()), 2) if hours else None,
            }
        )
    if frame:
        st.dataframe(pd.DataFrame(frame), hide_index=True)
    else:
        st.info("Henüz kayıtlı örnek yok.")


def _fits(cat: Catalog, lib: Library) -> None:
    st.markdown("### Kayıtlı örnekler formülleri nasıl güncelliyor?")
    conv = _to_eur(lib)
    samples, notes = collect_tool_samples(cat, conv)
    cur = load_model()
    fit = fit_linear([s.area_m2 for s in samples], [s.cost for s in samples])
    st.markdown("**Kalıp maliyeti** = sabit + m² başına × parça alanı")
    if fit is None:
        st.info(
            f"Formül için alanı ölçülmüş ve maliyeti girilmiş en az 3 farklı büyüklükte kalıp "
            f"gerekir (şimdi {len(samples)})."
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "": "Şu anki formül",
                        "sabit [EUR]": cur.base,
                        "m² başına [EUR]": cur.per_m2,
                        "örnek": cur.samples,
                        "R²": cur.r_squared,
                        "LOO hata": cur.leave_one_out_max_error,
                    },
                    {
                        "": "Kayıtlı örneklerden",
                        "sabit [EUR]": round(fit.base, 1),
                        "m² başına [EUR]": round(fit.slope, 1),
                        "örnek": fit.n,
                        "R²": round(fit.r_squared, 3),
                        "LOO hata": round(fit.loo_max_error, 3),
                    },
                ]
            ),
            hide_index=True,
        )
        st.caption(
            f"Geçerli alan aralığı {fit.x_min:.2f}-{fit.x_max:.2f} m². LOO hata: bir örneği "
            "çıkarıp onu tahmin ettirince en büyük sapma."
        )
        if fit.n < 5:
            st.warning("5'ten az örnek: formül yaklaşık, yeni örnekler geldikçe iyileşir.")
        for n in notes:
            st.caption(n)
        model_fit = fit_model_cost(collect_model_costs(cat, conv))
        if model_fit:
            cm = load_model_cost()
            st.markdown(
                f"**Model maliyeti**: ortalama {cm.mean:,.0f} → {model_fit.mean:,.0f} EUR "
                f"({model_fit.n} model, aralık {model_fit.min:,.0f}-{model_fit.max:,.0f})"
            )
        if st.button("Formülü bu verilerle güncelle", key="lr_apply"):
            names = ", ".join(s.part_no for s in samples[:12])
            try:
                backup = write_cost_models(
                    DEFAULT_PATH, fit, model_fit, f"öğrenme verisi: {len(samples)} kalıp ({names})"
                )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            else:
                st.success(
                    "Formül güncellendi. Yeni teklifler bunu kullanır."
                    + (f" Eski dosya yedeklendi: {backup.name}" if backup else "")
                )
    st.markdown("**İşçilik saatleri**")
    rows = labor_rows(cat)
    ops = sorted({o for r in rows for o in r.hours})
    if not ops:
        st.info("Henüz işçilik saati girilmiş örnek yok.")
        return
    op = st.selectbox("Operasyon", ops, key="lr_labor_op")
    lf = fit_labor(rows, op)
    if lf is None:
        st.info("Bu operasyon için en az 3 farklı büyüklükte parça gerekir.")
    else:
        st.write(
            f"{op}: süre [adam-saat] = **{lf.base:.2f} + {lf.slope:.2f} × alan [m²]** "
            f"({lf.n} parça, R² {lf.r_squared:.2f}, {lf.x_min:.2f}-{lf.x_max:.2f} m²). "
            "Değerleri Kütüphane sekmesindeki proses şablonuna elle işleyin."
        )


def render(lib: Library, catalog_path: Path) -> None:
    cat = Catalog(catalog_path)
    st.markdown(
        "Gerçek bir işi buradan öğretin: **parça STEP'i + bildiğiniz işçilik saatleri + "
        "kalıp ve model maliyetleri**. Kalıp/model STEP'iniz yoksa yalnız maliyetleri girin."
    )
    _case_form(cat, lib)
    st.divider()
    st.markdown("### Kayıtlı örnekler")
    _cases_table(cat)
    _fits(cat, lib)
