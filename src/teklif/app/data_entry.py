"""Streamlit "Veri girişi" tab: wage rates, STEP upload, tool/master model and part cards.

Everything entered here lands in the catalog database with its source (cad / derived /
user). Values measured from CAD can be overridden, but only with a reason.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from teklif.common.models import Source
from teklif.costing.from_catalog import quote_input_from_catalog
from teklif.geometry.plate_features import extract_plate_features, plate_step_values
from teklif.ingest.step_props import (
    EXTRACTOR_VERSION,
    StepUnavailableError,
    StepValue,
    derive_part_values,
    extract_step_props,
    ocp_available,
)
from teklif.library import Library, LibraryError
from teklif.library.editor import Change
from teklif.library.overlay import save_values
from teklif.storage.catalog import (
    CAD_KINDS,
    Catalog,
    CatalogError,
    MasterModel,
    Part,
    Tool,
)
from teklif.storage.catalog_import import import_labor_history, import_tool_history
from teklif.storage.history_import import read_workbook
from teklif.storage.history_schema import WORKBOOKS

KIND_LABELS = {
    "part": "Parça",
    "tool": "Kalıp",
    "master_model": "Master model",
    "assembly": "Komple",
}
# Values the user types in, per card type: key -> (label, unit).
USER_KEYS: dict[str, dict[str, tuple[str, str]]] = {
    "part": {
        "part_perimeter_m": ("Trim kenarı uzunluğu", "m"),
        "hole_count": ("Delik sayısı", "hole"),
    },
    "tool": {
        "tool_area_m2": ("Torbalama alanı (flanş + run-off dahil)", "m2"),
        "tool_perimeter_m": ("Torbalama çevresi (sızdırmazlık bandı)", "m"),
    },
}


def _labels(lib: Library, enum: str) -> dict[str, str]:
    return {i: label for i, label in ((x.id, x.label) for x in getattr(lib.enums, enum))}


def _add[T: (MasterModel, Tool, Part)](cat: Catalog, row: T) -> T:
    """Add a card; show a duplicate-number error instead of a traceback."""
    try:
        return cat.add(row)
    except CatalogError as exc:
        st.error(str(exc))
        st.stop()


def _num(value: Any, default: float = 0.0) -> float:
    """A data_editor cell as float; empty cells arrive as None/NaN."""
    return default if value is None or pd.isna(value) else float(value)


def _pick(label: str, options: dict[str, str], current: str = "", key: str | None = None) -> str:
    ids = list(options)
    index = ids.index(current) if current in ids else 0
    return st.selectbox(label, ids, index=index, format_func=lambda i: options[i], key=key)


def _badge(source: str) -> str:
    return {"cad": "CAD", "derived": "Türetilmiş", "user": "Kullanıcı"}.get(source, source)


# --- wage rates ----------------------------------------------------------------------


def _wages_tab(lib: Library, lib_dir: str, overlay: Path | None = None) -> None:
    wc = lib.work_centers
    eng_rate = wc.engineer_rate
    st.caption(
        "Saat ücreti mühendis ücretine çarpanla bağlıdır: iş merkezi ücreti = mühendis "
        "ücreti × ücret sınıfı çarpanı. Değişiklikler kütüphane kaydına gerekçeyle yazılır."
    )
    c1, c2 = st.columns(2)
    eng = c1.number_input(
        "Mühendis saat ücreti",
        min_value=0.0,
        value=(eng_rate.value if eng_rate else None) or 0.0,
        step=1.0,
        key="engineer_rate_input",
    )
    currency = c2.selectbox(
        "Para birimi",
        list(lib.currency.rates),
        index=list(lib.currency.rates).index(
            eng_rate.currency if eng_rate else lib.currency.base_currency
        ),
    )
    classes = pd.DataFrame(
        [{"id": c.id, "sınıf": c.name, "çarpan": c.multiplier.value} for c in wc.wage_classes]
    )
    edited = st.data_editor(
        classes, hide_index=True, disabled=["id", "sınıf"], key="wage_classes_editor"
    )
    reason = st.text_input("Gerekçe", key="wage_reason")
    if st.button("Ücretleri kaydet"):
        changes = [Change("work_centers.yaml", "engineer_rate", eng, currency)]
        changes += [
            Change("work_centers.yaml", f"wage_classes[{row['id']}].multiplier", row["çarpan"])
            for _, row in edited.iterrows()
        ]
        if not reason.strip():
            st.warning("Gerekçe zorunlu.")
        elif edited["çarpan"].isna().any() or (edited["çarpan"] <= 0).any():
            st.warning("Tüm çarpanlar sıfırdan büyük bir sayı olmalı.")
        elif eng <= 0:
            st.warning("Mühendis saat ücreti sıfırdan büyük olmalı.")
        else:
            try:
                save_values(lib_dir, changes, reason, overlay)
            except (LibraryError, KeyError, ValueError) as exc:
                st.error(f"Kaydedilmedi: {exc}")
            else:
                st.success("Kaydedildi.")
                st.rerun()
    st.subheader("İş merkezi saat ücretleri (sonuç)")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "iş merkezi": w.id,
                    "sınıf": w.wage_class or "(elle girilen)",
                    "saat ücreti": w.labor.value,
                    "para birimi": w.labor.currency,
                }
                for w in wc.work_centers
            ]
        ),
        hide_index=True,
    )
    st.caption("Sınıfsız iş merkezlerinin ücretini Kütüphane sekmesinden girebilirsiniz.")


# --- STEP upload ---------------------------------------------------------------------


def _measure(cat: Catalog, cad_id: int, path: Path, kind: str) -> list[StepValue] | None:
    try:
        props = extract_step_props(path)
    except StepUnavailableError as exc:
        st.warning(f"{path.name}: ölçülemedi ({exc}). Değerleri elle girebilirsiniz.")
        return None
    cat.record_values("cad_file", cad_id, props.values)
    extra: list[StepValue] = []
    if kind == "part":
        try:
            extra = plate_step_values(extract_plate_features(path))
        except StepUnavailableError as exc:
            st.warning(
                f"{path.name}: levha özellikleri çıkarılamadı ({exc}); yaklaşık alan kullanılır."
            )
        cat.record_values("cad_file", cad_id, extra)
    cat.mark_measured(cad_id, EXTRACTOR_VERSION)
    return props.values + (extra or derive_part_values(props))


def _upload_tab(cat: Catalog) -> None:
    if not ocp_available():
        st.info(
            "STEP ölçümü için `pip install teklif[cad]` gerekir; kurulu değilken dosyalar "
            "saklanır, değerler elle girilir."
        )
    files = st.file_uploader("STEP dosyaları", type=["step", "stp"], accept_multiple_files=True)
    kinds: dict[int, str] = {}
    for i, f in enumerate(files):
        kinds[i] = st.selectbox(
            f"{f.name}: dosya türü",
            [*CAD_KINDS],
            format_func=lambda k: KIND_LABELS[k],
            key=f"kind_{i}_{f.name}",
        )
    if files and st.button("Kaydet ve ölç"):
        rows: list[dict[str, Any]] = []
        for i, f in enumerate(files):
            row, created = cat.add_cad_file(f.getvalue(), f.name, kinds[i])
            if not created and row.extractor_version == EXTRACTOR_VERSION:
                st.info(f"{f.name}: aynı içerik zaten kayıtlı ve ölçülmüş ({row.name}).")
                continue
            values = _measure(cat, row.id, Path(row.stored_path), row.kind)
            for v in values or []:
                rows.append(
                    {
                        "dosya": f.name,
                        "değer": v.key,
                        "sonuç": v.value,
                        "birim": v.unit,
                        "kaynak": _badge(v.source.value),
                        "inceleme": v.review_reason or "",
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# --- values with override ------------------------------------------------------------


def _cad_options(cat: Catalog, kind: str) -> dict[int, str]:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from teklif.storage.catalog import CadFile

    with Session(cat.engine) as s:
        rows = s.scalars(select(CadFile).where(CadFile.kind == kind).order_by(CadFile.id))
        return {r.id: f"{r.name} ({r.sha256[:8]})" for r in rows}


def _link_cad_values(cat: Catalog, entity: str, entity_id: int, cad_id: int | None) -> None:
    """Copy the file's CAD values onto the card; derive card-level values from them."""
    if cad_id is None:
        return
    cad = cat.params("cad_file", cad_id)
    values = [
        StepValue(
            key=p.key,
            value=p.value,
            unit=p.unit,
            source=Source(p.source),
            review_reason=p.review_reason,
        )
        for p in cad.values()
    ]
    cat.record_values(entity, entity_id, values)
    if (
        entity == "part"
        and "part_area_m2" not in cad
        and "surface_area_m2" in cad
        and "volume_m3" in cad
    ):
        from teklif.ingest.step_props import StepProps

        derived = derive_part_values(StepProps(solid_count=1, file_length_units=[], values=values))
        cat.record_values(entity, entity_id, derived)
    if entity == "tool" and "envelope_x_mm" in cad and "envelope_y_mm" in cad:
        cat.record_values(
            entity,
            entity_id,
            [
                StepValue(
                    key=f"tool_footprint_{axis}_mm",
                    value=cad[f"envelope_{axis}_mm"].value,
                    unit="mm",
                    source=Source.DERIVED,
                    review_reason="axis-aligned envelope of the tool file; check autoclave layout",
                )
                for axis in ("x", "y")
            ],
        )


def _values_editor(cat: Catalog, entity: str, entity_id: int) -> None:
    params = cat.params(entity, entity_id)
    st.markdown("**Değerler ve kaynakları**")
    if params:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "değer": p.key,
                        "sonuç": p.value,
                        "birim": p.unit,
                        "kaynak": _badge(p.source),
                        "inceleme": p.review_reason or "",
                        "ezme gerekçesi": p.override_reason or "",
                    }
                    for p in params.values()
                ]
            ),
            hide_index=True,
        )
    keys = dict(USER_KEYS.get(entity, {}))
    options = sorted(set(params) | set(keys))
    if not options:
        return
    with st.form(f"val_{entity}_{entity_id}"):
        key = st.selectbox(
            "Değer",
            options,
            format_func=lambda k: keys[k][0] if k in keys else k,
        )
        value = st.number_input("Yeni değer", value=0.0, format="%.6f")
        reason = st.text_input("Ezme gerekçesi (CAD/türetilmiş değeri değiştiriyorsanız zorunlu)")
        if st.form_submit_button("Değeri kaydet"):
            unit = params[key].unit if key in params else keys[key][1]
            try:
                cat.set_param(
                    entity, entity_id, key, value, unit, Source.USER, override_reason=reason
                )
            except CatalogError as exc:
                st.error(str(exc))
            else:
                st.rerun()


# --- tool and master model cards -------------------------------------------------------


def _cost_form(cat: Catalog, entity: str, entity_id: int, lib: Library) -> None:
    st.markdown("**Maliyet (firmaya maliyeti; satış fiyatı değil)**")
    for rec in cat.costs(entity, entity_id):
        other = f", diğer {rec.other:,.2f}" if rec.other is not None else ""
        st.caption(
            f"{rec.cost_date or 'tarihsiz'}: toplam {rec.total:,.2f} {rec.currency}"
            + (f", malzeme {rec.material:,.2f}" if rec.material is not None else "")
            + other
        )
    with st.form(f"cost_{entity}_{entity_id}"):
        c1, c2, c3, c4 = st.columns(4)
        total = c1.number_input("Toplam maliyet", min_value=0.0, step=100.0)
        material = c2.number_input("Malzeme (0 = bilinmiyor)", min_value=0.0, step=100.0)
        currency = c3.selectbox("Para birimi", list(lib.currency.rates))
        when = c4.date_input("Tarih", value=date.today())
        if st.form_submit_button("Maliyeti ekle"):
            try:
                cat.add_cost(
                    entity,
                    entity_id,
                    total,
                    currency,
                    material=material or None,
                    cost_date=when,
                )
            except CatalogError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def _tool_tab(cat: Catalog, lib: Library) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    st.subheader("Master model")
    with Session(cat.engine) as s:
        models = list(s.scalars(select(MasterModel).order_by(MasterModel.id)))
        tools = list(s.scalars(select(Tool).order_by(Tool.id)))
        parts = list(s.scalars(select(Part).order_by(Part.id)))
    with st.form("new_model"):
        no = st.text_input("Master model no")
        mat = _pick("Malzeme", _labels(lib, "master_model_materials"))
        grade = st.text_input("Kalite / ticari ad")
        cads = _cad_options(cat, "master_model")
        cad_id = st.selectbox(
            "STEP dosyası", [None, *cads], format_func=lambda i: "(yok)" if i is None else cads[i]
        )
        if st.form_submit_button("Master modeli ekle") and no.strip():
            new_model = _add(
                cat,
                MasterModel(
                    model_no=no.strip(), material=mat, material_grade=grade, cad_file_id=cad_id
                ),
            )
            _link_cad_values(cat, "master_model", new_model.id, cad_id)
            st.rerun()
    if models:
        model = st.selectbox("Master model", models, format_func=lambda m: m.model_no)
        _cost_form(cat, "master_model", model.id, lib)

    st.subheader("Kalıp")
    with st.form("new_tool"):
        c1, c2 = st.columns(2)
        no = c1.text_input("Kalıp no")
        ttype = _pick("Kalıp tipi", _labels(lib, "tool_types"))
        mclass = _pick("Malzeme sınıfı", _labels(lib, "tool_material_classes"))
        grade = c2.text_input("Malzeme kalitesi / ticari ad")
        solution = _pick("Ters açı çözümü", _labels(lib, "undercut_solutions"))
        parts_n = c1.number_input("Gövde parça sayısı", min_value=1, value=1)
        cores = c2.number_input("Maça sayısı", min_value=0, value=0)
        life = c1.number_input("Hedef kalıp ömrü (çevrim)", min_value=0, value=0)
        sub = c2.checkbox("Altyapı / arka destek var")
        cads = _cad_options(cat, "tool")
        cad_id = st.selectbox(
            "STEP dosyası (flanş, run-off, altyapı dahil)",
            [None, *cads],
            format_func=lambda i: "(yok)" if i is None else cads[i],
        )
        model_id = st.selectbox(
            "Master model",
            [None, *[m.id for m in models]],
            format_func=lambda i: (
                "(yok)" if i is None else next(m.model_no for m in models if m.id == i)
            ),
        )
        served = st.multiselect(
            "Hizmet ettiği parçalar",
            [p.id for p in parts],
            format_func=lambda i: next(p.part_no for p in parts if p.id == i),
        )
        if st.form_submit_button("Kalıbı ekle") and no.strip():
            new_tool = _add(
                cat,
                Tool(
                    tool_no=no.strip(),
                    tool_type=ttype,
                    material_class=mclass,
                    material_grade=grade,
                    body_parts=int(parts_n),
                    loose_cores=int(cores),
                    undercut_solution=solution,
                    target_life=int(life) or None,
                    has_substructure=sub,
                    cad_file_id=cad_id,
                    master_model_id=model_id,
                ),
            )
            for pid in served:
                cat.link_tool_part(new_tool.id, pid)
            _link_cad_values(cat, "tool", new_tool.id, cad_id)
            st.rerun()
    if tools:
        tool = st.selectbox("Kalıp", tools, format_func=lambda t: t.tool_no)
        _cost_form(cat, "tool", tool.id, lib)
        _values_editor(cat, "tool", tool.id)


# --- part card -------------------------------------------------------------------------


def _tool_estimate_caption(cat: Catalog, part_id: int) -> None:
    from teklif.tooling_rules.cost_estimate import estimate_tool_cost

    area = cat.params("part", part_id).get("part_area_m2")
    if area is None or area.value is None or not area.value > 0:
        return
    est = estimate_tool_cost(area.value)
    text = (
        f"Kaba kalıp maliyeti tahmini: {est.value:,.0f} {est.currency} "
        f"({est.low:,.0f}-{est.high:,.0f}). {est.note}"
    )
    (st.caption if est.in_range else st.warning)(text)


def _part_tab(cat: Catalog, lib: Library) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(cat.engine) as s:
        parts = list(s.scalars(select(Part).order_by(Part.id)))
    process = _pick("Proses", _labels(lib, "processes"), key="new_part_process")
    process_tpl = lib.processes.get(process)
    with st.form("new_part"):
        c1, c2 = st.columns(2)
        no = c1.text_input("Parça no")
        rev = c2.text_input("Rev", value="A")
        name = c1.text_input("Ad")
        customer = c2.text_input("Müşteri")
        prepregs = {p.id: p.name for p in lib.materials.prepregs}
        prepreg = _pick("Varsayılan prepreg", prepregs)
        cures = {c.id: c.name for c in process_tpl.cure_cycles} if process_tpl else {"": "-"}
        cure = _pick("Kür çevrimi", cures)
        cx = _pick("Karmaşıklık", _labels(lib, "complexity"))
        cads = _cad_options(cat, "part")
        cad_id = st.selectbox(
            "STEP dosyası", [None, *cads], format_func=lambda i: "(yok)" if i is None else cads[i]
        )
        if st.form_submit_button("Parçayı ekle") and no.strip():
            if process_tpl is None:
                st.error(f"'{process}' için proses şablonu yok; parça eklenmedi.")
                return
            row = _add(
                cat,
                Part(
                    part_no=no.strip(),
                    rev=rev.strip() or "A",
                    name=name,
                    customer=customer,
                    process=process,
                    prepreg=prepreg,
                    cure_cycle=cure,
                    complexity=cx,
                    cad_file_id=cad_id,
                ),
            )
            _link_cad_values(cat, "part", row.id, cad_id)
            st.rerun()
    if not parts:
        st.info("Henüz parça yok.")
        return
    part = st.selectbox("Parça", parts, format_func=lambda p: f"{p.part_no} rev {p.rev}")
    _values_editor(cat, "part", part.id)
    _tool_estimate_caption(cat, part.id)
    _plies_editor(cat, part)
    _ops_editor(cat, lib, part)
    _open_quote(cat, lib, part)


def _open_quote(cat: Catalog, lib: Library, part: Part) -> None:
    st.markdown("**Bu parçadan teklif aç**")
    tools = cat.tools_for_part(part.id)
    if not tools:
        st.caption("Önce Master model / Kalıp sekmesinde bu parçaya hizmet eden bir kalıp ekleyin.")
        return
    c1, c2, c3, c4 = st.columns(4)
    tool = c1.selectbox("Kalıp", tools, format_func=lambda t: t.tool_no, key=f"q_tool_{part.id}")
    quote_no = c2.text_input("Teklif no", value=f"T-{part.part_no}", key=f"q_no_{part.id}")
    qty = c3.number_input("Adet", min_value=1, value=1, key=f"q_qty_{part.id}")
    batch = c4.number_input("Parti adedi", min_value=1, value=1, key=f"q_batch_{part.id}")
    f1, f2 = st.columns([1, 2])
    factor = f1.number_input(
        "Ücret çarpanı", min_value=0.01, value=1.0, step=0.05, key=f"q_wf_{part.id}"
    )
    reason = f2.text_input("Çarpan gerekçesi (1 değilse zorunlu)", key=f"q_wr_{part.id}")
    if st.button("Teklif girdisini hazırla", key=f"q_go_{part.id}"):
        if factor != 1.0 and not reason.strip():
            st.warning("Ücret çarpanı 1 değilse gerekçe zorunlu.")
            return
        try:
            res = quote_input_from_catalog(
                cat, lib, part.id, tool.id, quote_no, int(qty), int(batch), factor, reason
            )
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            st.error(f"Girdi kurulamadı: {exc}")
            return
        if res.input is None:
            st.error("Eksik veri var, teklif açılamadı:")
            for m in res.missing:
                st.markdown(f"- {m}")
            return
        st.session_state["catalog_base"] = res.input.model_dump(mode="json")
        st.success("Teklif girdisi hazır. 'Yeni teklif' sekmesine geçin; kırılım orada.")


def _plies_editor(cat: Catalog, part: Part) -> None:
    st.markdown("**Ply tablosu (düz desen alanı elle)**")
    existing = cat.plies(part.id)
    if existing:
        st.dataframe(
            pd.DataFrame(
                [{"ad": p.name, "alan (m²)": p.area_m2, "adet": p.count} for p in existing]
            ),
            hide_index=True,
        )
    new_rows = st.data_editor(
        pd.DataFrame(
            {
                "ad": pd.Series(dtype=str),
                "alan (m²)": pd.Series(dtype=float),
                "adet": pd.Series(dtype=int),
            }
        ),
        num_rows="dynamic",
        hide_index=True,
        key=f"plies_{part.id}_{len(existing)}",
    )
    if st.button("Ply satırlarını ekle", key=f"plies_add_{part.id}"):
        added = 0
        for _, row in new_rows.iterrows():
            area = _num(row["alan (m²)"])
            if area > 0:
                name = "" if pd.isna(row["ad"]) else str(row["ad"])
                cat.add_ply(part.id, area, int(_num(row["adet"], 1)) or 1, name=name)
                added += 1
        st.success(f"{added} ply satırı eklendi.")
        st.rerun()


def _ops_editor(cat: Catalog, lib: Library, part: Part) -> None:
    st.markdown("**Operasyonlar ve süreler**")
    st.caption(
        "Süre = bir kişinin değil, operasyonun geçen (takvim) dakikası, parça başına; "
        "'kişi' sütunu o sürede çalışan kişi sayısıdır. Geçmiş işler 'gerçekleşen', "
        "tahmininiz 'tahmin' olarak işaretlenir."
    )
    existing = cat.operations(part.id)
    done = {(o.seq, o.operation) for o in existing}
    if existing:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "sıra": o.seq,
                        "operasyon": o.operation,
                        "iş merkezi": o.work_center,
                        "kişi": o.persons,
                        "hazırlık (dk/parti)": o.setup_min,
                        "süre (dk/parça)": o.run_min_per_part,
                        "parti": o.batch_size,
                        "tür": "gerçekleşen" if o.kind == "actual" else "tahmin",
                        "tarih": o.done_on,
                    }
                    for o in existing
                ]
            ),
            hide_index=True,
        )
    tpl = lib.processes.get(part.process)
    seed = [
        {
            "sıra": op.seq,
            "operasyon": op.id,
            "iş merkezi": op.work_center,
            "kişi": 1.0,
            "hazırlık (dk/parti)": 0.0,
            "süre (dk/parça)": 0.0,
            "parti": 1,
            "tür": "gerçekleşen",
            "tarih": date.today(),
        }
        for op in (tpl.operations if tpl else [])
        if (op.seq, op.id) not in done
    ]
    st.caption("Şablon proses şablonundan gelir; süresi 0 olan veya bilinmeyen satırlar eklenmez.")
    edited = st.data_editor(
        pd.DataFrame(seed),
        num_rows="dynamic",
        hide_index=True,
        key=f"ops_{part.id}_{len(existing)}",
        column_config={"tür": st.column_config.SelectboxColumn(options=["gerçekleşen", "tahmin"])},
    )
    if st.button("Operasyonları ekle", key=f"ops_add_{part.id}"):
        count = 0
        for _, r in edited.iterrows():
            run = _num(r.get("süre (dk/parça)"))
            op_id = r.get("operasyon")
            if run <= 0 or op_id is None or pd.isna(op_id) or not str(op_id).strip():
                continue
            seq = int(_num(r.get("sıra")))
            if (seq, str(op_id)) in done:
                continue
            when = r.get("tarih")
            cat.add_operation(
                part.id,
                "estimate" if r.get("tür") == "tahmin" else "actual",
                seq=seq,
                operation=str(op_id),
                work_center=str(r.get("iş merkezi") or ""),
                persons=_num(r.get("kişi"), 1.0) or 1.0,
                setup_min=_num(r.get("hazırlık (dk/parti)")),
                run_min_per_part=run,
                batch_size=int(_num(r.get("parti"), 1)) or 1,
                done_on=when if isinstance(when, date) else None,
            )
            count += 1
        st.success(f"{count} operasyon eklendi.")
        st.rerun()


# --- Excel history import ------------------------------------------------------------


def _excel_tab(cat: Catalog, lib: Library) -> None:
    st.caption(
        "Geçmiş kalıp ve işçilik tabloları (Faz 0 şablonları) katalog kayıtlarına dönüşür. "
        "Var olan kayıtların üzerine yazılmaz."
    )
    kind = st.selectbox(
        "Tablo türü",
        ["kalip_gecmis", "iscilik_gecmis"],
        format_func=lambda k: WORKBOOKS[k].title,
        key="excel_kind",
    )
    upload = st.file_uploader("Excel dosyası", type=["xlsx"], key="excel_file")
    if upload is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.xlsx"
        path.write_bytes(upload.getvalue())
        result = read_workbook(path, WORKBOOKS[kind], lib)
    if not result.ok:
        st.error(f"{len(result.issues)} sorun var; hiçbir şey aktarılmadı.")
        st.dataframe(
            pd.DataFrame(
                [{"satır": i.row, "kolon": i.column, "sorun": i.message} for i in result.issues]
            ),
            hide_index=True,
        )
        return
    st.success(f"{len(result.rows)} satır doğrulandı.")
    if st.button("Kataloğa aktar", key="excel_go"):
        summary = (
            import_tool_history(cat, result.rows)
            if kind == "kalip_gecmis"
            else import_labor_history(cat, result.rows, lib)
        )
        st.write(summary.created or "Yeni kayıt yok.")
        for note in summary.skipped:
            st.warning(note)


# --- entry point -----------------------------------------------------------------------


def render(lib: Library, lib_dir: str, catalog_path: Path, overlay: Path | None = None) -> None:
    cat = Catalog(catalog_path)
    wages, upload, tools, parts, excel = st.tabs(
        ["Saat ücretleri", "STEP yükle", "Master model / Kalıp", "Parça", "Excel geçmişi"]
    )
    with wages:
        _wages_tab(lib, lib_dir, overlay)
    with upload:
        _upload_tab(cat)
    with tools:
        _tool_tab(cat, lib)
    with parts:
        _part_tab(cat, lib)
    with excel:
        _excel_tab(cat, lib)
