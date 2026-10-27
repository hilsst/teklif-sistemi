"""Column specifications for the history / reference upload workbooks.

One spec drives both the empty Excel templates (``teklif.reporting.excel_templates``)
and the importer (``teklif.storage.history_import``), so the format the user fills
in and the format the system reads can never drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from teklif.library import Library

ColumnKind = Literal["text", "int", "float", "date", "list"]

# List sources: "enums.<name>" reads enums.yaml, the others read their library file.
ListSource = Literal[
    "enums.processes",
    "enums.tool_material_classes",
    "enums.master_model_materials",
    "enums.tool_types",
    "enums.undercut_solutions",
    "enums.operations",
    "enums.complexity",
    "enums.yes_no",
    "work_centers",
    "currencies",
]


class Column(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    desc: str
    unit: str = ""
    kind: ColumnKind = "text"
    required: bool = False
    source: ListSource | None = None


class Workbook(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    title: str
    row_meaning: str
    columns: tuple[Column, ...]


def _c(
    key: str,
    desc: str,
    unit: str = "",
    kind: ColumnKind = "text",
    required: bool = False,
    source: ListSource | None = None,
) -> Column:
    if source is not None:
        kind = "list"
    return Column(key=key, desc=desc, unit=unit, kind=kind, required=required, source=source)


_ENVELOPE = "Sınırlayıcı kutu ölçüsü"

TOOL_HISTORY = Workbook(
    name="kalip_gecmis",
    title="Kalıp geçmişi",
    row_meaning="Her satır tamamlanmış bir kalıp (master model ve maçalar dahil).",
    columns=(
        _c("kalip_no", "Kalıp numarası", required=True),
        _c("tarih", "Kalıbın tamamlandığı tarih", kind="date", required=True),
        _c("parca_no", "Parça numarası", required=True),
        _c("parca_adi", "Parça adı"),
        _c("proses", "Parçanın üretim prosesi", required=True, source="enums.processes"),
        _c("kur_sicakligi", "Parçanın kür sıcaklığı", "°C", "float", True),
        _c("parca_zarf_x", f"{_ENVELOPE}, X", "mm", "float"),
        _c("parca_zarf_y", f"{_ENVELOPE}, Y", "mm", "float"),
        _c("parca_zarf_z", f"{_ENVELOPE}, Z", "mm", "float"),
        _c("parca_alani", "Parça yüzey alanı (tek yüz)", "m2", "float"),
        _c("karmasiklik", "Mühendis değerlendirmesi", source="enums.complexity"),
        _c("kalip_tipi", "Erkek / dişi / eşleşik", required=True, source="enums.tool_types"),
        _c(
            "kalip_cinsi",
            "Kalıp malzeme sınıfı",
            required=True,
            source="enums.tool_material_classes",
        ),
        _c("kalip_malzemesi", "Ticari ad / kalite (serbest metin)"),
        _c("kalip_zarf_x", "Kalıp dış ölçüsü, X (flanş ve altyapı dahil)", "mm", "float"),
        _c("kalip_zarf_y", "Kalıp dış ölçüsü, Y", "mm", "float"),
        _c("kalip_zarf_z", "Kalıp dış ölçüsü, Z", "mm", "float"),
        _c("kalip_yuzey_alani", "Kalıp çalışma yüzeyi alanı (run-off dahil)", "m2", "float"),
        _c("kalip_agirligi", "Kalıp ağırlığı", "kg", "float"),
        _c(
            "master_model",
            "Master model malzemesi (yoksa 'yok')",
            required=True,
            source="enums.master_model_materials",
        ),
        _c("ters_aci_cozumu", "Ters açı çözümü", required=True, source="enums.undercut_solutions"),
        _c("kalip_parca_sayisi", "Kalıp gövdesi kaç parça (tek parça = 1)", "adet", "int", True),
        _c("maca_sayisi", "Gevşek parça (maça) sayısı", "adet", "int"),
        _c("alt_yapi", "Arka destek / altyapı var mı", source="enums.yes_no"),
        _c("hedef_kalip_omru", "Tasarım kalıp ömrü", "çevrim", "int"),
        _c(
            "master_model_malzeme_maliyeti", "Master model malzeme maliyeti", "para birimi", "float"
        ),
        _c("master_model_cnc_saat", "Master model CNC süresi", "saat", "float"),
        _c(
            "master_model_iscilik_saat",
            "Master model el işçiliği (yüzey, cila vb.)",
            "adam-saat",
            "float",
        ),
        _c(
            "kalip_malzeme_maliyeti",
            "Kalıp malzemesi (board, tooling prepreg, metal)",
            "para birimi",
            "float",
        ),
        _c("kalip_sarf_maliyeti", "Kalıp imalatındaki sarf", "para birimi", "float"),
        _c("kalip_cnc_saat", "Kalıp CNC süresi", "saat", "float"),
        _c(
            "kalip_iscilik_saat", "Kalıp işçiliği (laminasyon, montaj, yüzey)", "adam-saat", "float"
        ),
        _c("kalip_kur_cevrim_sayisi", "Kalıp imalatındaki autoclave/fırın çevrimi", "adet", "int"),
        _c("maca_maliyeti", "Maçaların toplam maliyeti", "para birimi", "float"),
        _c("alt_yapi_maliyeti", "Altyapı / arka destek maliyeti", "para birimi", "float"),
        _c("dis_hizmet_maliyeti", "Fason işler (CNC, kaynak, ölçüm vb.)", "para birimi", "float"),
        _c(
            "fiili_toplam_maliyet",
            "Gerçekleşen toplam kalıp maliyeti (master model dahil)",
            "para birimi",
            "float",
            True,
        ),
        _c("teklif_edilen_fiyat", "Müşteriye teklif edilen kalıp fiyatı", "para birimi", "float"),
        _c(
            "para_birimi",
            "Bu satırdaki tüm tutarların para birimi",
            required=True,
            source="currencies",
        ),
        _c("notlar", "Serbest not"),
    ),
)

LABOR_HISTORY = Workbook(
    name="iscilik_gecmis",
    title="İşçilik geçmişi",
    row_meaning="Her satır bir iş emrindeki bir operasyon (parti toplamı).",
    columns=(
        _c("is_emri_no", "İş emri numarası", required=True),
        _c("tarih", "Operasyon tarihi", kind="date", required=True),
        _c("parca_no", "Parça numarası", required=True),
        _c("proses", "Üretim prosesi", required=True, source="enums.processes"),
        _c("operasyon", "Operasyon", required=True, source="enums.operations"),
        _c("is_merkezi", "İş merkezi", required=True, source="work_centers"),
        _c("karmasiklik", "Mühendis değerlendirmesi", source="enums.complexity"),
        _c("parti_adedi", "Bu satırdaki parça adedi", "adet", "int", True),
        _c(
            "uretim_sira_no",
            "Partinin ilk parçasının kümülatif sıra no (öğrenme eğrisi için)",
            "adet",
            "int",
        ),
        _c("ply_sayisi", "Ply sayısı (parça başı)", "adet", "int"),
        _c(
            "toplam_ply_alani",
            "Ply düz desen alanları toplamı (parça başı, fire hariç)",
            "m2",
            "float",
        ),
        _c("parca_alani", "Parça yüzey alanı (tek yüz)", "m2", "float"),
        _c("trim_cevresi", "Trim kenarı uzunluğu (parça başı)", "m", "float"),
        _c("delik_sayisi", "Delik sayısı (parça başı)", "adet", "int"),
        _c("kalip_alani", "Kalıp torbalama alanı", "m2", "float"),
        _c("kalip_cevresi", "Kalıp torbalama çevresi", "m", "float"),
        _c("operator_sayisi", "Operasyondaki kişi sayısı", "kişi", "int"),
        _c(
            "hazirlik_suresi",
            "Hazırlık süresi (parti başı, aşağıdaki toplama dahil)",
            "adam-dk",
            "float",
        ),
        _c("fiili_iscilik", "Partinin gerçekleşen toplam işçiliği", "adam-dk", "float", True),
        _c("makine_suresi", "Makine çalışma süresi (CNC, kesim, NDT vb.)", "dk", "float"),
        _c("standart_sure", "Varsa mevcut standart süre (parti toplamı)", "adam-dk", "float"),
        _c("notlar", "Serbest not (fire, tekrar işlem, beklenmedik durum)"),
    ),
)

REFERENCE_QUOTES = Workbook(
    name="referans_teklifler",
    title="Referans teklifler",
    row_meaning="Her satır bir teklif ve (varsa) gerçekleşen maliyeti. Tutarlar parça başıdır.",
    columns=(
        _c("teklif_no", "Teklif numarası", required=True),
        _c("tarih", "Teklif tarihi", kind="date", required=True),
        _c("musteri", "Müşteri"),
        _c("parca_no", "Parça numarası", required=True),
        _c("parca_adi", "Parça adı"),
        _c("cad_dosyasi", "STEP/IGES/DXF/PDF dosya adı (ayrıca yüklenecek)"),
        _c("proses", "Üretim prosesi", required=True, source="enums.processes"),
        _c("kur_sicakligi", "Kür sıcaklığı", "°C", "float"),
        _c("malzeme", "Prepreg / malzeme (serbest metin veya kütüphane id)"),
        _c("siparis_adedi", "Teklif edilen toplam adet", "adet", "int", True),
        _c("parti_adedi", "Tipik parti büyüklüğü", "adet", "int"),
        _c("ply_sayisi", "Ply sayısı", "adet", "int"),
        _c("toplam_ply_alani", "Ply düz desen alanları toplamı (fire hariç)", "m2", "float"),
        _c("parca_alani", "Parça yüzey alanı (tek yüz)", "m2", "float"),
        _c("parca_zarf_x", f"{_ENVELOPE}, X", "mm", "float"),
        _c("parca_zarf_y", f"{_ENVELOPE}, Y", "mm", "float"),
        _c("parca_zarf_z", f"{_ENVELOPE}, Z", "mm", "float"),
        _c("trim_cevresi", "Trim kenarı uzunluğu", "m", "float"),
        _c("delik_sayisi", "Delik sayısı", "adet", "int"),
        _c("karmasiklik", "Mühendis değerlendirmesi", source="enums.complexity"),
        _c("kalip_cinsi", "Kalıp malzeme sınıfı", source="enums.tool_material_classes"),
        _c("ters_aci_cozumu", "Ters açı çözümü", source="enums.undercut_solutions"),
        _c(
            "para_birimi",
            "Bu satırdaki tüm tutarların para birimi",
            required=True,
            source="currencies",
        ),
        _c("teklif_malzeme", "Teklif: malzeme (fire dahil)", "para birimi/adet", "float"),
        _c("teklif_sarf", "Teklif: sarf", "para birimi/adet", "float"),
        _c("teklif_iscilik_saat", "Teklif: işçilik süresi", "adam-saat/adet", "float"),
        _c("teklif_iscilik", "Teklif: işçilik tutarı", "para birimi/adet", "float"),
        _c("teklif_autoclave", "Teklif: autoclave payı", "para birimi/adet", "float"),
        _c(
            "teklif_kalip_nre", "Teklif: kalıp + master model (NRE, toplam)", "para birimi", "float"
        ),
        _c(
            "teklif_birim_fiyat",
            "Teklif: birim satış fiyatı (NRE hariç)",
            "para birimi/adet",
            "float",
            True,
        ),
        _c("fiili_malzeme", "Gerçekleşen: malzeme", "para birimi/adet", "float"),
        _c("fiili_sarf", "Gerçekleşen: sarf", "para birimi/adet", "float"),
        _c("fiili_iscilik_saat", "Gerçekleşen: işçilik süresi", "adam-saat/adet", "float"),
        _c("fiili_iscilik", "Gerçekleşen: işçilik tutarı", "para birimi/adet", "float"),
        _c("fiili_kalip_nre", "Gerçekleşen: kalıp + master model (toplam)", "para birimi", "float"),
        _c(
            "fiili_birim_maliyet",
            "Gerçekleşen: birim maliyet (NRE hariç)",
            "para birimi/adet",
            "float",
        ),
        _c("kazanildi", "Teklif kazanıldı mı", source="enums.yes_no"),
        _c("notlar", "Serbest not"),
    ),
)

WORKBOOKS: dict[str, Workbook] = {
    wb.name: wb for wb in (TOOL_HISTORY, LABOR_HISTORY, REFERENCE_QUOTES)
}


def list_values(lib: Library, source: ListSource) -> list[tuple[str, str]]:
    """``(id, label)`` pairs allowed for a list column, read from the library."""
    if source == "work_centers":
        return [(w.id, w.name) for w in lib.work_centers.work_centers]
    if source == "currencies":
        return [(c, c) for c in lib.currency.rates]
    name = source.removeprefix("enums.")
    return [(i.id, i.label) for i in getattr(lib.enums, name)]
