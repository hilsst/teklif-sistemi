"""Markdown report of every library value that still needs company data."""

from __future__ import annotations

from itertools import groupby

from teklif.library.loader import Library, list_todos

FILE_TITLES: dict[str, str] = {
    "currency.yaml": "Kurlar",
    "work_centers.yaml": "İş merkezleri ve saat ücretleri",
    "materials.yaml": "Malzeme ve sarf fiyatları, malzeme özellikleri",
    "autoclaves.yaml": "Autoclave ölçü ve maliyetleri",
    "thresholds.yaml": "Genel katsayılar",
}

PRIORITY_NOTE = (
    "Öncelik: Faz 1 (maliyet motoru) için iş merkezi saat ücretleri, prepreg ve sarf "
    "fiyatları, proses şablonundaki standart süreler, autoclave ölçü ve saat maliyeti ve "
    "genel katsayılar (fire, öğrenme eğrisi, genel gider, marj) gerekir. Kalıp malzemesi "
    "özellikleri (servis sıcaklığı, CTE, ömür, fiyat) Faz 3'e kadar bekleyebilir."
)

UNIT_LEGEND = (
    "Birim gösterimi: `min` = dakika, `h` = saat, `m2` = m², `degC` = °C, "
    "`ratio` = oran (0–1), `1e-6/K` = 10⁻⁶/K, `min/batch` = parti başına dakika, "
    "`EUR/m2` = m² başına EUR. Para birimi her kalemde ayrı seçilebilir "
    "(EUR, TRY, USD); farklı birim kullanırsanız belirtmeniz yeterli."
)


def _title(fname: str) -> str:
    if fname.startswith("processes/"):
        return f"Standart süreler ve sarf kullanımı ({fname})"
    return FILE_TITLES.get(fname, fname)


def render_todo_markdown(lib: Library) -> str:
    todos = list_todos(lib)
    lines = [
        "# Firma verisi ihtiyaç listesi",
        "",
        "Bu liste `teklif todo` komutuyla kütüphanedeki boş (`value: null`) alanlardan "
        "otomatik üretilir. Bir değer girildikçe listeden düşer.",
        "",
        PRIORITY_NOTE,
        "",
        UNIT_LEGEND,
        "",
        f"Toplam açık kalem: **{len(todos)}**",
    ]
    for fname, group in groupby(todos, key=lambda t: t.file):
        items = list(group)
        lines += ["", f"## {_title(fname)} — `{fname}` ({len(items)})", ""]
        lines += ["| # | Anahtar | Açıklama | Birim |", "|---|---|---|---|"]
        for n, t in enumerate(items, start=1):
            lines.append(f"| {n} | `{t.path}` | {t.desc or ''} | `{t.unit}` |")
    return "\n".join(lines) + "\n"
