# teklif-sistemi

Autoclave/prepreg kompozit parça ve kalıp teklif sistemi (kurum içi, tek kullanıcı, yerel).
CAD/teknik resimden kalıp ihtiyacı, ters açı çözümü, kalıp cinsi, master model, proses rotası
ve kalem kalem izlenebilir maliyet çıkarır. Durum: **Faz 1** (maliyet motoru, geometri elle girilir).

## Windows'ta kurulum ve çalıştırma

1. uv'yi kurun (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Repoyu indirin (GitHub Desktop veya `git clone https://github.com/hilsst/teklif-sistemi`).
3. Klasördeki **`baslat.bat`** dosyasına çift tıklayın. İlk çalıştırmada Python 3.12 ve
   paketler kurulur, ardından arayüz tarayıcıda açılır (http://localhost:8501).

PDF çıktısı saf Python'dur (fpdf2 + gömülü DejaVu yazı tipi); Windows'ta ek kurulum gerekmez.
CI testleri hem Ubuntu hem Windows üzerinde koşar.

## Kurulum (geliştirici)

```bash
# uv: https://docs.astral.sh/uv/
uv python install 3.12
uv sync --all-extras    # .venv + arayüz (streamlit)
uv run pytest           # testler
uv run ruff check . && uv run mypy
```

## Komutlar

```bash
uv run teklif validate                         # kütüphaneyi doğrula
uv run teklif todo --out docs/firma_veri_ihtiyaci.md   # eksik firma verisi listesi
uv run teklif templates --out templates        # boş yükleme Excel'leri
uv run teklif check-upload iscilik_gecmis dolu_dosya.xlsx   # doldurulmuş dosyayı denetle
```

### Teklif (Faz 1)

```bash
uv run teklif app                                   # Streamlit arayüzü (http://localhost:8501)
uv run teklif quote examples/ornek_parca.yaml --xlsx teklif.xlsx --pdf teklif.pdf --db data/teklif.sqlite
uv run teklif regress referans_teklifler.xlsx       # referans set: hesap vs fiili, ±%15
```

Girdi dosyası örneği: `examples/ornek_parca.yaml`. Kütüphanede TODO değer kaldıkça ilgili kalemler
ve toplamlar hesaplanmaz (`—`), her biri kütüphane yoluyla birlikte `manual_review` listesinde görünür.
Veritabanı varsayılan olarak `data/teklif.sqlite` (`TEKLIF_DB` ile değiştirilebilir); her kayıt yeni
bir revizyondur ve kullanılan kütüphanenin SHA-256 özetini saklar.

### Demo (dummy veri)

`examples/demo_library` gerçek kütüphanenin, TODO alanları **uydurma** değerlerle doldurulmuş
kopyasıdır (`scripts/make_demo_library.py`). Yalnızca sistemi denemek içindir; gerçek teklifte
kullanmayın.

```bash
uv run teklif --library examples/demo_library app
uv run teklif --library examples/demo_library quote examples/ornek_parca.yaml --pdf demo.pdf
uv run teklif --library examples/demo_library regress examples/demo_referans_teklifler.xlsx
```

Farklı bir kütüphane klasörü için `--library <klasör>` veya `TEKLIF_LIBRARY` ortam değişkeni.

## Klasör yapısı

```
library/                 Düzenlenebilir kütüphane (YAML) — koda gömülü sayı yok
  currency.yaml          Para birimleri ve kurlar (temel: EUR, karma para birimi destekli)
  enums.yaml             Sözlükler: proses, kalıp cinsi, operasyon, ters açı çözümü...
  materials.yaml         Prepreg, sarf, kalıp/master model malzemeleri
  work_centers.yaml      İş merkezleri, işçilik ve makine saat ücretleri, verim
  autoclaves.yaml        Autoclave ölçüleri, sınırları, saat maliyeti
  thresholds.yaml        Karar eşikleri (draft, maça oranı, tessellation) ve katsayılar
  processes/             Proses şablonları (operasyon, süre sürücüleri, sarf kullanımı)
src/teklif/
  common/                Param, Price, CostLine, ReviewFlag (izlenebilirlik), birim, kur
  library/               YAML şemaları, yükleyici, çapraz doğrulama, TODO raporu
  routing/               Proses şablonundan operasyon listesi ve sürücü miktarları
  costing/               Saf formüller (formulas), motor (engine), girdi şeması, regresyon
  storage/               SQLite (teklif, ezme kaydı, gerçekleşme), yükleme formatları, içe aktarıcı
  reporting/             Teklif Excel/PDF çıktısı, boş Excel şablon üretici
  app/                   Streamlit arayüzü
  ingest, drawing_reader, geometry, draft, tooling_rules
                         Sonraki fazlar için boş paketler
templates/               Üretilmiş boş Excel'ler (kalip_gecmis, iscilik_gecmis, referans_teklifler)
docs/                    Firma verisi ihtiyaç listesi
tests/
```

## Kütüphane kuralları

- Her sayı `{value, unit, source, desc, note}` yapısındadır; fiyatlar `{value, currency, per}`.
- `value: null` = firma verisi bekleniyor (TODO). Hesapta kullanılırsa sonuç üretilmez,
  kalem `manual_review` (`missing_value`) bayrağı alır.
- `source`: `user` (sizin verdiğiniz), `library`, `default` (literatür/varsayılan, doğrulanmalı),
  `cad`, `drawing`, `todo`.
- Birimler kontrollü bir sözlükten gelir (`min/m2`, `EUR/h`, `degC`...). Yazım hatası yükleme
  sırasında yakalanır.
- Proses şablonunda rate birimi sürücüyle uyumlu olmalıdır (`ply_area_total_m2` → `min/m2`);
  sarf miktarı `<fiyat birimi>/<sürücü birimi>` olmalıdır. Uyumsuzluk yüklemede hata verir.
- Para birimleri: her fiyat kendi para birimini taşır; `currency.yaml` içindeki kurla teklif
  para birimine çevrilir. Kur girilmemişse çevrim hata verir, tahmin yapılmaz.

## Kütüphaneyi arayüzden düzenleme

Arayüzdeki **Kütüphane** sekmesinde:
- **Kâr ve genel gider:** kâr formülü (`markup`: fiyat = maliyet × (1 + oran) /
  `margin`: fiyat = maliyet / (1 − oran)), genel gider tabanı ve oranlar.
- **Tüm kütüphane değerleri:** eksik (TODO) veya tüm değerler tablo olarak; değer ve para birimi
  doğrudan girilebilir.

Her kayıt gerekçe ister, kütüphane yeniden doğrulanır (hatalıysa hiçbir dosya değişmez), YAML
yorumları korunur ve değişiklik `library/changelog.csv` dosyasına (zaman, yol, eski, yeni,
gerekçe) eklenir. YAML dosyalarını doğrudan düzenlemek de mümkündür.

## Süre modeli (proses şablonu)

```
t_std (dk/parça) = setup / parti_adedi + Σ rate_i × sürücü_i     [× karmaşıklık çarpanı]
t_ödenen        = t_std / iş_merkezi.efficiency
T_n             = T_ref × (n/ref)^log2(LC)    ref = learning_reference_unit (varsayılan 1 → T_1)
öğrenme çarpanı = sipariş boyunca ortalama: mean((n/ref)^b), n = ilk_sıra .. ilk_sıra + adet - 1
```

Öğrenme eğrisi yalnız run süresine (setup hariç) ve `learning: true` operasyonlara uygulanır
(kesim, kür, CNC trim, NDT, ölçüm makine ağırlıklı olduğu için `false`).

## Maliyet modeli (parça başı, teklif para birimi)

| Kalem | Formül |
|---|---|
| İşçilik | süre / 60 × operatör × işçilik ücreti × kur |
| Makine | süre / 60 × makine ücreti × kur |
| Prepreg | ply alanı × adet × (1 + nesting firesi) × fiyat × kur |
| Sarf | sürücü × kullanım × (1 + sarf payı) × fiyat × kur |
| Autoclave | çevrim süresi × saat ücreti × kur × (X+2s)(Y+2s) / (L × W × doluluk) |
| Genel gider | (direkt maliyet veya işçilik) × oran — `overhead_basis` |
| Birim fiyat | markup: maliyet × (1 + oran); margin: maliyet / (1 − oran) — `margin_basis` |
| NRE | kullanıcı girişi (Faz 3'te hesaplanacak); ayrı satır veya `amortize_nre` ile birim fiyata yedirme |

Kontroller (`manual_review`): kalıp autoclave'a sığmıyor, çevrim autoclave sınırını aşıyor,
prepreg kür sıcaklığı çevrimle uyuşmuyor, NRE girilmemiş ve kalıp mevcut işaretlenmemiş.
Her kalem arayüzde gerekçe yazılarak ezilebilir; ezme sonraki tüm toplamlara yansır ve kaydedilir.

## Yükleme şablonları

| Dosya | Satır anlamı |
|---|---|
| `kalip_gecmis.xlsx` | Tamamlanmış bir kalıp (master model, maça dahil) |
| `iscilik_gecmis.xlsx` | Bir iş emrindeki bir operasyon (parti toplamı) |
| `referans_teklifler.xlsx` | Bir teklif ve gerçekleşen maliyeti (30–50 iş) |

Her dosyada `Veri` (turuncu başlık zorunlu), `Aciklama` (kolon, birim, geçerli değerler) ve
`Listeler` (açılır liste kaynakları, kütüphaneden üretilir) sayfaları vardır. Kütüphaneye yeni
iş merkezi veya proses eklenince `teklif templates` ile şablonları yeniden üretin.
