# kalip-analiz

Bir kalıp yüzeyi için **otomatik çekme yönü**, **kalıp ayrım (bölme) noktaları** ve **maça (core)
yerleri** bulan, sonucu 3B'de gösteren bağımsız Python paketi. Maliyet, veritabanı veya teklif
kodu içermez. Teklif-sistemi'nden ayrılmıştır; orada da aynı paket kullanılır.

Ne yapar: seçilen kalıp yüzeyine göre parçayı en iyi çıkaran yönü arar; ters açı (undercut)
bölgelerini bulur; tek parça / maça / bölünmüş kalıp / bölünmüş + maça yöntemlerini puanlayıp
karşılaştırır ve en uygununu önerir. Emin olamadığı durumu `manual_review` bayrağıyla işaretler.
Her sayı formül, girdi ve kaynağıyla `result.assessment.trace` içinde izlenebilir.

## Kurulum

```bash
# STEP okumak için [cad] gerekir (cadquery-ocp). Streamlit bileşeni için [streamlit].
pip install "kalip-analiz[cad] @ git+https://github.com/hilsst/teklif-sistemi#subdirectory=packages/kalip-analiz"
# geliştirme: depoyu klonla, sonra
pip install -e "packages/kalip-analiz[cad]"
```

Python ≥ 3.12. STEP yerine kendi üçgen ağınızı (numpy dizileri, STL, trimesh…) veriyorsanız
`[cad]` gerekmez (aşağıya bakın).

## Kullanım

```python
from kalip_analiz import analyze

# faces: kalıp yüzeyini oluşturan B-rep yüz numaraları (müşterinin parça yüzeyinin kalıba oturan tarafı)
r = analyze("parca.step", faces=[3, 4, 7])
# ya da bir yöne bakan tüm yüzler:  analyze("parca.step", faces="+Z")

print(r.summary())  # "Bölünmüş kalıp: 2 parça, 1 maça" + gerekçeler
print(r.pull)  # tek parça için en uygun çekme yönü (x, y, z)
print(r.assessment.methods)  # karşılaştırılan yöntemler, en iyisi başta (puan, parça, maça)
for p in r.method.pieces:  # her kalıp parçasının çekme yönü
    print(p.pull.direction, len(p.cores))
print(r.cores)  # M1.. maçalar (renk, alan, konum) ve çözülemeyen Ç1.. bölgeler
print(r.manual_review)  # True ise mühendis incelemesi gerekir (nedenler r.assessment.flags)

r.export_html("sonuc.html")  # tek dosya: 3B harita + karar + maça tablosu, tarayıcıda açılır
```

Seçenekler: `thresholds="benim.yaml"` (eşikler), `pull=(0, 0, 1)` (bu yönü değerlendir),
`max_pieces=2` (kalıp en fazla kaç parça), `method=1` (karşılaştırma tablosundaki başka yöntemi
anlat).

Başka bir mesh kaynağından:

```python
from kalip_analiz import analyze, mesh_from_arrays

mesh = mesh_from_arrays(
    vertices, triangles, face_of_triangle
)  # üçgenler parçanın DIŞINA bakan normalle
r = analyze(mesh, faces=[0, 1])
```

## Eşikler (YAML)

Hiçbir eşik kodda değildir. Paketle gelen [`thresholds.yaml`](src/kalip_analiz/thresholds.yaml)
varsayılanları taşır: `draft.*` (2° tek parça, 0° ters açı…), `undercut.*` (maça/bölme alan
oranı, maliyet ağırlıkları), `pull_search.*`, `mesh.*` (tessellation), `viewer.*`. Kendi dosyanıza
yalnız değiştirmek istediğiniz anahtarları yazın:

```yaml
draft:
  min_ok_deg: {value: 3.0}
undercut:
  max_split_pieces: {value: 2}
```

`analyze(..., thresholds="benim.yaml")`. Yöntem puanı ağırlıkları (`undercut.*_cost_weight`)
algoritma varsayılanıdır; kendi gerçek maliyet oranlarınıza göre ayarlayın.

## 3B görüntüleyici (ayrı bileşen)

`kalip_analiz.viewer` (three.js, `static/index.html`): çekme açısı haritası (yeşil ≥ 2°, sarı,
kırmızı ters açı), her kalıp parçası ayrı renkte, yüzeye dik ayırma yüzeyleri, parça başına çekme
oku, her maça kendi renginde **M1, M2…** numaralı artı + çizgi işaretiyle, çözülemeyen bölgeler
kırmızı **Ç1…**. Yüze tıklayınca seçilir (ids köşede listelenir).

* Tek dosya HTML: `r.export_html(yol)` veya `viewer.export_html(payload, yol, side_html)`.
* Streamlit: `comp = viewer.streamlit_component(); comp(**r.viewer_payload(), height=560, key="v", default=None)`;
  tıklamalar `{"toggles": [yüz no…]}` olarak döner (`viewer.select_pause_s` duraklamasından sonra).
* Web uygulaması: `static/index.html` + `three.min.js`'i servis edin; `r.viewer_payload()`
  JSON'unu sayfaya `window.KALIP_VIEWER_DATA` olarak verin (`export_html` bunu yapar).

## Sınırlar

Işın izleme yok (çarpışma/çıkış sırası doğrulanmaz); sonuçlar tessellation tahminidir ve
mühendis tarafından gözden geçirilmelidir. Yüzey seçimi girdidir, otomatik bulunmaz.

Testler: `pytest packages/kalip-analiz/tests`.
