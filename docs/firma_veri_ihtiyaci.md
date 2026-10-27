# Firma verisi ihtiyaç listesi

Bu liste `teklif todo` komutuyla kütüphanedeki boş (`value: null`) alanlardan otomatik üretilir. Bir değer girildikçe listeden düşer.

Öncelik: Faz 1 (maliyet motoru) için iş merkezi saat ücretleri, prepreg ve sarf fiyatları, proses şablonundaki standart süreler, autoclave ölçü ve saat maliyeti ve genel katsayılar (fire, öğrenme eğrisi, genel gider, marj) gerekir. Kalıp malzemesi özellikleri (servis sıcaklığı, CTE, ömür, fiyat) Faz 3'e kadar bekleyebilir.

Birim gösterimi: `min` = dakika, `h` = saat, `m2` = m², `degC` = °C, `ratio` = oran (0–1), `1e-6/K` = 10⁻⁶/K, `min/batch` = parti başına dakika, `EUR/m2` = m² başına EUR. Para birimi her kalemde ayrı seçilebilir (EUR, TRY, USD); farklı birim kullanırsanız belirtmeniz yeterli.

Toplam açık kalem: **174**

## Kurlar — `currency.yaml` (2)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `rates.TRY` | 1 EUR kaç TL | `TRY/EUR` |
| 2 | `rates.USD` | 1 EUR kaç USD | `USD/EUR` |

## Malzeme ve sarf fiyatları, malzeme özellikleri — `materials.yaml` (68)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `prepregs[PP_CF_UD_120].areal_weight` | Lif alan ağırlığı (FAW) | `g/m2` |
| 2 | `prepregs[PP_CF_UD_120].cured_ply_thickness` | Kürlenmiş ply kalınlığı (CPT) | `mm` |
| 3 | `prepregs[PP_CF_UD_120].roll_width` | Rulo eni | `mm` |
| 4 | `prepregs[PP_CF_UD_120].out_life` | Oda sıcaklığında çalışma ömrü | `h` |
| 5 | `prepregs[PP_CF_UD_120].price` | Prepreg birim fiyatı | `EUR/m2` |
| 6 | `prepregs[PP_CF_WOVEN_120].areal_weight` | Lif alan ağırlığı (FAW) | `g/m2` |
| 7 | `prepregs[PP_CF_WOVEN_120].cured_ply_thickness` | Kürlenmiş ply kalınlığı (CPT) | `mm` |
| 8 | `prepregs[PP_CF_WOVEN_120].roll_width` | Rulo eni | `mm` |
| 9 | `prepregs[PP_CF_WOVEN_120].out_life` | Oda sıcaklığında çalışma ömrü | `h` |
| 10 | `prepregs[PP_CF_WOVEN_120].price` | Prepreg birim fiyatı | `EUR/m2` |
| 11 | `prepregs[PP_CF_UD_180].areal_weight` | Lif alan ağırlığı (FAW) | `g/m2` |
| 12 | `prepregs[PP_CF_UD_180].cured_ply_thickness` | Kürlenmiş ply kalınlığı (CPT) | `mm` |
| 13 | `prepregs[PP_CF_UD_180].roll_width` | Rulo eni | `mm` |
| 14 | `prepregs[PP_CF_UD_180].out_life` | Oda sıcaklığında çalışma ömrü | `h` |
| 15 | `prepregs[PP_CF_UD_180].price` | Prepreg birim fiyatı | `EUR/m2` |
| 16 | `prepregs[PP_CF_WOVEN_180].areal_weight` | Lif alan ağırlığı (FAW) | `g/m2` |
| 17 | `prepregs[PP_CF_WOVEN_180].cured_ply_thickness` | Kürlenmiş ply kalınlığı (CPT) | `mm` |
| 18 | `prepregs[PP_CF_WOVEN_180].roll_width` | Rulo eni | `mm` |
| 19 | `prepregs[PP_CF_WOVEN_180].out_life` | Oda sıcaklığında çalışma ömrü | `h` |
| 20 | `prepregs[PP_CF_WOVEN_180].price` | Prepreg birim fiyatı | `EUR/m2` |
| 21 | `prepregs[PP_GF_WOVEN_120].areal_weight` | Lif alan ağırlığı (FAW) | `g/m2` |
| 22 | `prepregs[PP_GF_WOVEN_120].cured_ply_thickness` | Kürlenmiş ply kalınlığı (CPT) | `mm` |
| 23 | `prepregs[PP_GF_WOVEN_120].roll_width` | Rulo eni | `mm` |
| 24 | `prepregs[PP_GF_WOVEN_120].out_life` | Oda sıcaklığında çalışma ömrü | `h` |
| 25 | `prepregs[PP_GF_WOVEN_120].price` | Prepreg birim fiyatı | `EUR/m2` |
| 26 | `consumables[VAKUM_TORBA].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 27 | `consumables[VAKUM_TORBA].price` | Vakum torba filmi fiyatı | `EUR/m2` |
| 28 | `consumables[BREATHER].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 29 | `consumables[BREATHER].price` | Breather fiyatı | `EUR/m2` |
| 30 | `consumables[AYIRICI_FILM].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 31 | `consumables[AYIRICI_FILM].price` | Ayırıcı film fiyatı | `EUR/m2` |
| 32 | `consumables[PEEL_PLY].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 33 | `consumables[PEEL_PLY].price` | Peel ply fiyatı | `EUR/m2` |
| 34 | `consumables[SIZDIRMAZLIK_BANDI].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 35 | `consumables[SIZDIRMAZLIK_BANDI].price` | Sızdırmazlık bandı fiyatı | `EUR/m` |
| 36 | `consumables[KALIP_AYIRICI].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 37 | `consumables[KALIP_AYIRICI].price` | Kalıp ayırıcı fiyatı | `EUR/L` |
| 38 | `consumables[BOYA_ASTAR].max_temp` | Maksimum kullanım sıcaklığı | `degC` |
| 39 | `consumables[BOYA_ASTAR].price` | Boya sistemi fiyatı (karışım) | `EUR/L` |
| 40 | `tooling_materials[TB_120].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 41 | `tooling_materials[TB_120].cte` | Isıl genleşme katsayısı | `1e-6/K` |
| 42 | `tooling_materials[TB_120].density` | Yoğunluk | `kg/m3` |
| 43 | `tooling_materials[TB_120].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 44 | `tooling_materials[TB_120].price` | Blok fiyatı (hacim başına) | `EUR/m3` |
| 45 | `tooling_materials[TB_180].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 46 | `tooling_materials[TB_180].cte` | Isıl genleşme katsayısı | `1e-6/K` |
| 47 | `tooling_materials[TB_180].density` | Yoğunluk | `kg/m3` |
| 48 | `tooling_materials[TB_180].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 49 | `tooling_materials[TB_180].price` | Blok fiyatı (hacim başına) | `EUR/m3` |
| 50 | `tooling_materials[CFRP_TOOLING_PREPREG].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 51 | `tooling_materials[CFRP_TOOLING_PREPREG].cte` | Kalıp laminatı CTE (yarı-izotropik) | `1e-6/K` |
| 52 | `tooling_materials[CFRP_TOOLING_PREPREG].density` | Yoğunluk | `kg/m3` |
| 53 | `tooling_materials[CFRP_TOOLING_PREPREG].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 54 | `tooling_materials[CFRP_TOOLING_PREPREG].price` | Tooling prepreg fiyatı (ply alanı başına) | `EUR/m2` |
| 55 | `tooling_materials[GFRP_TOOLING].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 56 | `tooling_materials[GFRP_TOOLING].cte` | Kalıp laminatı CTE | `1e-6/K` |
| 57 | `tooling_materials[GFRP_TOOLING].density` | Yoğunluk | `kg/m3` |
| 58 | `tooling_materials[GFRP_TOOLING].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 59 | `tooling_materials[GFRP_TOOLING].price` | Kalıp laminatı fiyatı (ply alanı başına) | `EUR/m2` |
| 60 | `tooling_materials[INVAR36].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 61 | `tooling_materials[INVAR36].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 62 | `tooling_materials[INVAR36].price` | Plaka/blok fiyatı | `EUR/kg` |
| 63 | `tooling_materials[AL_PLAKA].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 64 | `tooling_materials[AL_PLAKA].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 65 | `tooling_materials[AL_PLAKA].price` | Plaka fiyatı | `EUR/kg` |
| 66 | `tooling_materials[CELIK].max_service_temp` | Maksimum servis sıcaklığı | `degC` |
| 67 | `tooling_materials[CELIK].tool_life` | Beklenen kalıp ömrü (kür çevrimi) | `cycle` |
| 68 | `tooling_materials[CELIK].price` | Plaka/blok fiyatı | `EUR/kg` |

## İş merkezleri ve saat ücretleri — `work_centers.yaml` (48)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `work_centers[KESIM_MASASI].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 2 | `work_centers[KESIM_MASASI].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 3 | `work_centers[KESIM_MASASI].operators` | Tipik operatör sayısı | `person` |
| 4 | `work_centers[KESIM_MASASI].efficiency` | Verim (0-1) | `ratio` |
| 5 | `work_centers[TEMIZ_ODA].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 6 | `work_centers[TEMIZ_ODA].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 7 | `work_centers[TEMIZ_ODA].operators` | Tipik operatör sayısı | `person` |
| 8 | `work_centers[TEMIZ_ODA].efficiency` | Verim (0-1) | `ratio` |
| 9 | `work_centers[AUTOCLAVE_YUKLEME].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 10 | `work_centers[AUTOCLAVE_YUKLEME].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 11 | `work_centers[AUTOCLAVE_YUKLEME].operators` | Tipik operatör sayısı | `person` |
| 12 | `work_centers[AUTOCLAVE_YUKLEME].efficiency` | Verim (0-1) | `ratio` |
| 13 | `work_centers[KALIPTAN_CIKARMA].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 14 | `work_centers[KALIPTAN_CIKARMA].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 15 | `work_centers[KALIPTAN_CIKARMA].operators` | Tipik operatör sayısı | `person` |
| 16 | `work_centers[KALIPTAN_CIKARMA].efficiency` | Verim (0-1) | `ratio` |
| 17 | `work_centers[CNC_TRIM].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 18 | `work_centers[CNC_TRIM].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 19 | `work_centers[CNC_TRIM].operators` | Tipik operatör sayısı | `person` |
| 20 | `work_centers[CNC_TRIM].efficiency` | Verim (0-1) | `ratio` |
| 21 | `work_centers[EL_ISCILIGI].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 22 | `work_centers[EL_ISCILIGI].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 23 | `work_centers[EL_ISCILIGI].operators` | Tipik operatör sayısı | `person` |
| 24 | `work_centers[EL_ISCILIGI].efficiency` | Verim (0-1) | `ratio` |
| 25 | `work_centers[NDT_UT].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 26 | `work_centers[NDT_UT].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 27 | `work_centers[NDT_UT].operators` | Tipik operatör sayısı | `person` |
| 28 | `work_centers[NDT_UT].efficiency` | Verim (0-1) | `ratio` |
| 29 | `work_centers[OLCUM].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 30 | `work_centers[OLCUM].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 31 | `work_centers[OLCUM].operators` | Tipik operatör sayısı | `person` |
| 32 | `work_centers[OLCUM].efficiency` | Verim (0-1) | `ratio` |
| 33 | `work_centers[BOYA].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 34 | `work_centers[BOYA].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 35 | `work_centers[BOYA].operators` | Tipik operatör sayısı | `person` |
| 36 | `work_centers[BOYA].efficiency` | Verim (0-1) | `ratio` |
| 37 | `work_centers[MONTAJ].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 38 | `work_centers[MONTAJ].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 39 | `work_centers[MONTAJ].operators` | Tipik operatör sayısı | `person` |
| 40 | `work_centers[MONTAJ].efficiency` | Verim (0-1) | `ratio` |
| 41 | `work_centers[KALIP_CNC].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 42 | `work_centers[KALIP_CNC].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 43 | `work_centers[KALIP_CNC].operators` | Tipik operatör sayısı | `person` |
| 44 | `work_centers[KALIP_CNC].efficiency` | Verim (0-1) | `ratio` |
| 45 | `work_centers[KALIP_ATOLYE].labor_rate` | İşçilik saat ücreti (kişi başı) | `EUR/h` |
| 46 | `work_centers[KALIP_ATOLYE].machine_rate` | Makine/tesis saat ücreti | `EUR/h` |
| 47 | `work_centers[KALIP_ATOLYE].operators` | Tipik operatör sayısı | `person` |
| 48 | `work_centers[KALIP_ATOLYE].efficiency` | Verim (0-1) | `ratio` |

## Autoclave ölçü ve maliyetleri — `autoclaves.yaml` (9)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `autoclaves[AC1].inner_diameter` | Kullanılabilir iç çap | `mm` |
| 2 | `autoclaves[AC1].usable_length` | Kullanılabilir boy | `mm` |
| 3 | `autoclaves[AC1].usable_width` | Yükleme arabası / raf kullanılabilir eni | `mm` |
| 4 | `autoclaves[AC1].max_temp` | Maksimum sıcaklık | `degC` |
| 5 | `autoclaves[AC1].max_pressure` | Maksimum basınç | `bar` |
| 6 | `autoclaves[AC1].vacuum_ports` | Vakum port sayısı | `pcs` |
| 7 | `autoclaves[AC1].thermocouples` | Termokupl kanal sayısı | `pcs` |
| 8 | `autoclaves[AC1].hourly_rate` | Çevrim saat maliyeti | `EUR/h` |
| 9 | `autoclaves[AC1].load_efficiency` | Taban alanının tipik doluluk oranı (0-1) | `ratio` |

## Genel katsayılar — `thresholds.yaml` (8)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `costing.nesting_scrap_ratio` | Varsayılan nesting firesi (fire alanı / net ply alanı) | `ratio` |
| 2 | `costing.learning_curve_ratio` | Öğrenme eğrisi oranı LC (örn. 0.85); T_n = T_ref × (n/ref)^log2(LC) | `ratio` |
| 3 | `costing.overhead_ratio` | Genel gider oranı (overhead_basis üzerine) | `ratio` |
| 4 | `costing.margin_ratio` | Kâr oranı (margin_basis'e göre) | `ratio` |
| 5 | `costing.consumable_waste_ratio` | Sarf malzeme kesim/bindirme payı | `ratio` |
| 6 | `costing.autoclave_spacing` | Autoclave yerleşiminde kalıp çevresine bırakılan boşluk (her kenar) | `mm` |
| 7 | `complexity_factors.orta` | Orta karmaşıklık süre çarpanı | `ratio` |
| 8 | `complexity_factors.yuksek` | Yüksek karmaşıklık süre çarpanı | `ratio` |

## Standart süreler ve sarf kullanımı (processes/autoclave_prepreg.yaml) — `processes/autoclave_prepreg.yaml` (39)

| # | Anahtar | Açıklama | Birim |
|---|---|---|---|
| 1 | `parameters.debulk_interval_plies` | Kaç ply'da bir ara debulk yapılır | `ply` |
| 2 | `cure_cycles[cure_120].pressure` | Autoclave basıncı | `bar` |
| 3 | `cure_cycles[cure_120].cycle_time` | Toplam çevrim süresi (ısıtma + bekleme + soğutma) | `h` |
| 4 | `cure_cycles[cure_180].pressure` | Autoclave basıncı | `bar` |
| 5 | `cure_cycles[cure_180].cycle_time` | Toplam çevrim süresi (ısıtma + bekleme + soğutma) | `h` |
| 6 | `operations[kesim].setup` | Nesting yükleme, rulo hazırlama, masa hazırlığı | `min/batch` |
| 7 | `operations[kesim].times[ply_area_total_m2].rate` | Kesim süresi, ply alanı başına | `min/m2` |
| 8 | `operations[kesim].times[ply_count].rate` | Etiketleme ve kitleme, ply başına | `min/ply` |
| 9 | `operations[serim].setup` | Kalıp temizliği ve ayırıcı uygulaması | `min/batch` |
| 10 | `operations[serim].times[ply_area_total_m2].rate` | Serim süresi, ply alanı başına | `min/m2` |
| 11 | `operations[serim].times[ply_count].rate` | Ply konumlandırma (laser projeksiyon/şablon), ply başına | `min/ply` |
| 12 | `operations[serim].consumables[KALIP_AYIRICI].qty` | Kalıp ayırıcı sarfiyatı, kalıp alanı başına | `L/m2` |
| 13 | `operations[debulk].times[debulk_count].rate` | Debulk başına süre (torba kurma + vakum bekleme + söküm) | `min/debulk` |
| 14 | `operations[torbalama].times[tool_area_m2].rate` | Torbalama süresi, kalıp alanı başına | `min/m2` |
| 15 | `operations[torbalama].times[tool_perimeter_m].rate` | Sızdırmazlık bandı ve kenar hazırlığı, çevre başına | `min/m` |
| 16 | `operations[torbalama].consumables[PEEL_PLY].qty` | Peel ply kullanımı, parça alanı başına | `m2/m2` |
| 17 | `operations[torbalama].consumables[AYIRICI_FILM].qty` | Ayırıcı film kullanımı, kalıp alanı başına | `m2/m2` |
| 18 | `operations[torbalama].consumables[BREATHER].qty` | Breather kullanımı, kalıp alanı başına | `m2/m2` |
| 19 | `operations[torbalama].consumables[VAKUM_TORBA].qty` | Vakum torba filmi (pile payı dahil), kalıp alanı başına | `m2/m2` |
| 20 | `operations[torbalama].consumables[SIZDIRMAZLIK_BANDI].qty` | Sızdırmazlık bandı, çevre başına | `m/m` |
| 21 | `operations[kur].setup` | Yükleme, termokupl/vakum bağlantısı, boşaltma | `min/batch` |
| 22 | `operations[kur].times[per_part].rate` | Parça başına bağlantı ve kontrol | `min/part` |
| 23 | `operations[kaliptan_cikarma].times[part_area_m2].rate` | Sarf söküm ve kalıptan ayırma, parça alanı başına | `min/m2` |
| 24 | `operations[kaliptan_cikarma].times[tool_area_m2].rate` | Kalıp temizliği, kalıp alanı başına | `min/m2` |
| 25 | `operations[trim].setup` | Fikstür bağlama, sıfırlama, program yükleme | `min/batch` |
| 26 | `operations[trim].times[part_perimeter_m].rate` | Trim süresi, kenar uzunluğu başına | `min/m` |
| 27 | `operations[trim].times[hole_count].rate` | Delik başına süre | `min/hole` |
| 28 | `operations[trim].times[per_part].rate` | Parça bağlama/sökme | `min/part` |
| 29 | `operations[el_isciligi].times[part_perimeter_m].rate` | Kenar işçiliği, kenar uzunluğu başına | `min/m` |
| 30 | `operations[ndt].setup` | Cihaz kalibrasyonu, referans blok | `min/batch` |
| 31 | `operations[ndt].times[part_area_m2].rate` | Tarama süresi, parça alanı başına | `min/m2` |
| 32 | `operations[ndt].times[per_part].rate` | Değerlendirme ve raporlama | `min/part` |
| 33 | `operations[olcum].setup` | Program ve fikstür hazırlığı | `min/batch` |
| 34 | `operations[olcum].times[per_part].rate` | Parça başına ölçüm | `min/part` |
| 35 | `operations[olcum].times[hole_count].rate` | Delik konum ölçümü, delik başına | `min/hole` |
| 36 | `operations[boya].setup` | Kabin ve tabanca hazırlığı | `min/batch` |
| 37 | `operations[boya].times[part_area_m2].rate` | Yüzey hazırlama + maskeleme + boyama, alan başına | `min/m2` |
| 38 | `operations[boya].consumables[BOYA_ASTAR].qty` | Boya sistemi sarfiyatı, alan başına | `L/m2` |
| 39 | `operations[montaj].times[per_part].rate` | Parça başına montaj süresi (komple için Faz 5'te ayrıntılanacak) | `min/part` |
