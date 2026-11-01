# Oturum 9 — Bağımsız Kapsamlı Denetim Raporu (SysRev-Meta-Analysis)

* **Denetlenen commit:** `main` = `df1a79a72d26dd5b41d0cde341472c47e5f5bc6c` (PR #13 merge). Yerel klon `git fetch` ile doğrulandı; çalışma ağacı temiz.
* **Ortam:** Bulut konteyneri, Python 3.13.18. Kullanıcının belirttiği Windows yoluna (`C:\Users\abidi\...\work\repo`) bu ortamdan erişilemedi; denetim aynı commit'in temiz bir klonu üzerinde yapıldı.
* **Kapsam dışı / yapılmayanlar:** Kaynak kod değiştirilmedi, merge yapılmadı, Oturum 10 başlatılmadı. Bu dosya commit edilmemiş, izlenmeyen (untracked) bir rapor dosyasıdır.

## 0. Temel doğrulama

| Kontrol | Sonuç |
|---|---|
| `pytest --cov=src` | **902 geçti, 1 atlandı, 6 deselect** (atlanan: R/metafor yerelde kurulu değil; CI'daki `r-validation` işi bunu çalıştırıyor). Kapsam **%93** |
| `ruff check` / `ruff format --check` | Temiz |
| `mypy` (strict) | 129 dosyada sorun yok |
| Kapsamı düşük modüller | `api_server/amendment_routes.py` %71 (karar, yükümlülük çözümü, MeSH, attest ve preview rotalarının HTTP testi yok), `ui/app.py` %75, `ui/operations_pages.py` %76, `workflows/ai_steps.py` %75 |

Aşağıdaki **Yüksek** ve **Orta** bulguların hepsi çalıştırılan testlerle doğrulandı. Ek A'daki 10 test doğru davranışı tanımlıyor ve `df1a79a` üzerinde **10'u da başarısız oluyor**. **Düşük** bulgular kod incelemesiyle doğrulandı; her birinde bu ayrıca belirtilmiştir.

---

## 1. Özet tablo

| ID | Önem | Alan | Kısa tanım |
|---|---|---|---|
| H1 | **Yüksek** | Kaynak izlenebilirliği / sürüm geçersizleştirme | Kol (arm) oluşturulduktan sonra kabul edilen ya da REJECTED durumundan düzeltilen değerler onaylı kola hiç girmiyor; hiçbir guard bunu fark etmiyor |
| H2 | **Yüksek** | İnsan onayı (Gate 3) | Gate 3 sonrasında (Stage 5) yeni değerler önerilip kabul edilebiliyor, onaylı kollar oluşturulabiliyor ve tam metin kararı değiştirilebiliyor. Stage 5→6 guard'ı Gate 3 koşullarını yeniden denetlemiyor |
| H3 | **Yüksek** | Kaynak izlenebilirliği / eskime koruması | Bağlı PDF her fazda (Stage 6 dahil) değiştirilebiliyor. Kabul edilmiş öğeler ve kollar eski PDF'e bağlı kalıyor. Bekleyen bir öğe eski PDF'e göre yapılmış doğrulamayla kabul edilebiliyor |
| M1 | Orta | İstatistik (KM) | `plain` ölçekte GA alt sınırı 0 olan tek bir çalışma tüm analizi `not_estimable` yapıyor |
| M2 | Orta | İstatistik (KM) | Simetri kontrolündeki yuvarlama toleransı en çok ondalığa göre seçiliyor; tutarlı çalışmalar yanlışlıkla `ci_inconsistent` sayılıyor |
| M3 | Orta | Raporlama doğruluğu | "Weight CE %" sütunu MH modelinde IV ağırlıklarını gösteriyor; oranlarda GLMM özetinin yanında IV-logit RE ağırlıkları gösteriliyor |
| M4 | Orta | Eskime / UI | Önceki çalıştırmanın forest plot SVG'leri silinmiyor; UI geçersiz istatistiklerin yanında eski grafikleri gösteriyor |
| M5 | Orta | Ayrıştırma / izlenebilirlik | `parse_reported_number("0,500")` = 500. İnceleme ekranları ayrıştırılmış değeri göstermiyor |
| L1 | Düşük | İnsan onayı (derinlemesine savunma) | `save_extracted_arm`, kimlik doğrulaması olmadan `approved_by_human=True` kabul ediyor |
| L2 | Düşük | Eskime koruması (API) | `build_arms` hiçbir gösterilen duruma bağlı değil; bazı API gövdelerinde `expected_*` isteğe bağlı |
| L3 | Düşük | Tarama | Dışa aktarılan `final_verdicts`, insan hakem kararına karşı otomatik uzlaşıyı öne alıyor (`screening_status`'un tersi) |
| L4 | Düşük | Maliyet | Önbellek yazma token'ları girdi fiyatıyla hesaplanıp "üst sınır" diye etiketleniyor (yazma fiyatı girdiden yüksek) |
| L5 | Düşük | Çifte sayım tutarlılığı | Guard serbest metin `timepoint`'e göre gruplama yapıyor, motor yapılandırılmış `follow_up`'a göre |
| L6 | Bilgi | Dokümantasyon | `numerics.py` PI kaynağı olarak Higgins 2009'u gösteriyor; uygulanan yöntem HKSJ-SE + t(k−2) (metafor `pi.type="Riley"`) |

---

## 2. Yüksek önemdeki bulgular

### H1 — Kol oluşturulduktan sonra kabul edilen veya düzeltilen değerler kola hiç girmiyor

* **Konum:**
  * `src/extraction/master_table.py:449` `build_arms`. Satır 482: `if next_id is None: continue`. Geçerli bir kol varsa yeni kabul edilmiş öğeler sessizce atlanıyor.
  * `src/extraction/master_table.py:399` `revise_reviewed_item`. Kol yalnızca `old.review_status == ACCEPTED` ise geçersizleştiriliyor. REJECTED bir öğenin düzeltilmesi kolu geçersizleştirmiyor.
  * `src/state/repository.py:2348` `build_guard_context`. Yalnızca `PENDING` öğeleri ve hiç kolu olmayan raporları sayıyor. "Kabul edilmiş ama geçerli bir kola yansımamış öğe" diye bir guard yok.
* **Sorun:** (a) Kol `reported_estimate` ile oluşturulduktan sonra aynı `arm_key` için sonradan önerilip kabul edilen `ci_lower/ci_upper/n_at_risk/events` gibi değerler kola eklenmiyor; `build_arms` `[]` döndürüyor. (b) REJECTED durumundaki bir değer `revise_reviewed_item` ile düzeltilip ACCEPTED olsa da kol geçersizleştirilmiyor (`invalidated == []`). Ana Çıkarım Tablosu ve dışa aktarım kabul edilmiş değeri gösteriyor; istatistik motoru ise eski kolu kullanıyor.
* **Etki:** Örneğin bir KM kolunun GA'sı sonradan kabul edilirse motor çalışmayı "GA yok" diye havuzlamıyor. Düzeltilmiş olay sayısı analize hiç girmiyor. Gate 3 geçiyor, `stats_output.json` girdi hash'i tutarlı kalıyor. Bu, "her sayı izlenebilir" ilkesini analiz düzeyinde bozuyor.
* **Yeniden üretim:** Ek A `test_h1_value_accepted_after_build_reaches_the_arm_or_blocks_gate3`, `test_h1_correcting_a_rejected_value_invalidates_the_arm` (master-table fikstürü, sentetik PDF).
* **Düzeltme yönü:** Aynı `arm_key` için kabul edilmiş ama geçerli kolda bulunmayan bir değer olduğunda kolu geçersizleştirin (veya `build_arms` yeniden oluştursun). REJECTED→ACCEPTED düzeltmesinde de geçersizleştirin. Gate 3'e "kabul edilmiş kol-alanı öğesi geçerli bir kola yansımamış" guard'ı ekleyin.

### H2 — Gate 3 onayından sonra veri tabanı değiştirilebiliyor; Stage 5→6 bunu denetlemiyor

* **Konum:**
  * `src/extraction/master_table.py:139` `submit_proposals`, `:210` `accept_item`, `:449` `build_arms`: faz kısıtlaması yok. Yalnızca `revise_reviewed_item` `REVISION_PHASES` (`:56`, Stage 4/Gate 3) ile sınırlı.
  * `src/state/repository.py:950` `require_stats_inputs_open` yalnızca Stage 6 ve sonrasını engelliyor. `:510` `save_screening_decision` Stage 5'te tam metin hakem kararına izin veriyor.
  * `src/state/machine.py:242` `_stats_valid` ve `:288`: Stage 5→6 yalnızca `stats_output.json`'u denetliyor.
* **Sorun:** Stage 5'te yeni öneri gönderilip kabul edilebiliyor ve `build_arms` yeni bir **onaylı** kol üretiyor. Gate 3'ün yeniden onayı gerekmiyor. Stage 5'te `adjudicate(FULL_TEXT)` ile, R1/R2 uzlaşısıyla dışlanmış bir rapor dahil edilebiliyor (RoB yok, kol yok, PRISMA anlık görüntüsü artık tutarsız). Stage 5 yeniden çalıştırıldığında Stage 6'ya geçiliyor. Stage 6'da `accept_item` hâlâ Ana Çıkarım Tablosu'nu değiştirebiliyor.
* **Kod içi çelişki:** `REVISION_PHASES` gerekçesi "sonraki fazlarda Gate 3'ün kolların yeniden onayını zorlayacak yolu yok" diyor. Aynı gerekçe yeni öğe ve kol oluşturmaya uygulanmamış.
* **Etki:** G3 ("insanlar her kapının sahibidir") Stage 5'te atlanıyor. Gate 3 koşulları (her dahil çalışma için onaylı RoB, nihai PRISMA, bekleyen öğe olmaması) Stage 6 öncesinde yeniden doğrulanmıyor.
* **Yeniden üretim:** Ek A `test_h2_no_new_approved_arms_after_gate3`.
* **Düzeltme yönü:** Gate 3'ten sonra öneri, kabul, red, kol oluşturma ve tam metin kararlarını reddedin. Ya da en azından Stage 5→6 guard'ına `_extraction_approved` koşullarını ekleyin. Stage 5'teki bağlantı (link) değişiklikleri bilinçli olarak destekleniyor; bu ayrım belgelenmeli.

### H3 — PDF değiştirme, kabul edilmiş kanıtı ve onaylı kolları geçersiz kılmıyor

* **Konum:** `src/fulltext/retrieval.py:272` `register_upload`, `:320` `confirm_pdf_match` (faz kontrolü yok, çıkarım öğelerine dokunmuyor). `src/extraction/master_table.py:210` `accept_item` yalnızca kayıtlı `verification.acceptable` değerine bakıyor; doğrulamanın hangi PDF'e karşı yapıldığını kontrol etmiyor. Guard'larda `SourceEvidence.pdf_sha256` ile bağlı PDF karşılaştırılmıyor (yalnızca `manuscript/evidence.py:633` referans doğrulamaları için yapılıyor).
* **Sorun:** Stage 6'da bile r1 için farklı bir PDF yüklenip insan onayıyla bağlanabiliyor. Onaylı kolun kanıtı eski PDF hash'ini taşımaya devam ediyor. Kol değişmediği için istatistik girdi hash'i de değişmiyor. Eski PDF'e göre doğrulanmış bekleyen bir öğe, PDF değiştirildikten sonra kabul edilebiliyor (`accepted.evidence.pdf_sha256 == eski_sha`).
* **Etki:** Analizdeki bir sayının kaynağı artık rapora bağlı belge değil. Denetimde "bu sayı bu PDF'in şu sayfasında" iddiası sessizce yanlışlanabiliyor.
* **Yeniden üretim:** Ek A `test_h3_pdf_replacement_after_stage5_is_refused`, `test_h3_pending_item_verified_against_old_pdf_cannot_be_accepted`.
* **Düzeltme yönü:** Kabul edilmiş öğesi veya kolu olan bir raporun PDF'i değiştirildiğinde: `reverify_items` çalıştırılsın, başarısız öğeler yeniden incelemeye düşsün ve ilgili kollar geçersizleşsin. Stage 5 ve sonrasında değiştirme reddedilsin. `accept_item`, `verification.checked_against_sha256` / `evidence.pdf_sha256` değerinin bağlı PDF'e eşit olmasını şart koşsun. Guard'a "kanıtı bağlı PDF'e ait olmayan kol değeri" eklensin.

---

## 3. Orta önemdeki bulgular

### M1 — KM/CIF `plain` ölçek: GA sınırı 0 olan tek çalışma tüm analizi düşürüyor
* **Konum:** `src/stats_engine/engine.py:948` (`plain` için sınır kontrolü atlanıyor) → `:603` `es.to_scale(lo, "plain")` p = 0 için `EstimationError` fırlatıyor → `run_analysis` tüm analizi `not_estimable` yapıyor ve her rapor `analysis_failed` oluyor.
* **Yeniden üretim:** s1 40,0 (30,0–50,0), s2 50,0 (42,0–58,0), s3 6,0 (0,0–12,0); `reported_ci_scale="plain"` → `not_estimable`. s3 çıkarılınca `pooled`. Ek A `test_m1_...`.
* **Düzeltme yönü:** Hatalı çalışma birim düzeyinde dışlanmalı (`boundary_estimate`/`ci_inconsistent`). Analiz düzeyinde istisna fırlatılmamalı.

### M2 — Yuvarlama toleransı her basılı sayının kendi hassasiyetine göre değil
* **Konum:** `src/stats_engine/engine.py:923–931` `_probabilities`: `dec = max(...)`, yani üç sayı için de en küçük yarım birim (h) kullanılıyor.
* **Sorun:** Kestirim "62" (±0,5) ve GA "56,3–67,8" (±0,05) olarak basıldığında, gerçek değer 62,34 % ve GA log(−log) ölçeğinde simetrik olduğu halde, kestirime ±0,05 uygulandığı için çalışma `ci_inconsistent` sayılıyor. Gerekçe metni de yanıltıcı ("yazarlar muhtemelen başka yöntem kullandı").
* **Yeniden üretim:** Ek A `test_m2_...`.
* **Düzeltme yönü:** Her sayıya kendi `h` değeri verilsin (p için h_p, L için h_L, U için h_U); köşe değerlendirmesi buna göre yapılsın.

### M3 — Ağırlıklar, gösterildikleri modele ait değil
* **Konum:** `src/stats_engine/engine.py:294–320` (`weight_common_pct` = IV ortak etki, `weight_random_pct` = IV rastgele etki); `src/stats_engine/report.py:79–88` (tablo başlığı "Weight CE % / Weight RE %"), `:302–307` (forest plot kutu boyutu ve % etiketi).
* **Sorun:** RR/OR'da ortak etki modeli Mantel-Haenszel iken "Weight CE %" IV ağırlıklarını gösteriyor. Örnek (comparative_project): IV 47,90/32,51/19,58; MH ağırlıkları wᵢ = cᵢ·n₁ᵢ/Nᵢ (Greenland & Robins 1985) 45,49/33,70/20,82. Tek grup oranlarında birincil rastgele etki modeli binom-normal GLMM (Stijnen ve ark. 2010) iken "Weight RE %" ve forest plot, 0,5 düzeltmeli IV-logit modelinin ağırlıklarını GLMM elmasının yanında gösteriyor. GLMM'in kapalı formda çalışma ağırlığı yok.
* **Etki:** Yayımlanacak tablo ve şekillerde yanlış atfedilmiş sayılar.
* **Yeniden üretim:** Ek A `test_m3_...`.
* **Düzeltme yönü:** MH için MH ağırlıkları gösterilsin, ya da sütun "IV CE" olarak açıkça etiketlensin. GLMM birincilse RE ağırlık sütunu ve kutu boyutları "IV-logit duyarlılık modeli" olarak etiketlensin veya hiç gösterilmesin.

### M4 — Eski forest plot'lar diskte kalıyor ve UI'da güncelmiş gibi görünüyor
* **Konum:** `src/stats_engine/stage5.py:55–75` (`plots/` dizini temizlenmiyor); `src/stats_engine/report.py:289` (pooled/single_study olmayan analiz için `None`); `src/ui/app.py:1101` (dizindeki tüm `*.svg` dosyaları, `stats_output` sorunlu olsa bile gösteriliyor ve `manifest.json` ile karşılaştırılmıyor).
* **Yeniden üretim:** Ek A `test_m4_...`: ikinci yazımda `comp.svg` manifest'te yok ama diskte duruyor.
* **Düzeltme yönü:** Yazmadan önce `plots/` temizlensin veya yalnızca manifest'teki dosyalar gösterilsin. `problems` boş değilse grafikler "geçersiz" olarak işaretlensin.

### M5 — Basılı sayı ayrıştırıcısı "0,500" değerini 500 okuyor; inceleme ekranı değeri göstermiyor
* **Konum:** `src/core/schemas/extraction.py:166` `_THOUSANDS_RE`, `:173` `parse_reported_number`. `src/ui/app.py` extraction tablosu ve `src/ui/operations_pages.py:327` yalnızca "as printed" metnini gösteriyor; ayrıştırılmış `value` gösterilmiyor.
* **Sorun:** Baştaki "0," grubu binlik ayırıcı olamaz; Avrupa ondalık yazımında "0,500" = 0,5. Değer 1000 kat hatalı saklanıyor (ör. SD) ve şema bunu "sözcüğü sözcüğüne kopya" olarak kabul ediyor. "1,250" gibi gerçekten belirsiz biçimler de sessizce binlik kabul ediliyor.
* **Yeniden üretim:** Ek A `test_m5_...`.
* **Düzeltme yönü:** `0,ddd` ondalık sayılsın. Belirsiz `d,ddd` biçimleri işaretlensin (flag) ve not olmadan kabul edilemesin. İnceleme ekranlarında ayrıştırılmış değer gösterilsin.

---

## 4. Düşük önem / bilgi

* **L1 — `save_extracted_arm` onay kimliğini doğrulamıyor** (`src/state/repository.py:843`). `approved_by_human=True` ve rastgele bir `approved_by` ile kol kaydedilebiliyor. `save_rob`, `save_protocol` ve `save_analysis_plan` kayıtlı gözden geçirmeni doğruluyor; bu metot doğrulamıyor. Yalnızca süreç içi çağrıyla erişilebilir (§6 tehdit modeli bunu kapsam dışı sayıyor), bu yüzden bir derinlemesine savunma tutarsızlığıdır. *Kod incelemesiyle doğrulandı.*
* **L2 — Eskime bağlaması eksik veya isteğe bağlı** (`src/api_server/review_routes.py:366` `POST .../arms`, kol onayını gösterilen duruma bağlamıyor; UI'da da bağlı değil. `:105` `StudyLinkBody` ve `:139` `DoubleCountingBody` içinde `expected_state_sha256` isteğe bağlı. `:535` eski `POST /stats/plan` protokol/son plan bağlaması olmadan onay veriyor; bu yol "legacy" olarak belgelenmiş). H1 ve H3 ile birleşince kol "onayı" gözden geçirmenin görmediği öğeleri kapsayabiliyor. *Kod incelemesiyle doğrulandı.*
* **L3 — `final_verdicts` önceliği ters** (`src/screening/agreement.py:115–138`, `:131`). R1/R2 uzlaşısı varsa insan hakem kararını yok sayıyor. `screening_status` (guard'lar, PRISMA) ise hakem kararını öne alıyor. Fonksiyon `src/screening/__init__.py` üzerinden dışa aktarılıyor ama `src` içinde kullanılmıyor (gizli tuzak). *Kod incelemesiyle doğrulandı.*
* **L4 — Maliyet etiketi** (`src/llm/catalogue.py:273, 296`). Fiyatı tanımsız önbellek token'ları girdi fiyatıyla hesaplanıp "upper bound" diye etiketleniyor. Bu okuma token'ları için doğru, yazma token'ları için yanlış: katalogdaki kısa katmanda yazma fiyatı girdiden yüksek (ör. `gpt-5.6-sol` 5,0 > 4,0). Uzun katmanda `long_prompt_cache_write` tanımsız. *Kod ve katalog incelemesiyle doğrulandı.*
* **L5 — Çifte sayım: guard ile motor farklı anahtar kullanıyor** (`src/state/repository.py:1389` serbest metin `timepoint`; `src/stats_engine/inputs.py` `_group_cohorts` yapılandırılmış `follow_up`). Aynı kohort için "5 years" ve "5-year" gibi metinler guard'ı geçiyor; motor her iki raporu insan kararı olmadan `overlapping_cohort` diye dışlıyor. Sonuç tutucu (çifte sayım yok), ama Gate 3 sessizce geçiyor ve veri kaybı ancak defterde (ledger) görülüyor. *Kod incelemesiyle doğrulandı.*
* **L6 — PI dokümantasyonu** (`src/stats_engine/numerics.py` modül docstring'i). Higgins, Thompson & Spiegelhalter 2009 gösteriliyor. Uygulanan μ̂ ± t(k−2)·√(τ̂² + SE²) formülünde HKSJ seçiliyse SE de HKSJ SE'si oluyor; bu, VALIDATION_REPORT'ta doğru belirtildiği gibi metafor `pi.type="Riley"` ile aynı. Sayısal hata değil; yalnızca kaynak gösterimi düzeltilmeli.

---

## 5. Mevcut sınırlamalar (kusur değil; belgelenmiş veya bilinçli tasarım)

1. HR, RD, IRR, oranlar, ortalama/medyan ve medyan olay süresi `unsupported` olarak raporlanıyor. Meta-regresyon, yayın yanlılığı testleri, leave-one-out ve Bayesçi modeller yok (ARCHITECTURE §10).
2. KM/CIF havuzlaması yazarların GA'sının beyan edilen ölçekte simetrik olduğunu varsayıyor. `1−S` olarak raporlanan ama GA'sı log(−log S) ölçeğinde hesaplanmış "olay olasılıkları" için ölçek seçeneği (cloglog-tamamlayıcı) yok; bu çalışmalar tutucu biçimde `ci_inconsistent` sayılıyor.
3. GLMM GA'sı Wald aralığıdır; az çalışmada düşük kapsama gösterebilir (belgelenmiş).
4. Canlı literatür API'leri ve gerçek LLM sağlayıcıları test edilmedi. UI Windows'ta çalıştırılmadı. PDF ayrıştırma yalnızca sentetik PDF'lerde test edildi.
5. Tehdit modeli: aynı OS kullanıcısıyla süreç içi kütüphane çağrıları ve doğrudan SQLite yazımları kapsam dışı (ARCHITECTURE §6).
6. Yerelde R bulunmadığından metafor/lme4 karşılaştırması bu denetimde yeniden çalıştırılamadı. Kayıtlı referans JSON ile yapılan karşılaştırmalar geçti; canlı R karşılaştırması CI'da çalışıyor.
7. Doğrulanmış olanlar: MH OR (RBG varyansı) ve MH RR (Greenland-Robins varyansı) formülleri, escalc kuralları (`add=1/2, to="only0"`, `drop00`), Hedges' g'nin tam J'si ve LS varyansı, Q/I²/H², REML (global maksimum), Q-profil GA'ları, HKSJ ve ad hoc HKSJ, Clopper-Pearson, binom ML ortak etki ve SoF mutlak etki formülleri (RR: ACR×RR; OR: ACR·OR/(1−ACR+ACR·OR); GRADE 12). Kodda sayısal bir hata bulunmadı.

---

## 6. Test kapsamı boşlukları

* H1–H3 ve M1–M5 senaryolarının hiçbiri için test yok (Ek A bu boşluğu dolduruyor).
* `amendment_routes.py` karar, yükümlülük çözümü, MeSH, attest ve preview rotalarının HTTP düzeyinde testi yok (%71).
* GLMM-birincil oran analizinde forest plot/tablo ağırlık etiketleri ve MH ağırlıkları test edilmiyor.
* UI'daki istatistik sayfasının grafikleri manifest/geçerlilikle ilişkilendirip ilişkilendirmediği test edilmiyor.

---

## Ek A — Doğru davranışı tanımlayan testler (`df1a79a` üzerinde 10/10 başarısız)

Önerilen yer: `tests/unit/test_session9_audit.py`. Yalnızca mevcut fikstür ve yardımcıları kullanıyor; tüm veriler SENTETİK.

```python
"""Session 9 audit: tests that state the CORRECT behaviour. Each one fails on main df1a79a."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.core.enums import Phase
from src.core.exceptions import SysRevError
from src.core.schemas.extraction import parse_reported_number
from src.core.schemas.stats import AnalysisSpec
from src.extraction import (
    ReviewError, build_arms, correction_state_sha256, reject_item, revise_reviewed_item,
)
from src.fulltext import register_upload
from src.stats_engine.engine import run_analysis
from src.stats_engine.stage5 import write_artifacts
from tests.manuscript_helpers import comparative_project
from tests.stats_helpers import OUTCOMES, YEARS_5, inputs, km_arm
from tests.unit.test_master_table import OTHER_PDF, PID, Project, crude, km
from tests.unit.test_master_table import project  # noqa: F401


def _arm(repo, rid="r1"):
    (a,) = [a for a in repo.load_extracted_arms(PID) if a.record_id == rid]
    return a


# H1
def test_h1_value_accepted_after_build_reaches_the_arm_or_blocks_gate3(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    project.accept(risk)
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert _arm(repo).n_at_risk is not None  # today: None


def test_h1_correcting_a_rejected_value_invalidates_the_arm(project: Project) -> None:
    repo = project.repo
    est, risk = project.submit(km("reported_estimate", "62.1", 1, 1, 3),
                               km("n_at_risk", "22", 2, 1, 3))
    project.accept(est)
    reject_item(repo, PID, risk.item_id, reviewer=project.reviewer, note="wrong column")
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
    repo.session.flush()
    shown = correction_state_sha256(repo, PID, repo.get_extraction_item(PID, risk.item_id))
    _, invalidated = revise_reviewed_item(
        repo, PID, risk.item_id, km("n_at_risk", "40", 2, 1, 3), project.doc("r1"),
        reviewer=project.reviewer, note="correct column", expected_state_sha256=shown)
    assert invalidated  # today: []


# H2
def test_h2_no_new_approved_arms_after_gate3(project: Project) -> None:
    repo = project.repo
    repo.get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
    repo.session.flush()
    with pytest.raises((ReviewError, SysRevError)):
        items = project.submit(crude("events", "45"), crude("n_analyzed", "57"))
        for i in items:
            project.accept(i)
        build_arms(repo, PID, "r1", reviewer=project.reviewer)  # today: succeeds


# H3
def test_h3_pdf_replacement_after_stage5_is_refused(project: Project) -> None:
    repo = project.repo
    repo.get_project(PID).phase = Phase.STAGE6_MANUSCRIPT.value
    repo.session.flush()
    with pytest.raises(SysRevError):  # today: the new PDF is linked
        register_upload(repo, project.store, PID, "r1", OTHER_PDF,
                        original_filename="other.pdf", reviewer=project.reviewer)


def test_h3_pending_item_verified_against_old_pdf_cannot_be_accepted(project: Project) -> None:
    repo = project.repo
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    register_upload(repo, project.store, PID, "r1", OTHER_PDF,
                    original_filename="other.pdf", reviewer=project.reviewer)
    with pytest.raises(ReviewError):  # today: accepted
        project.accept(risk)


# M1 / M2
def _km_spec(scale: str) -> AnalysisSpec:
    return AnalysisSpec(analysis_id="k", outcome_id="control_km",
                        effect_measure="KM_PROBABILITY", timepoints=[YEARS_5],
                        reported_ci_scale=scale)


def test_m1_one_plain_ci_at_zero_excludes_only_that_study() -> None:
    arms = [km_arm("s1", "40.0", "30.0", "50.0"), km_arm("s2", "50.0", "42.0", "58.0"),
            km_arm("s3", "6.0", "0.0", "12.0")]
    res = run_analysis(_km_spec("plain"), OUTCOMES["control_km"], inputs(arms))
    assert res.status == "pooled"  # today: not_estimable, every report analysis_failed
    assert {r.record_id: r.status for r in res.reports}["s3"] == "excluded"


def test_m2_rounding_tolerance_is_per_printed_number() -> None:
    # true estimate 62.34 % with a log(-log) CI (SE 0.10), printed as "62" and "56.3"-"67.8"
    arms = [km_arm("s1", "62", "56.3", "67.8"), km_arm("s2", "71.0", "65.4", "75.9")]
    res = run_analysis(_km_spec("log_log"), OUTCOMES["control_km"], inputs(arms))
    s1 = {r.record_id: r for r in res.reports}["s1"]
    assert s1.status == "included"  # today: ci_inconsistent


# M3 / M4
def test_m3_common_effect_weights_belong_to_the_mantel_haenszel_model(session: Session) -> None:
    repo, _, _ = comparative_project(session)
    a = repo.current_stats_output("mc1").analyses[0]
    assert a.common_effect.method.startswith("Mantel-Haenszel")
    mh = [s.data["events_control"] * s.data["n_intervention"]
          / (s.data["n_intervention"] + s.data["n_control"]) for s in a.studies]
    expected = [100 * w / sum(mh) for w in mh]
    assert [s.weight_common_pct for s in a.studies] == pytest.approx(expected, abs=1e-9)


def test_m4_rewriting_artifacts_removes_plots_of_the_previous_run(
    session: Session, tmp_path: Path
) -> None:
    repo, _, _ = comparative_project(session)
    out = repo.current_stats_output("mc1")
    write_artifacts(out, tmp_path, render_plots=True)
    no_plot = out.analyses[0].model_copy(update={"status": "no_data", "studies": []})
    write_artifacts(out.model_copy(update={"analyses": [no_plot]}), tmp_path, render_plots=True)
    assert not (tmp_path / "plots" / "comp.svg").exists()  # today: stale SVG kept


# M5
def test_m5_leading_zero_group_is_not_a_thousands_separator() -> None:
    assert parse_reported_number("0,500") == 0.5  # today: 500.0
```

## Kaynaklar

1. Mantel N, Haenszel W. *J Natl Cancer Inst* 1959;22:719-748.
2. Greenland S, Robins JM. Estimation of a common effect parameter from sparse follow-up data. *Biometrics* 1985;41:55-68.
3. Robins J, Breslow N, Greenland S. *Biometrics* 1986;42:311-323.
4. Stijnen T, Hamza TH, Özdemir P. Random effects meta-analysis of event outcome in the framework of the generalized linear mixed model. *Stat Med* 2010;29:3046-3067.
5. Higgins JPT, Thompson SG, Spiegelhalter DJ. *J R Stat Soc A* 2009;172:137-159.
6. Viechtbauer W. Conducting meta-analyses in R with the metafor package. *J Stat Softw* 2010;36(3):1-48.
7. Guyatt GH, et al. GRADE guidelines 12. *J Clin Epidemiol* 2013;66:158-72.
8. Higgins JPT, et al. (eds). *Cochrane Handbook for Systematic Reviews of Interventions*, v6, §10.4.
