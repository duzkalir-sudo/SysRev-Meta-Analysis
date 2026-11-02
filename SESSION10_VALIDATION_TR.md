# Oturum 10 — Gerçek dünya Windows doğrulaması: durum raporu

**Durum: TAMAMLANMADI.** Gerçek bir derlemenin uçtan uca pilotu (gerçek arama, PDF'ler,
canlı model çağrıları, insan kararları ve makale) **yürütülmedi**. Bu oturumun bulut
ortamının Windows'a erişimi hâlâ **yoktur**. `f2de5e6` commit'i, sizin bağımsız incelemenizde
gerçek bir Windows bilgisayarında çalıştırıldı. O çalıştırmanın sonuçları ve bulguları §0'da
özetlenmiştir; düzeltilmiş commit'in Windows'ta yeniden çalıştırılması sizdedir. Bu belge
hazırlanan, yürütülen ve engellenen işleri ayrı ayrı kaydeder.

| Sembol | Anlamı |
|---|---|
| **YÜRÜTÜLDÜ (Linux)** | Bu bulut kapsayıcısında (Ubuntu 24.04, Linux) gerçekten çalıştırıldı; Windows değildir |
| **YÜRÜTÜLDÜ (Windows, bağımsız inceleme)** | Sizin incelemenizde gerçek Windows 11'de, `f2de5e6` ile çalıştırıldı (ham raporlar depoya eklenmedi) |
| **HAZIRLANDI** | Betik/talimat yazıldı ve mümkün olduğu kadar Linux'ta denendi; düzeltilmiş hâli Windows'ta henüz çalıştırılmadı |
| **ENGELLENDİ** | Bu oturumdan yapılamaz; sizin bilgisayarınızda ya da sizin kararınızla yapılmalı |

## 0. PR #15 bağımsız incelemesi (`f2de5e6`) ve düzeltmeler (2026-11-02)

### 0.1 Gerçek Windows çalıştırması — YÜRÜTÜLDÜ (Windows, bağımsız inceleme)

Ortam: Windows 11 Home build 26300, Python 3.12.10, ANSI kod sayfası cp1252, UTF-8 modu
kapalı. `PYTHONUTF8` ve `PYTHONIOENCODING` testlerde kaldırıldı. Yolunda boşluk ve Türkçe
karakter bulunan yeni bir sanal ortam kullanıldı; `Run-Session10.ps1` değiştirilmeden
çalıştırıldı (PowerShell ayrıştırıcısında 0 sözdizimi hatası).

* Testler: **990 geçti, 1 başarısız, 5 atlandı** (R kurulu değil), **6 canlı test seçim dışı**.
  Tek başarısız test `test_sqlite_directory_created_owner_only` idi: dört açık ACL girdisi
  (geçerli kullanıcı, SYSTEM, Administrators, OWNER RIGHTS) vardı, test ise tek girdi bekliyordu.
* Daha önceki iki kodlama hatası **tekrarlanmadı**; UI başlatma ve UTF-8 dosya gidiş-dönüşü geçti.
* Ön kontrol: API `POST /projects` için 201 döndü, ancak hemen yeniden başlatmadan sonra proje
  **yoktu** (aşağıda P1-a). ACL değerlendirmesi başarısız sayıldı (P2-a).
* Ağ: PubMed, OpenAlex ve Crossref 200; iki sağlayıcının anahtarsız model listesi uçları 401.
  Europe PMC ilk denemede **400**, sonraki bir istekte 200 döndü. Nedeni belirlenmedi; bu,
  doğrulanmış kalıcı bir bağlantı hatası değildir.
* Dış `powershell.exe -File` süreci, başarısız test ve ön kontrole rağmen **0** ile çıktı (P2-c).
* Çıkarım (inference), ücretli çağrı, gerçek derleme ya da bilimsel karar **yapılmadı**.

### 0.2 Bulgular, yeniden üretim ve düzeltmeler

Her bulgu düzeltmeden önce bu Linux ortamında yeniden üretildi. İncelemenin yeniden üretim
testleri `f2de5e6` üzerinde Linux'ta çalıştırıldı: 7 testin 7'si başarısız oldu.
`Everyone:(F)` testi Linux'ta POSIX koluna düştüğü için başka bir nedenle başarısız oldu; Windows
kolu Linux'ta zorlanarak ayrıca üretildi (`owner_only: True`). PowerShell çıkış kodu kusuru
kaynakta görülüyor (betik hiç `exit` çağırmıyordu); burada PowerShell çalıştırılamadığı için
yeniden üretimi Windows'taki gerçek çalıştırmaya dayanır.

| # | Bulgu | Düzeltme | Regresyon testleri |
|---|---|---|---|
| P1-a | API, commit tamamlanmadan başarılı yanıt veriyordu (`get_session`'ın `yield` sonrası commit'i, FastAPI'nin varsayılan *request* kapsamında yanıttan sonra çalışıyor) | Oturum bağımlılığı `Depends(get_session, scope="function")`: commit yanıttan **önce**. Commit hatası geri alınır ve HTTP 500 ile bildirilir ("hiçbir şey kaydedilmedi"). FastAPI alt sınırı 0.121 (`scope` ilk kez 0.121.0'da; 0.117.1, 0.118.0 ve 0.120.0'da yok). FastAPI 0.121.0 ile 75 API testi geçti | `test_session10_commit.py`: gerçek HTTP (uvicorn), geciken commit, başarısız commit, kimliği doğrulanmış yazma (`PUT /protocol`), onaylanmış yazmadan hemen sonra 3 kez zorla sonlandırma + yeniden başlatma. 4'ü eski kodda başarısız, yeni kodda geçiyor |
| P1-b | Başarısız ACL kurulumu, yeniden açılışta atlanıyordu | Windows'ta dizinler kodun kendisi tarafından oluşturulur; veritabanı dosyası yaratılmadan önce ACL uygulanır ve **SID ile doğrulanır**. Başarısızlıkta yalnızca bu çağrının oluşturduğu boş dizinler kaldırılır. Var olan boş dizin yeniden korunup doğrulanır; dolu ve ilgisiz bir dizin hiç değiştirilmez (sahip-yalnız değilse başlatma reddedilir); var olan veritabanı dosyası eskisi gibi açılır | `test_session10_review.py`: incelemenin iki testi (Windows kolu zorlanarak), doğrulanamayan ACL, kalan boş dizin, ilgisiz klasörün korunması, var olan veritabanı, POSIX davranışının değişmemesi |
| P2-a | Sahip-yalnız kararı girdi sayısına dayanıyordu (`Everyone:(F)` kabul ediliyordu) | `src/state/acl.py`: sahip ve DACL SDDL olarak okunur (`GetNamedSecurityInfoW`) ve SID ile değerlendirilir; yerelleştirilmiş adlar ve `icacls` metni kullanılmaz. Ön kontrol ve Windows testi aynı denetimi kullanır | Everyone, Users, Authenticated Users, Guests, başka hesap, bilinmeyen takma ad, nesne ACE'si, devralınmış girdiler, null DACL ve başka sahip+OWNER RIGHTS reddedilir. Windows'ta gözlenen dört girdili ACL kabul edilir |
| P2-b | İstatistik denetleyicisi yanlış manifesti kabul ediyordu (`digest in json.dumps(manifest)`) | `stats_output_sha256` ve `files["stats_output.json"]` alanlarının **ikisi de** bulunmalı, biçimce doğru olmalı ve dosyanın özetiyle aynı olmalı | Yanlış, çelişkili, eksik ya da bozuk bağlamalar reddedilir. Olumlu kontrol olarak gerçek bir Aşama 5 manifesti kullanılır (R yalnızca bu mantığı ayırmak için taklit edildi; sayısal iddia yok) |
| P2-c | PowerShell betiği başarısız doğrulamaya rağmen 0 ile çıkıyordu | Genel karar: test çalıştırması, JUnit raporu, kurulum ve ön kontrolün hepsi başarılıysa 0; doğrulama başarısızsa 1; betik tamamlanamadıysa 2. Eksik ya da bozuk JUnit raporu ve sıfır test başarısızlık sayılır | `test_session10_powershell.py`: gerçek `powershell.exe` ile 14 süreç düzeyi kontrolü. **Yalnızca Windows'ta çalışır**; burada atlandı |

**Ek araç düzeltmeleri:**

* **Karşılaştırılmayan çıktı:** `check_pilot_stats` bir karar verir: `validated` (0),
  `failed` (1), `error` (2) ya da `inconclusive` (3). Örneğin desteklenmeyen bir HR çıktısı
  `inconclusive` olur. Yalnızca 0, istatistiklerin yeniden üretildiği anlamına gelir.
* **Süreç ağacı:** Ön kontrol, başlattığı sürecin bütün ağacını durdurur (Windows'ta
  `taskkill /T /F` yalnızca kendi PID'sine; POSIX'te süreç grubu) ve portların boşaldığını
  denetler. İlgisiz bir Python süreci durdurulmaz (testle doğrulandı). Linux kontrolü: eski
  davranışta (yalnızca başlatıcıyı durdurmak) çocuk HTTP sunucusu dinlemeye devam etti.
* **Europe PMC:** Sorgu, uygulamanın kendi istemcisinin parametreleriyle (`resultType=lite`)
  belgelenmiş `EXT_ID:1 AND SRC:MED` aramasına çevrildi. Her deneme kaydedilir, beklenmeyen
  yanıtın başı saklanır. Önce hata sonra başarı `warn` (geçici) olarak raporlanır. 400'ün
  nedeni bilinmiyor; kalıcı bağlantı hatası olarak sunulmuyor.
* **Heterojenlik popülasyonu (§7.4'teki gözlem):** MH birincil modeli kendi kestirimi olmayan
  tabloları da içerdiğinde sonuç paragrafı şunu yazar: "Heterogeneity (computed with
  inverse-variance weights from the 2 studies with a study-level estimate, not from the
  population of the primary model): …". GRADE karar desteği de aynı bilgiyi verir. Olgular,
  şablonlar, istatistikler ve kanıt parmak izi değişmedi: `evidence.py`, `stats_engine` ve
  `reporting/state.py` `f2de5e6` ile aynı; şemaya yalnızca bir yöntem eklendi.

**Düzeltmelerden sonraki Linux sonuçları (YÜRÜTÜLDÜ, Linux; Windows sonucu değildir):**

* Tam paket, UTF-8, R kurulu: **1053 geçti, 0 başarısız, 14 atlandı** (yalnızca Windows'a özgü
  PowerShell kontrolleri), 6 canlı test seçim dışı. `ruff`, `ruff format` ve `mypy` (strict,
  130 dosya) temiz.
* Tam paket, cp1254 öykünmesi, boşluk ve Türkçe karakter içeren `TMPDIR`: **1048 geçti,
  0 başarısız, 19 atlandı** (14 PowerShell kontrolü; R boşluklu geçici dizini reddettiği için
  5 R testi).
* metafor referansı yeniden üretildi: **1.879 karşılaştırmanın tümü tolerans içinde**. Pilot
  denetleyicisinin canlı R testleri (sentetik çıktılar) geçti.
* İncelemenin özgün testleri bu ağaçta, Linux'ta: commit zamanlaması ve üç denetleyici testi
  **geçti**. İki ACL testi ve `Everyone` testi Linux'ta POSIX koluna düştüğü için hâlâ
  başarısız; bunların Windows'ta geçmesi beklenir, ancak doğrulanmadı. Aynı oracle'larla
  uyarlanmış kopyaları Windows kolunu zorlayarak geçiyor.
* En düşük FastAPI sürümü (0.121.0, Starlette 0.49.3) ile API, commit, kimlik doğrulama,
  amendment ve işlem testleri: **75 geçti, 1 atlandı** (o ortamda Streamlit yok).
* Python 3.11 ile Windows'ta, pytest'in `tmp_path` dizinleri korumalı ACL almayabilir
  (CPython bunu 3.12.4'ten beri uygular). Var olan dolu bir dizinde yeni veritabanı reddedildiği
  için bazı testler orada farklı davranabilir. Windows'ta yalnızca 3.12.10 çalıştırıldı.

### 0.3 Pilot protokolü — sizin seçiminiz

Pilot için **sizin mevcut Gamma Knife vestibüler schwannom protokolünüz** kullanılacak; §5.1'deki
alternatif soru önerisi geçersizdir. Protokolün içeriği (PICO, sonuçlar, uygunluk ölçütleri,
veritabanları ve arama dizeleri) ve gereken insan kararları, gerçek pilot başlamadan önce
belirlenmelidir. Bu oturumda belgenin içeriği okunmadı ve depoya hiçbir bölümü eklenmedi.

## 1. Ortam

| Öğe | Değer |
|---|---|
| Çalışma yeri | Claude Code bulut oturumu, Ubuntu 24.04.5 LTS, 4 CPU, Linux 6.18 |
| Python | 3.13.18 (`/tmp/venv`, `pip install -e ".[dev,stats,ui,manuscript]"`) |
| R | 4.3.3; metafor 4.4-0; lme4 1.1-35.1; jsonlite (Ubuntu paketleri, bu oturumda kuruldu) |
| Windows erişimi | **Yok.** `list_computer_folders` boş liste döndürdü. Aracın kendi açıklamasına göre uzaktan bağlanabilecek klasörler yalnızca Mac veya Linux bilgisayarlarda olabilir; bu yolla bir Windows bilgisayarı bağlanamaz |
| Ağ politikası | Vekil sunucu şu adreslere bağlantıyı reddetti (HTTP 403, 2026-11-01): `eutils.ncbi.nlm.nih.gov`, `www.ebi.ac.uk`, `api.openalex.org`, `api.crossref.org`, `api.unpaywall.org`, `api.openai.com`, `pmc.ncbi.nlm.nih.gov`. `api.anthropic.com` erişilebilir (anahtarsız istek 401 döndü). Ortam ayarlarındaki *Network access* bölümünden izin verilebilir: https://code.claude.com/docs/en/cloud-environments#network-access |
| API anahtarları | Bu kapsayıcıda Anthropic/OpenAI anahtarı yok; canlı model çağrısı yapılmadı |

### Doğrulanan başlangıç noktası (YÜRÜTÜLDÜ)

* `origin/main` = `f0c07d4ff6904aa93eaae9896f65534152c33ad7` (PR #14 birleştirme commit'i).
* `f0c07d4^{tree}` = `d1a8e84^{tree}` = `b6d963b7ab3cc44a3fb7251dc8855f0d686a5da5`: kaynak ağacı,
  bağımsız incelenen `d1a8e84` ile birebir aynıdır.
* CI çalışması 41330099574 (GitHub API ile kontrol edildi): `test (3.11)`, `test (3.12)`,
  `test (3.13)` ve `r-validation` işlerinin dördü de `f0c07d4` üzerinde `success`.
* Oturum dalı `claude/practical-darwin-abm38r`, `origin/main`'den (`f0c07d4`) yeniden
  oluşturuldu; dalda başka iş yoktu (uzak dal da `f0c07d4`'teydi).
* Bu raporun ait olduğu son commit'in kimliği PR açıklamasında ve oturum devrinde verilir
  (bir dosya kendi commit kimliğini içeremez).

## 2. Durum tablosu

| Kapsam | Durum | Ayrıntı |
|---|---|---|
| 1. Windows kurulum ve başlatma | `f2de5e6`: **YÜRÜTÜLDÜ (Windows, bağımsız inceleme)**; bulgular düzeltildi, düzeltilmiş commit Windows'ta **henüz çalıştırılmadı** | §0, §3, §4 |
| `0700` izin varsayımı | Düzeltildi (`f2de5e6`); Windows'ta kalan ACL sorunları §0 P1-b/P2-a ile düzeltildi; Windows'ta yeniden çalıştırılmalı | §0, §3.1 |
| İki metin okuma hatası (`PYTHONUTF8=1` ile geçen) | Düzeltildi; gerçek Windows'ta (cp1252) **tekrarlanmadı** | §0.1, §3.2 |
| API commit zamanlaması | **Düzeltildi** (P1-a); Linux'ta gerçek HTTP ile test edildi | §0.2 |
| 2. Gerçek PubMed araması ve içe aktarma | **ENGELLENDİ**: bulut ağ politikası; Windows'ta PubMed erişilebilir (200). Protokol seçildi, içeriği henüz belirlenmedi | §0.3, §5 |
| 3. Gerçek PDF'ler ve veri çıkarma | **ENGELLENDİ** (PDF yok, Windows yok) | §5 |
| 4. Canlı model/API davranışı | **ENGELLENDİ** (anahtar yok; OpenAI ağ dışı). Yalnızca anahtarsız erişilebilirlik ölçüldü | §6 |
| 5. İstatistik ve raporlama | Bağımsız denetleyici **HAZIRLANDI**, sentetik veride **YÜRÜTÜLDÜ (Linux)**; pilot verisi yok | §7 |
| 6. Makale, denetim, dışa aktarma | **ENGELLENDİ** (pilot yok); duyarlılık cümlesi ve heterojenlik etiketi **düzeltildi** | §0.2, §8 |
| İnsan kararları | **Hiçbiri verilmedi ya da taklit edilmedi** | §5.3 |

## 3. Windows bulguları (kaynakta kanıtlanan, Linux'ta yeniden üretilen)

*(Bu bölüm `f2de5e6` öncesi durumu anlatır; Windows sonuçları için bkz. §0.1.)*

Daha önceki bağımsız Windows çalıştırmanızın sonucu (980 geçti, 1 R testi atlandı, 7 test
seçim dışı) bu oturumda yeniden üretilemez, çünkü Windows'a erişim yok. Aşağıdaki iki
bulgu kaynak kod ve Linux'ta öykünme ile kanıtlandı. Düzeltmelerin Windows'ta doğrulanması
§4'teki betikle sizin tarafınızdan yapılmalıdır.

### 3.1 `0700` izin varsayımı — güvenlik etkisi olan platform kusuru

* **Neden:** `prepare_sqlite_path` veritabanı dizinini `mkdir(mode=0o700)` ile oluşturuyordu.
  Windows bu POSIX kipini dikkate almaz, dizin üst dizinin ACL'sini devralır. Depo `C:\` altında
  bir klasördeyse, Windows'un varsayılan ACL'si genellikle tüm *Authenticated Users* hesaplarına
  alt klasörlerde değiştirme izni verir. Bu durumda gözden geçiren kimlik bilgisi özetlerini ve
  denetim izini tutan veritabanı başka hesaplarca okunabilir. `test_sqlite_directory_created_owner_only`
  Windows'ta bu yüzden başarısızdı: kod yalnızca POSIX'te koruma sağlıyordu.
* **Düzeltme (`src/state/db.py`):** Windows'ta *yeni oluşturulan* veritabanı dizinine açık bir
  sahip-yalnız ACL uygulanır: `icacls <dizin> /inheritance:r /grant:r *<geçerli kullanıcı SID>:(OI)(CI)F`.
  SID `whoami /user` ile alınır. Her iki araç da `PATH` üzerinden değil, `%SystemRoot%\System32`
  altındaki mutlak yoluyla çağrılır. ACL uygulanamazsa başlatma güvenli biçimde reddedilir
  (*fail closed*). POSIX'teki gibi, var olan bir dizin değiştirilmez. Ön kontrol betiği var olan
  dizinin sahip-yalnız olup olmadığını salt okunur olarak raporlar.
* **Kimlik doğrulama ve erişim denetimi gevşetilmedi.** Test atlanmadı ya da hariç tutulmadı:
  POSIX'te kip, Windows'ta ACL denetlenir.
* **Testler:** `tests/unit/test_session10_fixes.py`. Komut oluşturma, boşluk ve Türkçe karakter
  içeren yol, hata durumunda reddetme ve yalnızca yeni dizinler kuralı tüm platformlarda sahte
  bir süreç çalıştırıcıyla test edilir. `tests/unit/test_reviewer_auth.py::test_sqlite_directory_created_owner_only`
  Windows'ta gerçek ACL'yi `icacls` çıktısıyla denetler: tek giriş, devralınmamış, tam denetim.
  **Bu Windows dalı bu oturumda çalıştırılamadı.**
* **Var olan dizininiz için (isteğe bağlı, sizin kararınız):** `data\` klasörü daha önceki
  oturumlardan kalmışsa otomatik olarak değiştirilmez. Ön kontrol `owner_only: false`
  bildirirse, PowerShell'de şu komutla kısıtlayabilirsiniz (yalnızca kendi hesabınız tam
  denetim alır; yönetici hesapları sahipliği alarak yine erişebilir):
  `icacls data /inheritance:r /grant:r "${env:USERDOMAIN}\${env:USERNAME}:(OI)(CI)F"`

### 3.2 Metin kodlaması — `PYTHONUTF8=1` ile geçen iki hata

* **Yeniden üretim (YÜRÜTÜLDÜ, Linux):** Türkçe Windows'un ANSI kod sayfasını öykünmek için
  `localedef -i tr_TR -f CP1254` ile bir yerel ayar oluşturuldu. Değiştirilmemiş `main` ağacı
  `LC_ALL=tr_TR.CP1254`, `PYTHONUTF8=0` ve boşluk/Türkçe karakter içeren `TMPDIR` ile çalıştırıldı.
  Python'un varsayılan kodlaması `CP1254` idi.
  Sonuç: **979 geçti, 2 başarısız, 1 atlandı, 6 seçim dışı**. Başarısız olanlar:
  * `tests/integration/test_pipeline_flow.py::test_full_pipeline`: `UnicodeDecodeError`
  * `tests/unit/test_stats_persistence.py::test_artifacts_written_with_manifest`: `÷` karakteri
    `Ã·` olarak okundu
  
  Bu tablo, Windows'ta `PYTHONUTF8=1` ile geçen iki hatayla ve sizin 980/1/7 sonucunuzla
  tutarlıdır (981 testten `0700` testi seçim dışı bırakılmış; iki metin hatası UTF-8 kipinde
  geçmiş). Bu bir çıkarımdır; Windows test adlarınızla karşılaştırılarak doğrulanmalıdır.
* **Neden:** Uygulama (`src/`) tüm dosyalarını UTF-8 olarak yazıyor ve okuyordu; bu doğrulandı.
  Kusur 17 test/doğrulama çağrısındaydı: `read_text()` ve `write_text()` kodlama
  belirtmeden çağrılıyordu (Windows'ta ANSI kod sayfası kullanılır). `validation/compare_metafor.py`
  de bunlara dahildir.
* **Düzeltme:** Her çağrıya `encoding="utf-8"` eklendi. `test_no_text_io_depends_on_the_locale_encoding`
  testi `src`, `tests` ve `validation` altındaki bütün Python dosyalarında kodlama belirtilmemiş
  metin G/Ç'yi AST ile arar. Bu denetimin kusuru gerçekten yakaladığı ayrı bir testle gösterildi.
* **Düzeltmeden sonra (YÜRÜTÜLDÜ, Linux, aynı cp1254 + Türkçe `TMPDIR` koşulları):**
  **991 geçti, 0 başarısız, 5 atlandı, 6 seçim dışı** (son ağaçla). Atlanan 5 testin hepsi R gerektirir. R
  kuruluydu, ancak boşluk içeren `TMPDIR` ile başlamayı reddetti (§7.3). Bu nedenle testler doğru
  biçimde atlandı.
* **Geçici çözüm gerekmiyor:** Bu daldan sonra `PYTHONUTF8=1` gerekmemelidir. §4 betiği önce
  varsayılan Windows davranışıyla çalışır; `-CompareUtf8` yalnızca karşılaştırma içindir.

### 3.3 Diğer platform gözlemleri

* `src/` içindeki tüm `open`/`read_text`/`write_text` çağrıları AST ile tarandı; hepsi kodlama belirtiyor ya da ikili kipte. `src/` içindeki tek alt süreç (`src/ui/__main__.py`, Streamlit'i başlatır) metin okumuyor.
* Bu oturumda eklenen betikler, dar bir konsol kod sayfasında çıktı yazarken hata vermemek için
  `stdout`'u `errors="backslashreplace"` ile yeniden yapılandırır.
* `SYSREV_DATA_DIR` (PDF'ler ve çıktılar) hiçbir platformda sahip-yalnız oluşturulmaz. Bu, önceki
  sürümlerden kalan bir sınırlamadır; kapsam genişletilmedi (§9).

## 4. Windows'ta çalıştırmanız gerekenler (HAZIRLANDI)

Ayrıntılı İngilizce talimat: [`validation/session10/README.md`](validation/session10/README.md).

```powershell
cd "C:\...\SysRev-Meta-Analysis"
git fetch origin
git checkout claude/practical-darwin-abm38r
git pull
powershell -ExecutionPolicy Bypass -File validation\session10\Run-Session10.ps1 -Network -CompareUtf8
```

`Run-Session10.ps1` şunları yapar:

1. Commit'i, çalışma ağacının temizliğini, Windows/Python sürümünü ve kod sayfalarını kaydeder.
2. `%USERPROFILE%\SysRev Doğrulama ş10 İ\venv` altında **yeni** bir sanal ortam kurar
   (yolda boşluk ve Türkçe karakter var).
3. Tüm testleri **varsayılan Windows davranışıyla** çalıştırır: `PYTHONUTF8` kaldırılır,
   `TEMP`/`TMP` Türkçe karakterli bir klasöre yönlendirilir. Sonuç JUnit XML olarak kaydedilir.
4. Yalnızca `-CompareUtf8` verilirse aynı testleri `PYTHONUTF8=1` ile ikinci kez çalıştırır;
   bu çalıştırma ayrı raporlanır.
5. Ön kontrolü çalıştırır: veritabanı + ACL, API ve UI başlatma, yeniden başlatma, UTF-8 dışa
   aktarma ve `-Network` ile anahtarsız servis erişilebilirliği.

Reviewer kaydı yapmaz, anahtar ve token okumaz, `data\` klasörünüze dokunmaz.
`session10_results\windows_<UTC>\` klasörünü bana gönderin; paylaşmadan önce içeriğini kontrol edin.

**Bu betiğin durumu:** Yalnızca ASCII karakterlerle yazıldı; Windows PowerShell 5.1 BOM'suz
dosyayı ANSI olarak okuduğu için Türkçe adlar karakter kodlarından oluşturulur. `f2de5e6`
sürümü bağımsız incelemede Windows'ta değiştirilmeden çalıştırıldı (§0.1). Çıkış kodu kusuru
(P2-c) düzeltildi: betik artık `summary.json` içine genel bir karar (`verdict`) ve nedenlerini
yazar ve buna göre çıkar (0 başarılı, 1 doğrulama başarısız, 2 betik tamamlanamadı).
`-VerdictOnly <summary.json>` yalnızca kararı hesaplar. **Düzeltilmiş betik bu ortamda
çalıştırılamadı:** PowerShell yok, indirilen bir PowerShell ikili dosyasını çalıştırma izni de
reddedildi. Süreç düzeyi kontrolleri (`tests/unit/test_session10_powershell.py`) Windows'ta
tam test paketinin içinde çalışır.

### 4.1 Ön kontrolün Linux sonucu (YÜRÜTÜLDÜ, Linux — Windows sonucu değildir)

Karalama klasörü `/tmp/SysRev Doğrulama ş10 İ/run 20261101T175416Z`:

| Denetim | Sonuç |
|---|---|
| Veritabanı oluşturma, koruma, yeniden açma | geçti (`0o700`, yeniden açınca proje ve aşama korunmuş) |
| API: başlatma, `/health`, HTTP ile proje oluşturma, sunucu yeniden başlatma | geçti (Türkçe başlıklı proje yeniden başlatmadan sonra mevcut) |
| UI: Streamlit başlatma ve `/_stcore/health` | geçti (`ok`). Tarayıcıda gerçek kullanım denetlenmedi |
| UTF-8 dışa aktarma (`dışa aktarım/özet ğüşİ.md`) | geçti |
| Ağ | PubMed, Europe PMC, OpenAlex, Crossref ve OpenAI: vekil sunucu 403. Anthropic: 401 (erişilebilir). Unpaywall: e-posta tanımlı değil, atlandı |

Düzeltmelerden sonra yeniden çalıştırma (YÜRÜTÜLDÜ, Linux, ağ denetimleri olmadan): veritabanı
`{"owner_only": true, "problems": []}`; API iki kez **zorla** sonlandırıldı ve onaylanan Türkçe
başlıklı proje yeniden başlatmadan sonra mevcuttu; her iki durdurmadan sonra port boşaldı. UI
sağlık uç noktası `ok` döndü, durdurmadan sonra port boşaldı ve Streamlit süreci kalmadı.
Sonuç: 0 başarısız denetim.

## 5. Pilot: girdiler, plan ve insan kapıları

### 5.1 Girdiler — sizden gerekenler

*(Güncel: pilot için sizin Gamma Knife vestibüler schwannom protokolünüz seçildi, bkz. §0.3.
Aşağıdaki alternatif soru önerisi geçersizdir ve yalnızca kayıt olarak bırakıldı.)*

Depoda klinik soru, onaylı protokol ya da seçilmiş literatür dosyası yok. Google Drive'ınızda
yalnızca **dosya adları** arandı; içerik açılmadı. Bir protokol belgesine benzeyen bir dosya
bulundu. Adını bu depoya yazmadım; oturumda size ilettim. Bunun bu pilot için *onaylı
sistematik derleme protokolü* olup olmadığını siz söylemelisiniz. Bulunan diğer klasörler
görüntü ve doz verisi içeriyordu; literatür değildi ve kullanılmadı.

Sizin onaylı sorunuz yoksa aşağıdaki **öneri**yi kullanabilirsiniz. Bu yalnızca bir öneridir ve
hiçbir karar yerine geçmez:

* **Soru (öneri):** İşitmesi korunmuş (AAO-HNS A/B ya da Gardner-Robertson I-II) tek taraflı
  sporadik vestibüler schwannomlu erişkinlerde, birincil Gamma Knife radyocerrahisinden sonra
  son izlemde işitmenin korunma oranı nedir?
* **Ölçüt:** Tek grup oranı (`PROPORTION`, son izlem), birincil model binom-normal GLMM. Bu
  model metafor ile doğrulanmıştır. Duyarlılık modeli logit ters varyanstır; düzeltilen
  duyarlılık cümlesi de böylece pilotta sınanır.
* **Veritabanı:** Yalnızca PubMed (gerçek ve tekrarlanabilir bir arama). Pilot olduğu için
  kapsam bilerek dar tutulur; tam bir derleme değildir ve böyle raporlanmalıdır.
* **Arama (öneri):**
  `("Neuroma, Acoustic"[MeSH Terms] OR "vestibular schwannoma"[Title/Abstract]) AND ("Radiosurgery"[MeSH Terms] OR "gamma knife"[Title/Abstract]) AND ("hearing"[Title/Abstract]) AND ("low dose"[Title/Abstract] OR "12 Gy"[Title/Abstract] OR "marginal dose"[Title/Abstract]) AND ("long-term"[Title/Abstract] OR "10-year"[Title/Abstract] OR "10 years"[Title/Abstract])`,
  yayın tarihi 2015–2025.
  * **Planlama ölçümü (uygulama dışı):** 2026-11-01'de Claude'un PubMed bağlayıcısıyla bu sorgu
    **27 kayıt** döndürdü. Sorgu çevirisi PubMed'in kendi çevirisiydi. Bu ölçüm uygulamanın kendi
    PubMed istemcisiyle yapılmadı; pilotta geçerli olan, uygulamanın kaydedeceği sorgu, tarih,
    sayılar ve yanıt SHA-256 değerleridir.
* **Uygunluk (öneri):** başlangıçta işitmesi korunmuş ≥ 20 hasta; işitme sonucunun sayı
  olarak verilmesi; NF2, önceden tedavi, olgu sunumu ve derleme dışlanır.
* **PDF'ler:** 3–5 dahil çalışma (açık erişim + sizin sağladıklarınız). En az birinde zor bir
  örnek olmalı: taranmış sayfa, ondalık virgül ya da karmaşık tablo.

**Sizden gerekenler:** (a) soru ve protokolün onayı ya da kendi protokolünüz;
(b) `.env` içinde `SYSREV_NCBI_EMAIL` ve `SYSREV_UNPAYWALL_EMAIL` için kendi adresiniz;
(c) ücretli çağrılar için onayınız ve uygulamada zaten seçili modeller. Model seçimlerinizi
göremiyorum; hiçbir model sessizce başka bir modelle değiştirilmez. (d) Erişilemeyen PDF'ler.

### 5.2 Çağrılar ve maliyet kaydı

* Hangi modellerin seçili olduğunu ve fiyatlarını göremediğim için **maliyet tahmini vermiyorum**.
  Tahmini maliyet *Usage & costs* sayfasında liste fiyatlarından hesaplanır. Ölçülen maliyet
  sağlayıcı konsolundaki faturadır. Rapor ikisini ayrı tutacaktır.
* Küçük ve amaçlı çağrılar: 27 kaydın iki tarama geçişi, 3–5 PDF'nin veri çıkarma önerileri,
  RoB taslakları; isteğe bağlı olarak tek bir makale bölümünün model ile yazımı.
* Her çağrının sağlayıcısı, model kimliği, token sayıları ve katalog özeti
  `collect_pilot_evidence` (`ai_usage.json`) ile kaydedilir.

### 5.3 İnsan kapıları — sizin vereceğiniz kararlar

Aşağıdaki kararların hiçbiri bu oturumda verilmedi, taklit edilmedi ya da test yardımcılarıyla
geçilmedi. Her biri kimliği doğrulanmış reviewer olarak sizin işinizdir:

1. Protokol onayı ve Gate 1 (sorgu sözdizimi PubMed tarafından doğrulanır).
2. PubMed araması; sayı uyumsuzluğu çıkarsa onaylanması; tekilleştirmedeki belirsiz çiftler.
3. Başlık/özet taramasında iki model geçişinin çatışmalarının hakemliği ve Gate 1.5.
4. Tam metin: PDF yükleme, kimlik onayı (karşılaştırılan PDF'nin SHA-256'sı), "elde edilemez"
   kararları ve Gate 2.
5. Veri çıkarma: her değerin sayfa, konum ve birebir alıntısıyla kabul, ret ya da düzeltmesi.
   Gerekirse sayı biçimi (ör. `decimal_comma`) ve kaynak notu. Ardından kolların yeniden
   oluşturulması. Doğrulanamayan içerik işaretli kalır.
6. Yanlılık riski onayı ve Gate 3.
7. Analiz planı onayı ve Aşama 5.
8. GRADE alanları, PRISMA 2020 ve özet kontrol listeleri ve SWiM uygulanabilirliği.
9. Makale: yazar girdileri, denetim bulgularının onaylanması, insan imzası ve tamamlama.

Pilotun ortasında API ve UI en az bir kez durdurulup yeniden başlatılmalıdır (yeniden
başlatma/sürdürme).

## 6. Canlı servis kontrolleri (gerçekte yapılanlar)

| Kontrol | Ne yapıldı | Sonuç | Uygulamanın kendi istemcisi mi? |
|---|---|---|---|
| PubMed | Claude PubMed bağlayıcısıyla tek sayım sorgusu | 27 kayıt (2026-11-01) | **Hayır** (yalnızca planlama) |
| NCBI/EBI/OpenAlex/Crossref/Unpaywall | `curl` ve ön kontrol | Vekil sunucu 403 | Denenemedi |
| Anthropic `/v1/models` | Anahtarsız GET | 401 (erişilebilir; çağrı faturalanmadı) | Uygulamanın adaptörü değil |
| OpenAI `/v1/models` | Anahtarsız GET | Vekil sunucu 403 | Denenemedi |
| Model çıkarımı (inference) | — | **Yapılmadı** | — |
| Windows ön kontrolü (bağımsız inceleme, `f2de5e6`) | Uygulama dışı `httpx` GET'leri | PubMed, OpenAlex, Crossref 200; Europe PMC önce 400, sonra 200; Anthropic/OpenAI anahtarsız 401 | Erişilebilirlik; uygulamanın istemcisi değil |

Mock'lu testler canlı uyumluluğu kanıtlamaz. Sağlayıcı uyumluluğu, kullanım kaydı ve hata
davranışı **doğrulanmamıştır**.

## 7. İstatistik

### 7.1 Mevcut metafor doğrulaması (YÜRÜTÜLDÜ, Linux)

`Rscript validation/r/metafor_reference.R …` ve `python -m validation.compare_metafor`:
**1.879 karşılaştırmanın tümü tolerans içinde** (R 4.3.3, metafor 4.4-0, lme4 1.1-35.1).
En büyük farklar: kapalı form 4.8e-13; GLMM 1.3e-5; REML 2.9e-10.

Son ağaçla tam test paketi (YÜRÜTÜLDÜ, Linux, UTF-8 yerel ayarı, R kurulu): **996 geçti,
0 başarısız, 0 atlandı, 6 seçim dışı** (`live` işaretli testler, varsayılan olarak seçim dışı).
`ruff check`, `ruff format --check` ve `mypy` (strict) temiz.

### 7.2 Pilot için bağımsız denetleyici (HAZIRLANDI; sentetik veride YÜRÜTÜLDÜ, Linux)

`validation/session10/check_pilot_stats.py` ve `pilot_reference.R`, pilotun kaydedilmiş
`stats_output.json` dosyasından şunları yapar:

* Dosyanın SHA-256'sının manifestte olduğunu doğrular.
* Raporlanan her modeli (ortak etki, rastgele etkiler, duyarlılık) R/metafor ile yeniden
  hesaplar: `rma`, `rma.mh`, `rma.glmm`.
* Kestirimleri, güven aralıklarını (analiz ve doğal ölçek), τ²'yi, k'yi ve modelin
  kullandığı çalışma kimliklerini karşılaştırır.
* Her modelin popülasyonunu (katılımcı ve olay toplamları) çalışma verisinden yeniden toplar.
* Modele özgü ağırlıkları, Q, p ve I²'yi karşılaştırır.

Tanınmayan bir model "karşılaştırılmadı" olarak listelenir, asla "geçti" sayılmaz.
İnceleme üzerine (§0.2) dosyanın özeti iki manifest alanının ikisiyle de birebir eşleşmelidir.
Sonuç bir karar olarak verilir: `validated`, `failed`, `error` ya da `inconclusive`. Yalnızca
`validated` (çıkış 0), istatistiklerin yeniden üretildiği anlamına gelir.

Sentetik sınama (`tests/unit/test_session10_tooling.py`): RR ve OR (birincil model ortak ve
rastgele; çift tam olaylı tablo dahil, MH k = 3) ile oranlar (GLMM ve logit-IV):
**248 karşılaştırma, 0 hata, 0 karşılaştırılmamış**. Kasıtlı olarak değiştirilmiş bir güven
sınırı ve bozulmuş bir popülasyon yakalandı. Gerçek bir Aşama 5 klasörü (korumalı iş akışı)
de olduğu gibi kabul edildi. Bu testler CI'nın `r-validation` işine eklendi; orada R
bulunmazsa atlanmaz, başarısız olur.

**Pilot verisi olmadığı için pilot istatistikleri denetlenmedi.**

### 7.3 R ve boşluklu geçici dizin

R, oturum geçici dizininin yolunda boşluk varsa başlamıyor:
`Fatal error: 'R_TempDir' contains space` (Linux'ta gözlendi). Denetleyici POSIX'te bu
durumda R'a `/tmp` verir. Windows'taki davranış denetlenmedi; talimat README'de.

### 7.4 Heterojenlik popülasyonu (inceleme üzerine düzeltildi, §0.2)

*Aşağıdaki gözlem PR #15 incelemesinde düzeltilmesi istenen bir etiketleme sorunu olarak
kabul edildi ve düzeltildi; sayılar değişmedi.*


Üç denemeli sentetik örnekte birincil model Mantel-Haenszel olduğunda sonuç paragrafı
"3 studies … including 600 participants … Heterogeneity: Q = 0.68 (df = 1 …); I² = 0%"
diyor. Heterojenlik istatistikleri ters varyans popülasyonuna (k = 2) aittir. df = 1 bunu
dolaylı olarak gösterse de, cümle bunu açıkça etiketlemiyor. Sayılar doğrudur; sorun yalnızca
etiketlemedir. Bu bir pilot bulgusu değildir; kapsamı genişletmemek için düzeltilmedi.
Düzeltilip düzeltilmeyeceği sizin kararınızdır.

## 8. Makale: duyarlılık analizi cümlesi (düzeltildi)

* **Kusur (YÜRÜTÜLDÜ, Linux, yeniden üretildi):** `src/manuscript/drafting.py`,
  `stats.<analiz>.sensitivity.` ile başlayan her olguyu seçiyordu. Session 9 incelemesinden
  beri bunlar duyarlılık modelinin popülasyon sayılarını da içerir. Birincil model MH olan
  sentetik örnekte taslak şöyleydi:
  `Sensitivity analysis: 2; 400; 70; 1.35 (95% CI 0.88 to 2.06); common-effect model (inverse variance).`
* **Düzeltme:** Kestirimler rollerine göre (`sensitivity_estimate`) seçilir. Duyarlılık
  modelinin kendi popülasyonu varsa, sayıları birimleriyle kestirimin yanına yazılır:
  `… common-effect model (inverse variance) (population of this model: 2 studies, 400 participants, 70 events).`
  Popülasyon birincil modelle aynıysa cümle eskisiyle aynıdır.
* **Korunanlar:** Hesaplamalar, olgu metinleri ve olgu özetleri değişmedi. Saklanmış makale
  sürümleri yeniden yazılmaz. Önceki imzalar saklanmış denetim raporuna bağlıdır ve geçerli
  kalır.
* **Yükseltme etkisi (testle doğrulandı):** Eski cümleyi içeren saklanmış bir sürüm yeniden
  denetlenirse değerleri hâlâ doğrulanır. Ancak cümle artık şablon metni sayılmaz; Denetçi
  onu engellemeyen tek bir `REQUIRES_HUMAN_REVIEW` bulgusuyla insana sorar. Sürümün özeti
  değişmez.
* **Testler:** `test_sensitivity_sentence_selects_estimates_and_labels_counts` (her iki birincil
  model; denetim bulgusu yok), `test_sensitivity_clauses_ignore_non_estimate_facts_with_the_prefix`,
  `test_a_stored_draft_with_the_old_sentence_is_routed_to_human_review`.

## 9. Kalan sınırlamalar ve tamamlanmamış işler

**Session 10'u tamamlamak için hâlâ gerekenler (hepsi sizin Windows bilgisayarınızda):**

1. Düzeltilmiş commit'in `Run-Session10.ps1` ile Windows'ta yeniden çalıştırılması:
   ACL testi, PowerShell süreç kontrolleri, API yeniden başlatma, UI süreç ağacı ve Europe PMC.
   Bu düzeltmelerin Windows kolları bu ortamda çalıştırılamadı. Bu, ctypes ile SDDL okumayı
   (`GetNamedSecurityInfoW`) da kapsar.
2. Pilot protokolünüzün içeriğinin ve gereken insan kararlarının belirlenmesi (§0.3).
3. Tarayıcıda normal kullanım: yükleme, indirme ve dışa aktarma.
4. Pilotun tamamı: gerçek arama, PDF'ler, canlı çağrılar, insan kararları ve makale.
5. `collect_pilot_evidence` ve `check_pilot_stats` çıktıları; PRISMA, GRADE/SoF,
   kontrol listeleri ve makalenin pilot kanıtına göre denetlenmesi.

**Önceden belgelenmiş ve değişmeyen sınırlamalar:**

* JATS/OCR yok; yalnızca görüntüden oluşan sayfalar işaretlenir.
* HR/RD/IRR desteklenmiyor.
* Abonelikli veritabanı API'leri yok.
* Gerçek dışa aktarma dosyalarıyla ayrıştırıcı denemesi yapılmadı.
* SQLite dışında kilitleme garantisi yok.
* `SYSREV_DATA_DIR` sahip-yalnız değil.
* Windows'ta önceden var olan bir veritabanı dosyası, dizini doğrulanmadan eskisi gibi açılır
  (yükseltme uyumluluğu); ön kontrol bu dizinin durumunu raporlar.
* `src.ui` başlatıcısı Streamlit'i bir çocuk süreç olarak başlatır. Başlatıcı zorla
  sonlandırılırsa çocuk süreç kalabilir. Ön kontrol kendi ağacını temizler; uygulamanın
  başlatıcısı değiştirilmedi.
* Çok reviewer'lı imza yok.

**Hariç tutmalar ve geçici çözümler:** Bu dalda hiçbir test hariç tutulmadı ya da atlanmadı;
R testleri yalnızca R yoksa atlanır. `PYTHONUTF8=1` gerekmemelidir. Windows'ta ihtiyaç duyulursa
ayrıca raporlanacaktır.

## 10. Yeniden üretme (Linux kanıtları)

```bash
pip install -e ".[dev,stats,ui,manuscript]"
pytest                                                          # tam test paketi
localedef -i tr_TR -f CP1254 /tmp/locales/tr_TR.CP1254          # Türkçe Windows kod sayfası öykünmesi
LOCPATH=/tmp/locales LC_ALL=tr_TR.CP1254 PYTHONUTF8=0 TMPDIR="/tmp/tmp dizin ğüşİ" pytest
Rscript validation/r/metafor_reference.R validation/r/synthetic_cases.json /tmp/ref.json
python -m validation.compare_metafor /tmp/ref.json
pytest tests/unit/test_session10_fixes.py tests/unit/test_session10_tooling.py
python -m validation.session10.preflight --out session10_results --network
```
