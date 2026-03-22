# GitHub Actions ile Uzak Eğitim

Bu projede `Model Eğitimi` butonu iki modda çalışır:

- Yerel mod: `GH_OWNER`, `GH_REPO`, `GH_TOKEN` tanımlı değilse mevcut yerel eğitim çalışır.
- Uzak mod: Bu değişkenler tanımlıysa eğitim GitHub Actions üzerinde çalışır ve model dosyaları otomatik indirilir.

## 1) GitHub tarafı

1. Repo içinde workflow dosyası olmalı: `.github/workflows/train.yml`
2. Repo `Settings > Actions > General` altında Actions aktif olmalı.
3. `BEED_Data.csv` gibi eğitim verisi repoda bulunmalı.

## 2) Uygulama çalıştırmadan önce ortam değişkenleri

PowerShell örneği:

```powershell
$env:GH_OWNER = "YavuzTG"
$env:GH_REPO = "bitirme-ml"
$env:GH_TOKEN = "ghp_xxx..."
$env:GH_WORKFLOW_FILE = "train.yml"   # opsiyonel
$env:GH_BRANCH = "main"               # opsiyonel
```

> `GH_TOKEN` için klasik PAT kullanıyorsan en az `repo` ve `workflow` izinleri ver.

## 3) Uygulama akışı

1. Uygulamayı başlat.
2. `Model Eğitimi` sekmesinde istersen CSV seç (uzak modda sadece dosya adı kullanılır).
3. `Modelleri Eğit` butonuna bas.
4. Log ekranında sırasıyla:
   - uzak tetikleme,
   - run bekleme,
   - run durumu,
   - artifact indirme
   mesajları görünür.
5. Bittiğinde şu dosyalar proje klasörüne iner/güncellenir:
   - `model_cnn.keras`
   - `model_lstm.keras`
   - `trained_models.pkl`
   - `metrics.json`

## 4) Sorun giderme

- `requests paketi eksik` hatası alırsan:

```powershell
pip install requests
```

- Workflow tetiklenmiyorsa:
  - token izinlerini,
  - repo adını,
  - workflow dosya adını (`train.yml`) kontrol et.

- Eğitim başarısızsa logdaki GitHub Actions linkini açıp failed step'i incele.
