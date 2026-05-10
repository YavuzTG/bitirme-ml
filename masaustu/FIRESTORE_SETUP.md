# Firestore Realtime Database Setup

Bu dokumanda Firestore Realtime Database'i kurma ve güvenlik kurallarını ayarlama işlemleri açıklanmıştır.

## 🚀 Adım 1: Firestore Realtime Database Oluştur

1. [Firebase Console](https://console.firebase.google.com) aç
2. Proje: `beed-sinyal-siniflandirma` seç
3. Sol menüden **"Realtime Database"** tıkla
4. **"Veritabanı Oluştur"** tuşuna tıkla yapın
5. **Konum**: `europe-west1` (veya yakın bir bölge)
6. **Güvenlik Kuralları**: **"Test modunda başlat"** seç
7. **"Oluştur"** tıkla

Firestore başarıyla oluşturulduktan sonra, veritabanı URL'i bu formatta olacaktır:
```
https://beed-sinyal-siniflandirma.firebaseio.com
```

Bu URL zaten `firebase_config.py`'de `databaseURL` olarak yapılandırılmıştır! ✅

---

## 🔐 Adım 2: Güvenlik Kurallarını Ayarla

**ÖNEMLİ:** Başında "Test modunda" demiştik. Bu üretime göre kötü. Şimdi güvenlik kurallarını düzenleyelim.

Firebase Console'da:
1. **Realtime Database** → **"Kurallar"** sekmesi
2. Aşağıdaki kodu yapıştır:

```json
{
  "rules": {
    ".read": "auth != null",
    ".write": "auth != null",
    
    "users": {
      "$uid": {
        ".read": "$uid === auth.uid || root.child('users').child($uid).child('isAdmin').val() === true",
        ".write": "$uid === auth.uid"
      }
    },
    
    "models": {
      ".read": "auth != null",
      "$modelId": {
        ".write": "root.child('users').child(auth.uid).child('isAdmin').val() === true"
      }
    },
    
    "predictions": {
      ".read": "auth != null",
      "$predId": {
        ".read": "data.child('userId').val() === auth.uid || root.child('users').child(auth.uid).child('isAdmin').val() === true",
        ".write": "auth != null && !data.exists() || data.child('userId').val() === auth.uid"
      }
    }
  }
}
```

3. **"Yayınla"** tıkla

---

## 📊 Veri Yapısı

Firestore'da şu yapıda veriler kaydedilecektir:

### Users Collection
```
users/
  yavuzturker_icloud_com/
    ├── email: "yavuzturker@icloud.com"
    ├── isAdmin: true
    ├── createdAt: "2026-03-30T10:30:00"
    └── lastLogin: "2026-03-30T10:31:00"
```

### Models Collection
```
models/
  -Oa5x1a2b3c4d5e6/
    ├── name: "Model_20260330_103000"
    ├── trainedBy: "yavuzturker@icloud.com"
    ├── accuracy: {
    │   ├── cnn: 0.95
    │   ├── svm: 0.92
    │   └── lstm: 0.94
    │ }
    ├── createdAt: "2026-03-30T10:30:00"
    └── status: "ready"
```

### Predictions Collection
```
predictions/
  -PQr7s8t9u0v1w2x/
    ├── userId: "yavuzturker@icloud.com"
    ├── modelId: "latest_model"
    ├── input: [1.2, 3.4, 2.1, ...]
    ├── result: {
    │   ├── cnn: 2
    │   ├── svm: 1
    │   └── lstm: 2
    │ }
    └── timestamp: "2026-03-30T10:35:00"
```

---

## 🔍 Admin Paneli Örnekleri

Eğer admin paneli yapacaksan, şu sorgular kullanabilirsin:

### Tüm Modelleri Listele
```python
from firestore_handler import FirestoreHandler

all_models = FirestoreHandler.get_all_models()
for model_id, model_data in all_models.items():
    print(f"{model_data['name']} — {model_data['trainedBy']}")
```

### Kullanıcının Tahmin Geçmişini Görüntüle
```python
predictions = FirestoreHandler.get_user_predictions("yavuzturker@icloud.com", limit=10)
for pred_id, pred_data in predictions.items():
    print(f"Tahmin: {pred_data['result']} — {pred_data['timestamp']}")
```

### Kullanıcının İstatistikleri
```python
stats = FirestoreHandler.get_prediction_stats("yavuzturker@icloud.com")
print(f"Toplam Tahmin: {stats['total']}, Bugün: {stats['today']}")
```

---

## ✅ Kontrol Adımları

1. ✅ Firestore Realtime Database oluşturuldu
2. ✅ Güvenlik kuralları ayarlandı
3. ✅ Veri yapısı hazırlandı
4. ✅ Firestore handler integre edildi
5. ✅ App.py'ye Firestore entegrasyonu eklendi

Şimdi **uygulamayı çalıştırabilirsin!** 🚀

```bash
python app.py
```

---

## 🆘 Sorun Giderme

### `Database connection error`
**Çözüm**: Firebase Console > Realtime Database > URL'in doğru olup olmadığını kontrol et

### `Permission denied` hatası
**Çözüm**: Güvenlik kurallarını kontrol et. Kurallar doğru şekilde yapıştırıldı mı?

### Veri kaydetilmiyor
**Çözüm**: 
- Firebase Console > Authentication > Email/Password etkinleştirildi mi?
- Güvenlik kurallarında `.write` permission var mı?

---

## 📝 Notlar

- Üretim ortamında, bu kuralları daha katı yapmalısın
- Prediksiyon verilerini periyodik olarak backup al
- Model versiyonlama için `createdAt` timestamp'ini kullan
