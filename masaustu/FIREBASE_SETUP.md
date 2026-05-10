# Firebase Authentication Kurulumu

Bu uygulama Firebase ile kullanıcı giriş sistemi kullanmaktadır.

## 🎯 Adım 1: Firebase Project Oluştur

1. [Firebase Console](https://console.firebase.google.com) sayfasına git
2. Giriş yap (Google hesabı ile)
3. **"Proje oluştur"** butonuna tıkla
4. Proje adını gir (örn: `BEED-Sinyal-Siniflandirma`)
5. **"Firebase'i etkinleştir"** seçeneğini seç ve projeyi oluştur

## 🔑 Adım 2: Web App Ekle

Firebase Console'da:
1. Sol menüden **"Proje Ayarları"** (Dişli ikonu) tıkla
2. **"Genel"** sekmesine gel
3. **"Uygulamalarım"** bölümüne git veya sol menüde bulunabilir
4. **"Web'i seç"** veya **"Uygulaması ekle"** > **"Web"** seç
5. Uygulama adı gir (örn: `BEED`)
6. **"Uygulamayı Kayıt Et"** tuşuna tıkla

## 📋 Adım 3: Firebase Credentials Al

Kayıt ettikten sonra, aşağıdakine benzer bir kod göreceksin:

```javascript
// For Firebase JS SDK v7.20.0 and later, measure performance of real user sessions, 
// by passing `reportWebVitals` function as fourth argument to `initializeApp`
const firebaseConfig = {
  apiKey: "AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxx",
  authDomain: "beed-xxxx.firebaseapp.com",
  projectId: "beed-xxxx",
  storageBucket: "beed-xxxx.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:xxxxxxxxxxxx"
};
```

Bu bilgileri kopyala.

## 🔧 Adım 4: firebase_config.py Dosyasını Güncelle

`firebase_config.py` dosyasını aç ve aşağıdaki bilgileri değiştir:

```python
config = {
    "apiKey": "AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxx",  # ^ buradan
    "authDomain": "beed-xxxx.firebaseapp.com",   # ^ buradan
    "databaseURL": "https://beed-xxxx.firebaseio.com",
    "projectId": "beed-xxxx",  # ^ buradan
    "storageBucket": "beed-xxxx.appspot.com",    # ^ buradan
    "messagingSenderId": "123456789",             # ^ buradan
    "appId": "1:123456789:web:xxxxxxxxxxxx"       # ^ buradan
}
```

## 🔐 Adım 5: Firebase Authentication'ı Etkinleştir

Firebase Console'da:
1. Sol menüden **"Authentication"** seç
2. **"Sign-in method"** sekmesine tıkla
3. **"Email/Password"** sağlayıcısını etkinleştir
4. **"Enable"** tuşuna tıkla ve **"Kaydet"**

## 👨‍💼 Adım 6: Admin Email Ekle

`auth_widgets.py` dosyasını aç ve admin email'lerini ayarla:

```python
# auth_widgets.py içinde
ADMIN_EMAILS = [
    "admin@beed.com",           # İlk admin email
    "senin_email@gmail.com"     # Kendi email'ni ekle
]
```

## 📦 Adım 7: Gerekli Kütüphaneleri Yükle

```bash
pip install pyrebase4
```

## ✅ Adım 8: Test Et

1. Terminalde uygulamayı başlat:
```bash
python app.py
```

2. Giriş ekranında **"Kayıt Ol"** sekmesine git
3. Admin email'i ile bir hesap oluştur:
   - Email: `senin_email@gmail.com`
   - Şifre: `Secure123` (minimum 6 karakter)

4. Giriş yap ✅

---

## 🆘 Sorun Giderme

### `ModuleNotFoundError: No module named 'pyrebase4'`
Çözüm: `pip install pyrebase4` komutu çalıştır

### `Firebase bağlantısı kurulamadı`
Çözüm: `firebase_config.py` dosyasındaki bilgileri kontrol et

### `Email not found` hatası
Çözüm: Firebase Console > Authentication > Users sekmesinden email'i kontrol et

---

## 📝 User Rolleri

- **Admin**: Model eğitebilir + Tahmin yapabilir
- **Regular User**: Sadece tahmin yapabilir

---

İtici başarısızlığında, Türkçe yazılı sorunlarını gözden geçir! 🚀
