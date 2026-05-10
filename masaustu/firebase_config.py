# Firebase Konfigürasyonu
# Not: Firebase Console'dan Api Key ve diğer bilgileri alıp aşağıya yapıştır
# https://console.firebase.google.com/

import os
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

# ======= Firebase Credentials =======
# Firebase Console > Project Settings > Web API Key vs
# Not: Bu bilgileri gizli tutmaya çalış, production'da .env dosyası kullan

config = {
    "apiKey": "AIzaSyA3Rfl2VNpafpcMl4fSUXWIeeO76sJbW7k",
    "authDomain": "beed-sinyal-siniflandirma.firebaseapp.com",
    "databaseURL": "https://beed-sinyal-siniflandirma.firebaseio.com",
    "projectId": "beed-sinyal-siniflandirma",
    "storageBucket": "beed-sinyal-siniflandirma.firebasestorage.app",
    "messagingSenderId": "399959028519",
    "appId": "1:399959028519:web:f2736259213cd894ffef02"
}

# Firebase init
try:
    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    print("[OK] Firebase basarili!")
except Exception as e:
    print(f"[WARN] Firebase baslatilamadi: {e}")
    print("Lutfen firebase_config.py'de API bilgilerini kontrol et!")
    firebase = None
    auth = None

# Cloud Firestore init (service account gerekli)
firestore_db = None
try:
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
    if os.path.exists(service_account_path):
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        firestore_db = firestore.client()
        print("[OK] Cloud Firestore baglandi!")
    else:
        print("[WARN] serviceAccountKey.json bulunamadi, Cloud Firestore kapali.")
except Exception as e:
    print(f"[WARN] Cloud Firestore baslatilamadi: {e}")
    firestore_db = None
