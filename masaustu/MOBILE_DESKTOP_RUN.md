# BEED - Masaustu ve Mobil Ayrimi

Bu proje iki calisma moduna ayrildi:

1. Masaustu uygulama (PyQt5): masaustu/desktop_app.py
2. Mobil istemci icin backend API (FastAPI): mobil/mobile_api.py

## 1) Masaustu Baslatma

```powershell
python masaustu/desktop_app.py
```

## 2) Mobil API Baslatma

```powershell
uvicorn mobil.mobile_api:app --host 0.0.0.0 --port 8000
```

## API Sozlesmesi

### Health
GET /health

Donus:
```json
{"status":"ok"}
```

### Predict
POST /predict

Istek:
```json
{
  "features": [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]
}
```

Donus:
```json
{
  "input_feature_count": 16,
  "predictions": {
    "cnn": {
      "predicted_class_index": 1,
      "predicted_y_label": "1",
      "confidence": 0.91,
      "probabilities": [0.02, 0.91, 0.07]
    },
    "svm": {
      "predicted_class_index": 1,
      "predicted_y_label": "1",
      "confidence": null,
      "probabilities": null
    },
    "lstm": {
      "predicted_class_index": 2,
      "predicted_y_label": "2",
      "confidence": 0.83,
      "probabilities": [0.08, 0.09, 0.83]
    }
  }
}
```

Not:
- Telefon modeli calistirmaz, sadece API'ye veri yollar.
- masaustu/trained_models.pkl yoksa once masaustunden egitim yapilmalidir.
