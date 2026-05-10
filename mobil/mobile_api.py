from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

if __package__:
    from .ml_service import ModelService
else:
    from ml_service import ModelService


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=16, max_items=16)


app = FastAPI(
    title="BEED Mobile Inference API",
    version="1.0.0",
    description="Mobile clients send X1..X16 and receive backend ML predictions.",
)

service = ModelService()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
        return HTMLResponse(
                """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BEED Mobil Tahmin</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #0f172a;
            --panel: #111827;
            --card: #f8fafc;
            --accent: #14b8a6;
            --accent-2: #2563eb;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --border: #1f2937;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.20), transparent 28%),
                radial-gradient(circle at bottom right, rgba(37, 99, 235, 0.22), transparent 26%),
                linear-gradient(180deg, #020617 0%, #0f172a 100%);
            color: var(--text);
            padding: 20px;
        }
        .shell {
            max-width: 720px;
            margin: 0 auto;
        }
        .hero {
            padding: 24px 0 18px;
        }
        .hero h1 {
            margin: 0 0 8px;
            font-size: 32px;
            line-height: 1.1;
        }
        .hero p {
            margin: 0;
            color: var(--muted);
            max-width: 58ch;
        }
        .panel {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 18px;
            backdrop-filter: blur(18px);
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.35);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        label {
            display: block;
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 6px;
        }
        input {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(255, 255, 255, 0.95);
            color: #0f172a;
            padding: 14px 14px;
            font-size: 16px;
            outline: none;
        }
        input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.16);
        }
        .actions {
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }
        button {
            border: 0;
            border-radius: 999px;
            padding: 14px 18px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
        }
        .primary {
            background: linear-gradient(135deg, var(--accent), #22c55e);
            color: white;
            box-shadow: 0 14px 30px rgba(20, 184, 166, 0.25);
        }
        .secondary {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .result {
            margin-top: 16px;
            background: rgba(255, 255, 255, 0.96);
            color: #0f172a;
            border-radius: 18px;
            padding: 16px;
            min-height: 120px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .hint {
            margin-top: 12px;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.5;
        }
        @media (max-width: 640px) {
            body { padding: 14px; }
            .grid { grid-template-columns: 1fr; }
            .hero h1 { font-size: 28px; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <h1>BEED Mobil Tahmin</h1>
            <p>16 ozelliği gir, backend modeli calistirsin. Emulatorde bu sayfayi acip dogrudan tahmin alabilirsin.</p>
        </section>

        <section class="panel">
            <div class="grid" id="inputs"></div>

            <div class="actions">
                <button class="primary" id="predictBtn">Tahmin Et</button>
                <button class="secondary" id="fillBtn" type="button">Sifirla</button>
            </div>

            <div class="result" id="result">Sonuc burada gorunecek.</div>
            <div class="hint">Android emulatorde bu sayfayi <strong>http://10.0.2.2:8000/</strong> adresinden ac.</div>
        </section>
    </div>

    <script>
        const inputContainer = document.getElementById('inputs');
        const resultBox = document.getElementById('result');
        const predictBtn = document.getElementById('predictBtn');
        const fillBtn = document.getElementById('fillBtn');

        function buildInputs() {
            const count = 16;
            for (let i = 1; i <= count; i += 1) {
                const wrap = document.createElement('div');
                const label = document.createElement('label');
                label.setAttribute('for', `x${i}`);
                label.textContent = `X${i}`;
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = `x${i}`;
                input.value = '0';
                inputContainer.appendChild(wrap);
                wrap.appendChild(label);
                wrap.appendChild(input);
            }
        }

        function setResult(text) {
            resultBox.textContent = text;
        }

        function collectFeatures() {
            return Array.from({ length: 16 }, (_, index) => {
                const value = document.getElementById(`x${index + 1}`).value;
                return Number.parseFloat(value || '0');
            });
        }

        function resetInputs() {
            for (let i = 1; i <= 16; i += 1) {
                document.getElementById(`x${i}`).value = '0';
            }
            setResult('Sonuc burada gorunecek.');
        }

        predictBtn.addEventListener('click', async () => {
            try {
                predictBtn.disabled = true;
                setResult('Tahmin hesaplanıyor...');
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: collectFeatures() })
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Tahmin alinamadi.');
                }
                setResult(JSON.stringify(data, null, 2));
            } catch (error) {
                setResult(`Hata: ${error.message}`);
            } finally {
                predictBtn.disabled = false;
            }
        });

        fillBtn.addEventListener('click', resetInputs);

        buildInputs();
    </script>
</body>
</html>"""
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        predictions = service.predict(payload.features)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model files are missing. Train models first on desktop.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "input_feature_count": len(payload.features),
        "predictions": predictions,
    }
