# ResilienceTrack — Forecast API

Flask service that serves the trained Random Forest v2 models as an HTTP API.
Deployed on [Render](https://render.com) (free tier, Singapore region).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check; returns model status and known locations/types |
| POST | `/predict` | Single-scenario prediction |
| POST | `/predict/batch` | Batch predictions |

### POST /predict — request body (JSON)

```json
{
  "date": "2026-06-15",
  "location": "FABRICA",
  "disaster_type": "TYPHOON",
  "families_affected": 650,
  "individuals_affected": 2600
}
```

### POST /predict — response

```json
{
  "success": true,
  "ffp_pack": 812,
  "heb": 150,
  "rtef": 132,
  "shelter_kits": 195,
  "hygiene_kits": 195,
  "sleeping_kits": 195,
  "kitchen_kits": 33,
  "family_kits": 48,
  "portable_water": 860,
  "confidence_note": "Within training distribution",
  "model_version": "Random Forest v2",
  "model_source": "rf",
  "location_known": true,
  "disaster_type_known": true,
  "input_summary": { ... }
}
```

---

## Deploying to Render (step-by-step)

### 1. Create a GitHub repository for this folder only

```bash
# From inside the forecast-api/ folder:
git init
git add .
git commit -m "Initial forecast API"
git remote add origin https://github.com/YOUR_USERNAME/resiliencetrack-forecast-api.git
git push -u origin main
```

> **Important:** The `.pkl` model files are included in the repo so Render can load them.
> They are ~5 MB total — fine for GitHub.

### 2. Create a Web Service on Render

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub account and select `resiliencetrack-forecast-api`
3. Settings:
   - **Name:** `resiliencetrack-forecast-api`
   - **Region:** Singapore
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --timeout 60 --bind 0.0.0.0:$PORT`
   - **Plan:** Free
4. Click **Create Web Service**

Render will build and deploy. Your URL will be:
`https://resiliencetrack-forecast-api.onrender.com`

Test it: `https://resiliencetrack-forecast-api.onrender.com/health`

### 3. Set RF_API_URL in the PHP app

Open `config/ai.php` and set:

```php
define('RF_API_URL', 'https://resiliencetrack-forecast-api.onrender.com');
```

Or if you use environment variables on your hosting provider, set:
```
RF_API_URL=https://resiliencetrack-forecast-api.onrender.com
```

That's it. The PHP app will now call the Random Forest model first,
fall back to OpenRouter LLM if unavailable, and fall back to the
statistical model if both fail.

---

## Free tier note

Render free services **spin down after 15 minutes of inactivity** and
take ~20–30 seconds to cold-start on the next request. The PHP controller
uses a 35-second timeout to handle this gracefully. On warm instances,
responses are typically under 1 second.

## Local testing

```bash
pip install -r requirements.txt
python app.py
# Server running on http://localhost:5000

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-06-01","location":"FABRICA","disaster_type":"TYPHOON","families_affected":500,"individuals_affected":2000}'
```
