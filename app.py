"""
app.py — ResilienceTrack Demand Forecast API
=============================================
Flask service that loads the trained Random Forest models (.pkl) and
exposes a /predict endpoint consumed by the PHP DemandForecastingController.

Endpoints
---------
GET  /health          — liveness check; returns model load status
POST /predict         — run RF prediction for a single disaster scenario
POST /predict/batch   — run RF predictions for multiple scenarios at once
"""

import os
import sys
import json
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow cross-origin calls from any PHP host

# ── Model loading ─────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_forecaster = None
_load_error = None


def _get_forecaster():
    """Lazy-load the DemandForecaster singleton. Thread-safe for Gunicorn workers."""
    global _forecaster, _load_error
    if _forecaster is not None:
        return _forecaster, None
    if _load_error is not None:
        return None, _load_error

    try:
        # predict.py lives in this same directory
        sys.path.insert(0, os.path.dirname(__file__))
        from predict import DemandForecaster
        _forecaster = DemandForecaster(version="v2")
        log.info("DemandForecaster loaded successfully (v2).")
        return _forecaster, None
    except Exception as exc:
        _load_error = str(exc)
        log.error("Failed to load DemandForecaster: %s", _load_error)
        return None, _load_error


# Pre-warm on startup so the first real request isn't slow
with app.app_context():
    _get_forecaster()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness + readiness probe used by Render and the PHP app."""
    forecaster, err = _get_forecaster()
    if forecaster:
        # Surface known locations and disaster types for informational use
        known_locations = sorted(forecaster.le_location.classes_.tolist())
        known_types     = sorted(forecaster.le_disaster.classes_.tolist())
        return jsonify({
            "status":           "ok",
            "model_version":    "v2",
            "models_loaded":    list(forecaster.models.keys()),
            "known_locations":  known_locations,
            "known_types":      known_types,
        }), 200
    return jsonify({"status": "error", "error": err}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
        date                 string   ISO date e.g. "2026-06-15"
        location             string   Barangay name
        disaster_type        string   e.g. "TYPHOON"
        families_affected    int
        individuals_affected int

    Returns JSON matching the shape expected by DemandForecastingController::mlForecast().
    """
    forecaster, err = _get_forecaster()
    if not forecaster:
        return jsonify({"success": False, "error": f"Model unavailable: {err}"}), 503

    # ── Parse request body ────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        # Also accept form-encoded (fallback)
        data = request.form.to_dict()

    if not data:
        return jsonify({"success": False, "error": "No JSON body received."}), 400

    required = ["date", "location", "disaster_type",
                "families_affected", "individuals_affected"]
    missing = [f for f in required
               if f not in data or str(data[f]).strip() == ""]
    if missing:
        return jsonify({
            "success": False,
            "error": f"Missing required fields: {', '.join(missing)}"
        }), 422

    # ── Coerce + validate numerics ────────────────────────────────────────────
    try:
        families    = int(float(data["families_affected"]))
        individuals = int(float(data["individuals_affected"]))
        if families < 0 or individuals < 0:
            raise ValueError("Negative count")
    except (TypeError, ValueError) as exc:
        return jsonify({
            "success": False,
            "error": f"Invalid numeric input: {exc}"
        }), 422

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = forecaster.predict(
            date=str(data["date"]),
            location=str(data["location"]),
            disaster_type=str(data["disaster_type"]),
            families_affected=families,
            individuals_affected=individuals,
        )
    except Exception as exc:
        log.error("Prediction error: %s", exc)
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500

    # ── Build response (matches shape PHP controller expects) ─────────────────
    loc_upper = str(data["location"]).upper().strip()
    dt_upper  = str(data["disaster_type"]).upper().strip()
    loc_known = loc_upper in [c.upper() for c in forecaster.le_location.classes_]
    dt_known  = dt_upper  in [c.upper() for c in forecaster.le_disaster.classes_]

    response = {
        "success":             True,
        "ffp_pack":            max(0, int(result.get("ffp_pack",       0))),
        "heb":                 max(0, int(result.get("heb",            0))),
        "rtef":                max(0, int(result.get("rtef",           0))),
        "shelter_kits":        max(0, int(result.get("shelter_kits",   0))),
        "hygiene_kits":        max(0, int(result.get("hygiene_kits",   0))),
        "sleeping_kits":       max(0, int(result.get("sleeping_kits",  0))),
        "kitchen_kits":        max(0, int(result.get("kitchen_kits",   0))),
        "family_kits":         max(0, int(result.get("family_kits",    0))),
        "portable_water":      max(0, int(result.get("portable_water", 0))),
        "confidence_note":     result.get("confidence_note", "Random Forest estimate."),
        "model_version":       "Random Forest v2",
        "model_source":        "rf",          # used by PHP to label the source
        "location_known":      loc_known,
        "disaster_type_known": dt_known,
        "input_summary": {
            "date":                 str(data["date"]),
            "location":             str(data["location"]),
            "disaster_type":        str(data["disaster_type"]),
            "families_affected":    families,
            "individuals_affected": individuals,
        },
    }

    return jsonify(response), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    POST /predict/batch
    Body (JSON): array of scenario objects (same shape as /predict body).
    Returns array of prediction objects.
    """
    forecaster, err = _get_forecaster()
    if not forecaster:
        return jsonify({"success": False, "error": f"Model unavailable: {err}"}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"success": False, "error": "Expected a JSON array."}), 400

    results = []
    for i, item in enumerate(data):
        try:
            families    = int(float(item.get("families_affected", 0)))
            individuals = int(float(item.get("individuals_affected", 0)))
            pred = forecaster.predict(
                date=str(item.get("date", "")),
                location=str(item.get("location", "")),
                disaster_type=str(item.get("disaster_type", "")),
                families_affected=families,
                individuals_affected=individuals,
            )
            pred["success"] = True
            pred["model_source"] = "rf"
            pred["model_version"] = "Random Forest v2"
            results.append(pred)
        except Exception as exc:
            results.append({"success": False, "error": str(exc), "index": i})

    return jsonify(results), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
