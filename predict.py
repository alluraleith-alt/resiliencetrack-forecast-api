"""
predict.py — ResilienceTrack Demand Forecasting Prediction Module
=================================================================
API-ready module for making demand predictions using the trained
Random Forest models.

Usage example:
    from ml.predict import DemandForecaster

    forecaster = DemandForecaster()
    result = forecaster.predict(
        date="2026-03-15",
        location="FABRICA",
        disaster_type="TYPHOON",
        families_affected=500,
        individuals_affected=1800,
    )
    print(result)
    # {
    #   "ffp_pack": 123,
    #   "hygiene_kits": 0,
    #   "sleeping_kits": 45,
    #   "model_version": "v1",
    #   "confidence_note": "..."
    # }

Or run directly for a quick test:
    python ml/predict.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ML_DIR     = Path(__file__).parent
MODELS_DIR = ML_DIR / "models"

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Run: python -m pip install joblib scikit-learn")
    sys.exit(1)


class DemandForecaster:
    """
    Loads the trained Random Forest models and preprocessors,
    then provides a predict() method for new disaster-event records.
    """

    def __init__(self, version: str = "v2"):
        self.version = version
        self._load_models()

    def _load_models(self):
        """Load all trained models and the preprocessor from disk."""
        pp_path = MODELS_DIR / f"preprocessor_{self.version}.pkl"
        if not pp_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {pp_path}. "
                "Run ml/train.py first."
            )

        self.preprocessor = joblib.load(pp_path)
        self.le_location   = self.preprocessor["le_location"]
        self.le_disaster   = self.preprocessor["le_disaster"]
        self.severity_map  = self.preprocessor["severity_map"]
        self.feature_cols  = self.preprocessor["feature_cols"]
        self.global_ffp_ppf = self.preprocessor["global_ffp_per_family"]
        self.loc_history   = self.preprocessor.get("location_ffp_history", {})

        self.models = {}
        for target in self.preprocessor["target_cols"]:
            model_path = MODELS_DIR / f"rf_{target}_{self.version}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.models[target] = joblib.load(model_path)

    def _encode_location(self, location: str) -> int:
        """Encode location, falling back to nearest known value for unseen locations."""
        loc_upper = location.upper().strip()
        known = list(self.le_location.classes_)
        if loc_upper in known:
            return int(self.le_location.transform([loc_upper])[0])
        # Unseen location: return median encoding index
        return len(known) // 2

    def _encode_disaster(self, disaster_type: str) -> int:
        """Encode disaster type, falling back to median for unseen types."""
        dt_upper = disaster_type.upper().strip()
        known = list(self.le_disaster.classes_)
        if dt_upper in known:
            return int(self.le_disaster.transform([dt_upper])[0])
        return len(known) // 2

    def build_feature_row(
        self,
        date: str,
        location: str,
        disaster_type: str,
        families_affected: int,
        individuals_affected: int,
        loc_hist_ffp_per_family: float = None,
    ) -> pd.DataFrame:
        """
        Construct the feature vector for a single prediction request.

        Parameters
        ----------
        date                    : ISO date string e.g. "2026-03-15"
        location                : Barangay name
        disaster_type           : Disaster category string
        families_affected       : Number of affected families
        individuals_affected    : Number of affected individuals
        loc_hist_ffp_per_family : Historical FFP-per-family ratio for this
                                  location (optional — uses stored history if None)
        """
        dt = pd.to_datetime(date)
        month     = dt.month
        year      = dt.year
        is_wet    = int(6 <= month <= 11)
        severity  = self.severity_map.get(disaster_type.upper().strip(), 2)
        fam       = max(int(families_affected), 0)
        ind       = max(int(individuals_affected), 0)
        ppf       = ind / fam if fam > 0 else 0.0

        if loc_hist_ffp_per_family is None:
            loc_key = location.upper().strip()
            loc_hist_ffp_per_family = self.loc_history.get(loc_key, self.global_ffp_ppf)

        row = {
            "location_enc":           self._encode_location(location),
            "disaster_type_enc":       self._encode_disaster(disaster_type),
            "disaster_severity":       severity,
            "year":                    year,
            "month":                   month,
            "is_wet_season":           is_wet,
            "families_affected":       float(fam),
            "individuals_affected":    float(ind),
            "indiv_per_family":        ppf,
            "loc_hist_ffp_per_family": float(loc_hist_ffp_per_family),
        }

        return pd.DataFrame([row], columns=self.feature_cols)

    def predict(
        self,
        date: str,
        location: str,
        disaster_type: str,
        families_affected: int,
        individuals_affected: int,
        loc_hist_ffp_per_family: float = None,
    ) -> dict:
        """
        Predict demand quantities for a disaster event.

        Returns
        -------
        dict with keys:
            ffp_pack, hygiene_kits, sleeping_kits  (non-negative integers)
            model_version
            confidence_note
        """
        X = self.build_feature_row(
            date=date,
            location=location,
            disaster_type=disaster_type,
            families_affected=families_affected,
            individuals_affected=individuals_affected,
            loc_hist_ffp_per_family=loc_hist_ffp_per_family,
        )

        result = {"model_version": self.version}
        for target, model in self.models.items():
            raw_pred = float(model.predict(X)[0])
            result[target] = max(0, int(round(raw_pred)))

        # Confidence note
        loc_known = location.upper().strip() in [c.upper() for c in self.le_location.classes_]
        dt_known  = disaster_type.upper().strip() in [c.upper() for c in self.le_disaster.classes_]
        notes = []
        if not loc_known:
            notes.append(f"location '{location}' not in training data")
        if not dt_known:
            notes.append(f"disaster_type '{disaster_type}' not in training data")
        if families_affected > 5000:
            notes.append("families_affected exceeds training data range — extrapolation")
        result["confidence_note"] = (
            "Low confidence: " + "; ".join(notes) if notes
            else "Within training distribution"
        )

        return result

    def predict_batch(self, records: list) -> list:
        """
        Predict for a list of dicts, each containing the same keys as predict().
        """
        return [self.predict(**r) for r in records]


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading forecaster...")
    try:
        f = DemandForecaster()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    test_cases = [
        {
            "date": "2026-06-01",
            "location": "FABRICA",
            "disaster_type": "TYPHOON",
            "families_affected": 800,
            "individuals_affected": 3200,
        },
        {
            "date": "2026-08-20",
            "location": "PARAISO",
            "disaster_type": "FLOOD",
            "families_affected": 300,
            "individuals_affected": 1200,
        },
        {
            "date": "2025-12-10",
            "location": "OLD SAGAY",
            "disaster_type": "SUPER TYPHOON",
            "families_affected": 1500,
            "individuals_affected": 4500,
        },
        {
            "date": "2026-03-01",
            "location": "NEW BARANGAY",   # unseen location — tests fallback
            "disaster_type": "FLOOD",
            "families_affected": 200,
            "individuals_affected": 800,
        },
    ]

    print("\nPrediction results:")
    print(f"{'Date':<12} {'Location':<20} {'Disaster':<22} "
          f"{'FFP':>6} {'Hygiene':>8} {'Sleeping':>9} Note")
    print("-" * 100)
    for tc in test_cases:
        r = f.predict(**tc)
        print(f"{tc['date']:<12} {tc['location']:<20} {tc['disaster_type']:<22} "
              f"{r['ffp_pack']:>6} {r['hygiene_kits']:>8} {r['sleeping_kits']:>9}  "
              f"{r['confidence_note']}")
