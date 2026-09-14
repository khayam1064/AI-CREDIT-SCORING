"""P01 — Identity & Trust Pillar Scorer (F1, F2) | Weight: 8%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.08
PILLAR_NAME = "Identity_Trust"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_identity_trust.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_identity_trust.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Identity/Trust score (0-100).
    Uses the trained LightGBM classification model if available,
    otherwise falls back to rule-based coherence checks.
    """
    # ── Try ML Model Inference ───────────────────────────────────────────────
    if MODEL_PATH.exists() and FEATS_PATH.exists():
        try:
            model, features = _get_cached_artifacts()
            if model is None or features is None:
                return s
            
            # Extract CNIC validity target if missing from df
            df_temp = df.copy()
            if "cnic" in df_temp.columns:
                df_temp["is_cnic_valid"] = df_temp["cnic"].astype(str).str.match(r"^\d{5}-\d{7}-\d$", na=False).astype(int)
            
            X = df_temp[features].copy()
            cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                
            # LightGBM Classifier predict_proba output
            probs = model.predict_proba(X)[:, 1]
            return pd.Series(probs * 100, index=df.index).round(2)
        except Exception:
            pass

    # ── Fallback: Heuristics & Rules ─────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)

    if "cnic" in df.columns:
        cnic_valid = df["cnic"].astype(str).str.match(r"^\d{5}-\d{7}-\d$", na=False)
        s += np.where(cnic_valid, 15.0, -25.0)
    else:
        s -= 10.0

    if "nationality" in df.columns:
        s += np.where(df["nationality"].str.upper() == "PAKISTANI", 10.0, 0.0)

    if "age" in df.columns:
        age = df["age"].fillna(0)
        s += np.where((age >= 18) & (age <= 70), 10.0,
             np.where((age > 70) & (age <= 75), 5.0, -15.0))

    if "dob" in df.columns and "age" in df.columns:
        dob_dt = pd.to_datetime(df["dob"], errors="coerce")
        calc_age = 2026 - dob_dt.dt.year
        age_match = (df["age"] - calc_age).abs() <= 1
        s += np.where(age_match, 8.0, -10.0)

    if "country" in df.columns:
        s += np.where(df["country"].str.upper() == "PAKISTAN", 7.0, 0.0)

    return s.clip(0.0, 100.0).round(2)
