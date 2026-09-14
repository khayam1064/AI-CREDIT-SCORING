"""P08 — Digital Footprint Pillar Scorer (F11) | Weight: 4%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.04
PILLAR_NAME = "Digital_Footprint"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_digital_footprint.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_digital_footprint.joblib"

EMAIL_PROVIDER_QUALITY = {
    "gmail.com": 90,
    "outlook.com": 88,
    "yahoo.com": 80,
    "hotmail.com": 78,
}


def score(df: pd.DataFrame) -> pd.Series:
    """
    Digital Footprint score (0-100).
    Uses the trained LightGBM model if available.
    """
    # ── Try ML Model Inference ───────────────────────────────────────────────
    if MODEL_PATH.exists() and FEATS_PATH.exists():
        try:
            model, features = _get_cached_artifacts()
            if model is None or features is None:
                return s
            
            X = df[features].copy()
            cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            for col in cat_cols:
                X[col] = X[col].astype("category")
                
            pred_maturity = model.predict(X)
            # Map digital maturity index (0 to 100) directly to score
            return pd.Series(pred_maturity, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(0.0, index=df.index)
    if "email_age_days" in df.columns:
        ea = df["email_age_days"].fillna(0).clip(0, 5475)
        s += (ea / 5475 * 30).round(2)

    if "sim_age_days" in df.columns:
        sa = df["sim_age_days"].fillna(0).clip(0, 3650)
        s += (sa / 3650 * 25).round(2)

    if "email_provider" in df.columns:
        ep_scores = df["email_provider"].map(EMAIL_PROVIDER_QUALITY).fillna(50)
        s += (ep_scores / 100 * 15).round(2)

    if "phone_type" in df.columns:
        s += np.where(df["phone_type"] == "Postpaid", 15.0, 8.0)

    if "dev_vpn_detected" in df.columns:
        s -= df["dev_vpn_detected"].fillna(False).astype(float) * 8
    if "dev_proxy_detected" in df.columns:
        s -= df["dev_proxy_detected"].fillna(False).astype(float) * 6

    if "dev_app_integrity_passed" in df.columns:
        s += df["dev_app_integrity_passed"].fillna(True).astype(float) * 5

    if "digital_maturity_index" in df.columns:
        s = s * 0.6 + df["digital_maturity_index"].fillna(50) * 0.4

    return s.clip(0, 100).round(2)
