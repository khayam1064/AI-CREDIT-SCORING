"""P10 — Relationship Pillar Scorer (F13) | Weight: 0-25% (dynamic)"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

PILLAR_NAME = "Relationship"
MAX_WEIGHT = 0.25
RAMP_MONTHS = 36

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_relationship.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_relationship.joblib"


def effective_weight(df: pd.DataFrame) -> pd.Series:
    if "is_existing_customer" not in df.columns:
        return pd.Series(0.0, index=df.index)
    is_existing = df["is_existing_customer"].fillna(False).astype(bool)

    if "months_as_customer" in df.columns:
        months = df["months_as_customer"].fillna(0)
    elif "ldr_months_as_customer" in df.columns:
        months = df["ldr_months_as_customer"].fillna(0)
    else:
        months = pd.Series(0, index=df.index)

    ramp = np.minimum(1.0, months / RAMP_MONTHS)
    weight = np.where(is_existing, ramp * MAX_WEIGHT, 0.0)
    return pd.Series(weight, index=df.index)


def score(df: pd.DataFrame) -> pd.Series:
    """
    Relationship score (0-100).
    Uses the trained LightGBM model if available.
    """
    is_existing = df.get("is_existing_customer", pd.Series(False, index=df.index)).fillna(False).astype(bool)

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
                
            pred_rel = model.predict(X)
            # Relationship score is already trained on ldr_relationship_score range (0-100)
            score_series = pd.Series(pred_rel, index=df.index)
            score_series = np.where(is_existing, score_series, 0.0)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)
    if "ldr_on_time_payment_rate" in df.columns:
        otp = df["ldr_on_time_payment_rate"].fillna(0.5).clip(0, 1)
        s = otp * 40 + 30
    else:
        s = pd.Series(30.0, index=df.index)

    if "ldr_prior_loans_count" in df.columns:
        n_loans = df["ldr_prior_loans_count"].fillna(0).clip(0, 8)
        s += (n_loans / 8 * 15).round(2)

    if "ldr_early_repayments_count" in df.columns:
        er = df["ldr_early_repayments_count"].fillna(0).clip(0, 5)
        s += (er / 5 * 10).round(2)

    if "ldr_avg_dpd_last_12m" in df.columns:
        dpd = df["ldr_avg_dpd_last_12m"].fillna(0)
        s -= (dpd / 30 * 20).clip(0, 20).round(2)
    if "ldr_max_dpd_ever" in df.columns:
        max_dpd = df["ldr_max_dpd_ever"].fillna(0)
        s -= np.where(max_dpd > 90, 15,
             np.where(max_dpd > 30, 8,
             np.where(max_dpd > 7, 3, 0)))

    if "ldr_collection_contacts_count" in df.columns:
        col = df["ldr_collection_contacts_count"].fillna(0)
        s -= (col * 3).clip(0, 20).round(2)

    if "ldr_app_engagement_recency_days" in df.columns:
        recency = df["ldr_app_engagement_recency_days"].fillna(60)
        s += np.where(recency <= 7, 5,
             np.where(recency <= 30, 3, 0))

    if "customer_segment" in df.columns:
        seg_bonus = df["customer_segment"].map({
            "Private Banking": 8, "Premium": 5, "Affluent": 3, "Mass": 0
        }).fillna(0)
        s += seg_bonus

    s = np.where(is_existing, s, 0.0)
    return pd.Series(s, index=df.index).clip(0, 100).round(2)
