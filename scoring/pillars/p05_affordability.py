"""P05 — Affordability Pillar Scorer (F6, F12) | Weight: 15%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.15
PILLAR_NAME = "Affordability"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_affordability.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_affordability.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Affordability score (0-100).
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
                
            pred_foir = model.predict(X)
            # Higher FOIR = lower score. Map FOIR % (normally 0 to 100) to 0-100 score.
            # FOIR <= 15% gets 100, FOIR >= 50% gets 0
            score_series = 100.0 - (pred_foir - 15) * (100 / 35)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)
    if "foir_pct" in df.columns:
        foir = df["foir_pct"].fillna(50.0)
    else:
        inc = df.get("total_monthly_income", pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
        utility = df.get("utility_debit_amt_12m", pd.Series(0, index=df.index)).fillna(0) / 12
        foir = (utility / inc * 100).clip(0, 100)

    foir_score = np.where(foir < 15, 100,
                 np.where(foir < 30, 80,
                 np.where(foir < 40, 60,
                 np.where(foir < 45, 40, 0.0))))
    s = pd.Series(foir_score.astype(float), index=df.index)

    inc = df.get("total_monthly_income", df.get("net_monthly_salary", pd.Series(0, index=df.index))).fillna(0)
    utility_monthly = df.get("utility_debit_amt_12m", pd.Series(0, index=df.index)).fillna(0) / 12
    ndi = (inc - utility_monthly).clip(lower=0)

    ndi_bonus = np.where(ndi > 100000, 10,
                np.where(ndi > 50000, 6,
                np.where(ndi > 25000, 3, 0)))
    s += ndi_bonus

    cr = df.get("total_credit_amt_12m", pd.Series(0, index=df.index)).fillna(0)
    db = df.get("total_debit_amt_12m", pd.Series(0, index=df.index)).fillna(0)
    spending_ratio = np.where(cr > 0, db / cr.clip(lower=1), 1.0)
    s -= np.where(spending_ratio > 1.0, 10.0, np.where(spending_ratio > 0.90, 5.0, 0.0))

    if "balance_to_income_ratio" in df.columns:
        bir = df["balance_to_income_ratio"].fillna(0).clip(0, 5)
        s += (bir * 2).clip(0, 8)

    return s.clip(0, 100).round(2)
