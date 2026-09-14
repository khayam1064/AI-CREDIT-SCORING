"""P06 — Stability Pillar Scorer (F2, F5, F6) | Weight: 8%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.08
PILLAR_NAME = "Stability"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_stability.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_stability.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Stability score (0-100).
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
                
            pred_sim_age = model.predict(X)
            # Map SIM age days to a score of 0-100 (5+ years = 100)
            score_series = (pred_sim_age / (365.25 * 5) * 100)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(0.0, index=df.index)
    if "years_at_address" in df.columns:
        yrs = df["years_at_address"].fillna(0).clip(0, 20)
        res_score = np.where(yrs >= 10, 30,
                    np.where(yrs >= 5, 25,
                    np.where(yrs >= 3, 20,
                    np.where(yrs >= 1, 10,
                    np.where(yrs >= 0.5, 5, 0)))))
        s += res_score

    if "residential_status" in df.columns:
        own_bonus = np.where(df["residential_status"] == "Owned", 5.0,
                    np.where(df["residential_status"] == "Government", 3.0, 0.0))
        s += own_bonus

    if "max_account_age_months" in df.columns:
        acc_mo = df["max_account_age_months"].fillna(0).clip(0, 120)
        s += (acc_mo / 120 * 25).round(2)
    elif "account_tenure_months" in df.columns:
        acc_mo = df["account_tenure_months"].fillna(0).clip(0, 120)
        s += (acc_mo / 120 * 25).round(2)

    if "sim_age_days" in df.columns:
        sim_yrs = df["sim_age_days"].fillna(0) / 365.25
        sim_score = np.where(sim_yrs >= 5, 20,
                    np.where(sim_yrs >= 3, 15,
                    np.where(sim_yrs >= 1, 10,
                    np.where(sim_yrs >= 0.5, 5, 0))))
        s += sim_score

    job_col = "job_tenure_months" if "job_tenure_months" in df.columns else (
              "employment_tenure_months" if "employment_tenure_months" in df.columns else None)
    if job_col:
        job_mo = df[job_col].fillna(0).clip(0, 120)
        s += (job_mo / 120 * 20).round(2)

    return s.clip(0, 100).round(2)
