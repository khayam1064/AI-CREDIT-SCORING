"""P09 — Bureau Proxy Pillar Scorer (F9, F6) | Weight: 12% (dynamic)"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.12
PILLAR_NAME = "Bureau_Proxy"
THIN_FILE_THRESHOLD_MONTHS = 6

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_bureau_proxy.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_bureau_proxy.joblib"


def effective_weight(df: pd.DataFrame) -> pd.Series:
    if "max_account_age_months" in df.columns:
        tenure = df["max_account_age_months"].fillna(0)
    elif "account_tenure_months" in df.columns:
        tenure = df["account_tenure_months"].fillna(0)
    else:
        return pd.Series(0.0, index=df.index)
    return np.where(tenure >= THIN_FILE_THRESHOLD_MONTHS, WEIGHT, 0.0)


def score(df: pd.DataFrame) -> pd.Series:
    """
    Bureau proxy score (0-100).
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
                
            pred_od = model.predict(X)
            # Map overdraft utilization (normally 0 to 1) to score (lower utilization is better)
            score_series = 100.0 - (pred_od * 100)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)
    if "max_account_age_months" in df.columns:
        mo = df["max_account_age_months"].fillna(0)
    elif "account_tenure_months" in df.columns:
        mo = df["account_tenure_months"].fillna(0)
    else:
        mo = pd.Series(0, index=df.index)

    tenure_score = np.where(mo >= 60, 35,
                   np.where(mo >= 36, 28,
                   np.where(mo >= 24, 22,
                   np.where(mo >= 12, 15,
                   np.where(mo >= 6,  8, 0)))))
    s = pd.Series(tenure_score.astype(float), index=df.index)

    if "overdraft_utilization_rate" in df.columns:
        od = df["overdraft_utilization_rate"].fillna(0).clip(0, 1)
    elif "overdraft_utilization" in df.columns:
        od = df["overdraft_utilization"].fillna(0).clip(0, 1)
    else:
        od = pd.Series(0.5, index=df.index)

    od_score = np.where(od < 0.20, 25,
               np.where(od < 0.40, 20,
               np.where(od < 0.60, 14,
               np.where(od < 0.80, 8, 2))))
    s += od_score

    if "ldr_avg_dpd_last_12m" in df.columns:
        dpd = df["ldr_avg_dpd_last_12m"].fillna(0)
        dpd_score = np.where(dpd == 0, 25,
                    np.where(dpd < 5, 18,
                    np.where(dpd < 15, 10,
                    np.where(dpd < 30, 5, 0))))
        s += dpd_score
    else:
        s += 12.5

    if "number_of_accounts" in df.columns or "bank_account_count" in df.columns:
        n = df.get("number_of_accounts", df.get("bank_account_count", pd.Series(1, index=df.index))).fillna(1)
        mix_score = np.where(n >= 3, 15,
                    np.where(n == 2, 10,
                    np.where(n == 1, 5, 0)))
        s += mix_score

    return s.clip(0, 100).round(2)
