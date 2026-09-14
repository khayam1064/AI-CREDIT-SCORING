"""P11 — Collection Risk Pillar Scorer (F6, F13) | Weight: 5%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.05
PILLAR_NAME = "Collection_Risk"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_collection_risk.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_collection_risk.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Collection Risk score (0-100).
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
                
            pred_contacts = model.predict(X)
            # Higher contact count = higher risk = lower score
            # Scale contact counts (normally 0 to 20) to 0-100 score
            score_series = 100.0 - (pred_contacts * 5.0)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(70.0, index=df.index)

    if "overdraft_utilization_rate" in df.columns:
        od = df["overdraft_utilization_rate"].fillna(0).clip(0, 1)
    elif "overdraft_utilization" in df.columns:
        od = df["overdraft_utilization"].fillna(0).clip(0, 1)
    else:
        od = pd.Series(0.3, index=df.index)
    s -= (od * 20).round(2)

    if "income_stability_score" in df.columns:
        stab = df["income_stability_score"].fillna(50).clip(0, 100)
        s += (stab - 50) * 0.20

    if "ldr_avg_dpd_last_12m" in df.columns:
        dpd = df["ldr_avg_dpd_last_12m"].fillna(0)
        s -= np.where(dpd > 30, 25,
             np.where(dpd > 15, 15,
             np.where(dpd > 5, 8, 0)))
    if "ldr_collection_contacts_count" in df.columns:
        col = df["ldr_collection_contacts_count"].fillna(0)
        s -= (col * 5).clip(0, 25)

    if "phone_type" in df.columns:
        s += np.where(df["phone_type"] == "Postpaid", 5.0, 0.0)
    if "has_mobile_banking" in df.columns:
        s += df["has_mobile_banking"].fillna(False).astype(float) * 5

    if "wal_chargeback_count_ever" in df.columns:
        cb = df["wal_chargeback_count_ever"].fillna(0)
        s -= (cb * 8).clip(0, 20)

    if "nsf_count_total" in df.columns:
        s -= (df["nsf_count_total"].fillna(0) * 5).clip(0, 15)

    return s.clip(0, 100).round(2)
