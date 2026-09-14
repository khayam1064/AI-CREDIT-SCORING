"""P07 — Behavioral Pillar Scorer (F4, F6) | Weight: 7%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.07
PILLAR_NAME = "Behavioral"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_behavioral.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_behavioral.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Behavioral score (0-100).
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
                
            pred_sessions = model.predict(X)
            # Scale session count (normally 0 to 20) to 40-100 range for credit score calibration
            score_series = 40.0 + (pred_sessions * 3.0)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)
    if "income_stability_score" in df.columns:
        stab = df["income_stability_score"].fillna(50).clip(0, 100)
        s = stab * 0.30 + 35

    if "dev_behavioral_risk_score" in df.columns:
        brisk = df["dev_behavioral_risk_score"].fillna(50).clip(0, 100)
        s += (100 - brisk) * 0.25

    if "dev_session_count_7d" in df.columns:
        sess = df["dev_session_count_7d"].fillna(1).clip(1, 20)
        s += (sess / 20 * 15).round(2)

    if "dev_avg_session_duration_sec" in df.columns:
        dur = df["dev_avg_session_duration_sec"].fillna(60).clip(0, 600)
        ideal = np.where((dur >= 120) & (dur <= 480), 10,
                np.where((dur >= 60) & (dur < 120), 5,
                np.where(dur > 480, 6, 2)))
        s += ideal

    if "dev_copy_paste_id_field" in df.columns:
        s -= df["dev_copy_paste_id_field"].fillna(False).astype(float) * 10

    if "income_confidence_score" in df.columns:
        conf = df["income_confidence_score"].fillna(50).clip(0, 100)
        s += (conf - 50) * 0.10

    if "wal_wallet_credit_signal" in df.columns:
        ws = df["wal_wallet_credit_signal"].fillna(0).clip(0, 100)
        s += (ws / 100 * 5).round(2)

    return s.clip(0, 100).round(2)
