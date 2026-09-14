"""P02 — Fraud Risk Pillar Scorer (F3, F4, F14) | Weight: Gate + 7%"""
from attr import s
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.07
PILLAR_NAME = "Fraud_Risk"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_fraud_risk.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_fraud_risk.joblib"


def hard_gate(df: pd.DataFrame) -> pd.Series:
    """Returns True where customer must be DECLINED regardless of score."""
    failed = pd.Series(False, index=df.index)
    if "has_fraud_flag" in df.columns:
        failed |= df["has_fraud_flag"].fillna(False).astype(bool)
    if "any_fraud_flag" in df.columns:
        failed |= df["any_fraud_flag"].fillna(False).astype(bool)
    if "dev_emulator_detected" in df.columns:
        failed |= df["dev_emulator_detected"].fillna(False).astype(bool)
    if "any_aml_flag" in df.columns:
        failed |= df["any_aml_flag"].fillna(False).astype(bool)
    return failed


def score(df: pd.DataFrame) -> pd.Series:
    """
    Fraud risk score (0-100, higher = lower risk).
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
                
            pred_risk = model.predict(X)
            # Invert: 100 - predicted risk score
            return pd.Series(100.0 - pred_risk, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    risk = pd.Series(0.0, index=df.index)
    if "has_fraud_flag" in df.columns:
        risk += df["has_fraud_flag"].fillna(False).astype(float) * 60
    if "dev_emulator_detected" in df.columns:
        risk += df["dev_emulator_detected"].fillna(False).astype(float) * 40
    if "dev_is_rooted" in df.columns:
        risk += df["dev_is_rooted"].fillna(False).astype(float) * 25
    if "any_aml_flag" in df.columns:
        risk += df["any_aml_flag"].fillna(False).astype(float) * 50

    if "dev_vpn_detected" in df.columns:
        risk += df["dev_vpn_detected"].fillna(False).astype(float) * 10
    if "dev_proxy_detected" in df.columns:
        risk += df["dev_proxy_detected"].fillna(False).astype(float) * 8
    if "dev_timezone_gps_mismatch" in df.columns:
        risk += df["dev_timezone_gps_mismatch"].fillna(False).astype(float) * 12
    if "dev_impossible_travel_flag" in df.columns:
        risk += df["dev_impossible_travel_flag"].fillna(False).astype(float) * 15
    if "dev_sim_swap_count_6m" in df.columns:
        risk += (df["dev_sim_swap_count_6m"].fillna(0) * 5).clip(0, 20)

    if "dev_copy_paste_id_field" in df.columns:
        risk += df["dev_copy_paste_id_field"].fillna(False).astype(float) * 8
    if "dev_session_replay_anomaly_score" in df.columns:
        risk += (df["dev_session_replay_anomaly_score"].fillna(0) / 100 * 10)
    if "dev_behavioral_risk_score" in df.columns:
        risk += (df["dev_behavioral_risk_score"].fillna(0) / 100 * 15)

    if "app_velocity_risk_score" in df.columns:
        risk += (df["app_velocity_risk_score"].fillna(0) / 100 * 12)
    if "app_applications_same_ip_24h" in df.columns:
        risk += ((df["app_applications_same_ip_24h"].fillna(1) - 1) * 5).clip(0, 15)
    if "app_cross_lender_velocity_30d" in df.columns:
        risk += (df["app_cross_lender_velocity_30d"].fillna(0) * 6).clip(0, 20)

    return (100.0 - risk.clip(0, 100)).round(2)
