"""P04 — Cash Flow Pillar Scorer (F6, F8) | Weight: 15%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.15
PILLAR_NAME = "Cash_Flow"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_pillar_cash_flow.joblib"
_MODEL_CACHE = {}

def _get_cached_artifacts():
    if not _MODEL_CACHE:
        if MODEL_PATH.exists() and FEATS_PATH.exists():
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['feats'] = joblib.load(FEATS_PATH)
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('feats')

FEATS_PATH = BASE_DIR / "models" / "features_cash_flow.joblib"


def score(df: pd.DataFrame) -> pd.Series:
    """
    Cash flow score (0-100).
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
                
            pred_ratio = model.predict(X)
            # Scale savings ratio (typically 0.0 to 0.45) to 45-100 range for credit score calibration
            score_series = 45.0 + (pred_ratio * 120.0)
            return pd.Series(score_series, index=df.index).clip(0, 100).round(2)
        except Exception:
            pass

    # ── Fallback Heuristics ──────────────────────────────────────────────────
    s = pd.Series(50.0, index=df.index)
    cr = df.get("total_credit_amt_12m", df.get("total_monthly_credit", pd.Series(0, index=df.index))).fillna(0)
    db = df.get("total_debit_amt_12m",  df.get("total_monthly_debit",  pd.Series(0, index=df.index))).fillna(0)
    net_flow_ratio = np.where(cr > 0, (cr - db) / cr, -1.0)
    flow_score = np.clip(net_flow_ratio * 60 + 50, 0, 40)
    s = s * 0 + flow_score

    if "avg_monthly_balance" in df.columns or "total_avg_monthly_balance" in df.columns:
        bal = df.get("avg_monthly_balance", df.get("total_avg_monthly_balance", pd.Series(0, index=df.index))).fillna(0)
        inc = df.get("total_monthly_income", df.get("total_monthly_credit", pd.Series(1, index=df.index))).fillna(1).clip(lower=1)
        bal_ratio = (bal / inc).clip(0, 5)
        bal_score = np.minimum(40.0, bal_ratio * 20)
        s += bal_score

    if "nsf_count_total" in df.columns:
        nsf = df["nsf_count_total"].fillna(0)
        s -= (nsf * 5).clip(0, 25)
    if "returned_cheques_total" in df.columns:
        s -= (df["returned_cheques_total"].fillna(0) * 8).clip(0, 20)

    if "savings_ratio" in df.columns:
        sv = df["savings_ratio"].fillna(0).clip(0, 1)
        s += (sv * 15).round(2)

    if "overdraft_utilization_rate" in df.columns:
        od = df["overdraft_utilization_rate"].fillna(0).clip(0, 1)
        s -= (od * 15).round(2)

    if "cash_flow_surplus_pct" in df.columns:
        surplus = df["cash_flow_surplus_pct"].fillna(0)
        s += np.where(surplus > 20, 5.0, np.where(surplus > 10, 2.0, 0.0))

    return s.clip(0, 100).round(2)
