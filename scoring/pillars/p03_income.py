"""P03 — Income Pillar Scorer (F6, F12) | Weight: 12%"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

WEIGHT = 0.12
PILLAR_NAME = "Income"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# ── PKR income percentiles for scoring (approximate Pakistan distribution) ──
PKR_PERCENTILE_MAP = [
    (0,      25000,  10),
    (25000,  40000,  20),
    (40000,  60000,  35),
    (60000,  100000, 50),
    (100000, 150000, 65),
    (150000, 250000, 78),
    (250000, 400000, 88),
    (400000, float("inf"), 97),
]

def _income_percentile_score(income_val: float) -> float:
    for lo, hi, sc in PKR_PERCENTILE_MAP:
        if lo <= income_val < hi:
            return float(sc)
    return 10.0


def score(df: pd.DataFrame) -> pd.Series:
    """
    Income pillar score (0-100). Uses pre-predicted P10/P50/P90 columns
    if available (from the batch scoring pipeline that pre-runs model inference),
    otherwise falls back to declared income columns.
    """
    scores = pd.Series(50.0, index=df.index)

    # ── Determine income to use ─────────────────────────────────────────────
    if "p50_income" in df.columns:
        income_mid = df["p50_income"].fillna(0)
        income_low = df.get("p10_income", income_mid).fillna(0)
        income_high = df.get("p90_income", income_mid).fillna(0)
    elif "total_monthly_income" in df.columns:
        income_mid = df["total_monthly_income"].fillna(0)
        income_low  = income_mid * 0.80
        income_high = income_mid * 1.25
    elif "net_monthly_salary" in df.columns:
        income_mid  = df["net_monthly_salary"].fillna(0)
        income_low  = income_mid * 0.80
        income_high = income_mid * 1.25
    else:
        return scores  # cannot score

    # ── Base score from P50 income level ────────────────────────────────────
    base = income_mid.apply(_income_percentile_score)
    scores = base.copy()

    # ── Income bandwidth penalty (wide band = uncertainty) ───────────────────
    bandwidth = (income_high - income_low).clip(lower=0)
    bw_pct = np.where(income_mid > 0, bandwidth / income_mid.clip(lower=1) * 100, 50)
    # >80% bandwidth = high uncertainty penalty
    bw_penalty = np.where(bw_pct > 80, 12, np.where(bw_pct > 50, 7, np.where(bw_pct > 30, 3, 0)))
    scores -= bw_penalty

    # ── Income confidence / stability bonus ─────────────────────────────────
    if "income_confidence_score" in df.columns:
        conf = df["income_confidence_score"].fillna(50) / 100
        scores += (conf * 8).round(2)
    if "income_stability_score" in df.columns:
        stab = df["income_stability_score"].fillna(50) / 100
        scores += (stab * 5).round(2)

    # ── Verified income bonus ────────────────────────────────────────────────
    if "verified_income" in df.columns and "salary_verified_bank" in df.columns:
        verified = df["salary_verified_bank"].fillna(False).astype(bool)
        scores += np.where(verified, 5.0, 0.0)

    return scores.clip(0, 100).round(2)
