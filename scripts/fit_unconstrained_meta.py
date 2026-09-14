import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODELS_DIR = Path("models")
FEATURE_STORE = Path("feature_store/customer_features_v2.parquet")

df = pd.read_parquet(FEATURE_STORE)
sub_df = pd.read_parquet("data/processed/credit_scores_250k.parquet")

feats = joblib.load(MODELS_DIR / "pd_ensemble_features.joblib")
m_lgb = joblib.load(MODELS_DIR / "pd_lgb.joblib")
m_xgb = joblib.load(MODELS_DIR / "pd_xgb.joblib")
m_lr = joblib.load(MODELS_DIR / "pd_logistic.joblib")
scaler = joblib.load(MODELS_DIR / "pd_scaler.joblib")

raw_features = pd.DataFrame(index=df.index)
raw_features["age"] = df["age"].fillna(35.0)
raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0)
raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0)
raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0)
raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0)
raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0)

sub_cols = {f"p{i:02d}": sub_df[f"p{i:02d}_score"] for i in range(1, 12)}
X = pd.concat([pd.DataFrame(sub_cols, index=df.index), raw_features], axis=1)[feats]

default_logic = (
    (df["ldr_max_dpd_ever"].fillna(0) > 45) | 
    (df["ldr_avg_dpd_last_12m"].fillna(0) > 10) | 
    (df["has_fraud_flag"].fillna(False) == True) | 
    (df["any_aml_flag"].fillna(False) == True) |
    (df["nsf_count_total"].fillna(0) > 8)
)
y = default_logic.astype(int).values

p_lgb = m_lgb.predict_proba(X)[:, 1]
p_xgb = m_xgb.predict_proba(X)[:, 1]
p_lr = m_lr.predict_proba(scaler.transform(X))[:, 1]

# Stacking without intercept shift or with calibrated unconstrained weights
X_meta = np.column_stack([p_lgb, p_xgb, p_lr])

# Non-negative weights stacking with intercept fit
meta = LogisticRegression(C=1.0, fit_intercept=True, random_state=42)
meta.fit(X_meta, y)

ens_pd = meta.predict_proba(X_meta)[:, 1]
odds = (1.0 - ens_pd) / ens_pd
scores = np.clip(503.727 + 57.7078 * np.log(odds), 300, 900).round().astype(int)

print("Calibrated Stacking Meta Fit:")
print(f"  Intercept: {meta.intercept_[0]:.4f}")
print(f"  Coefficients: LGB={meta.coef_[0][0]:.4f}, XGB={meta.coef_[0][1]:.4f}, LR={meta.coef_[0][2]:.4f}")
print(f"  Min PD: {ens_pd.min()*100:.3f}% | Max Score: {scores.max()}")
print(f"  Grade A (Score >= 780, PD < 1.5%): {(scores >= 780).sum():,} customers")
print(f"  Grade B (720 - 779, PD 1.5-3.0%): {((scores >= 720) & (scores < 780)).sum():,} customers")
