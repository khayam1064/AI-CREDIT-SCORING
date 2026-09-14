import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE = BASE_DIR / "feature_store" / "customer_features_v2.parquet"

df = pd.read_parquet(FEATURE_STORE)

# Check predictions of base models on a sample of 250k
m_lgb = joblib.load(MODELS_DIR / "pd_lgb.joblib")
m_xgb = joblib.load(MODELS_DIR / "pd_xgb.joblib")
m_lr = joblib.load(MODELS_DIR / "pd_logistic.joblib")
scaler = joblib.load(MODELS_DIR / "pd_scaler.joblib")
feats = joblib.load(MODELS_DIR / "pd_ensemble_features.joblib")

# Read sub scores from scoring
from scoring import composite_scorer
sub_df = pd.read_parquet("data/processed/credit_scores_250k.parquet")

# Build X
raw_features = pd.DataFrame(index=df.index)
raw_features["age"] = df["age"].fillna(35.0)
raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0)
raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0)
raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0)
raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0)
raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0)

sub_cols = {f"p{i:02d}": sub_df[f"p{i:02d}_score"] for i in range(1, 12)}
X = pd.concat([pd.DataFrame(sub_cols, index=df.index), raw_features], axis=1)[feats]

# Sample 50k
idx = np.random.RandomState(42).choice(len(X), 50000, replace=False)
X_s = X.iloc[idx]

pred_lgb = m_lgb.predict_proba(X_s)[:, 1]
pred_xgb = m_xgb.predict_proba(X_s)[:, 1]
pred_lr = m_lr.predict_proba(scaler.transform(X_s))[:, 1]

print("Base Models Minimum predicted probabilities:")
print(f"  Min LightGBM PD: {pred_lgb.min()*100:.3f}%")
print(f"  Min XGBoost PD:  {pred_xgb.min()*100:.3f}%")
print(f"  Min Logistic PD: {pred_lr.min()*100:.3f}%")
