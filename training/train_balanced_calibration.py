import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from scoring.pillars import (
    p01_identity_trust, p02_fraud_risk, p03_income, p04_cash_flow,
    p05_affordability, p06_stability, p07_behavioral, p08_digital_footprint,
    p09_bureau_proxy, p10_relationship, p11_collection_risk,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FEATURE_STORE = BASE_DIR / "feature_store" / "customer_features_v2.parquet"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_balanced_calibration():
    n_cpu = os.cpu_count() or 8
    logging.info(f"Loading feature store v2 from {FEATURE_STORE}...")
    df = pd.read_parquet(FEATURE_STORE)

    logging.info("Computing 11 sub-scores across all 250,000 profiles...")
    sub_scores = pd.DataFrame(index=df.index)
    sub_scores["p01"] = p01_identity_trust.score(df)
    sub_scores["p02"] = p02_fraud_risk.score(df)
    sub_scores["p03"] = p03_income.score(df)
    sub_scores["p04"] = p04_cash_flow.score(df)
    sub_scores["p05"] = p05_affordability.score(df)
    sub_scores["p06"] = p06_stability.score(df)
    sub_scores["p07"] = p07_behavioral.score(df)
    sub_scores["p08"] = p08_digital_footprint.score(df)
    sub_scores["p09"] = p09_bureau_proxy.score(df)
    sub_scores["p10"] = p10_relationship.score(df)
    sub_scores["p11"] = p11_collection_risk.score(df)

    raw_features = pd.DataFrame(index=df.index)
    raw_features["age"] = df["age"].fillna(35.0)
    raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0)
    raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0)
    raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0)
    raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0)
    raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0)

    X = pd.concat([sub_scores, raw_features], axis=1)

    # Clean default label without synthetic noise injection
    logging.info("Deriving pure ground-truth default targets...")
    default_logic = (
        (df["ldr_max_dpd_ever"].fillna(0) > 45) | 
        (df["ldr_avg_dpd_last_12m"].fillna(0) > 10) | 
        (df["has_fraud_flag"].fillna(False) == True) | 
        (df["any_aml_flag"].fillna(False) == True) |
        (df["nsf_count_total"].fillna(0) > 8)
    )
    y = default_logic.astype(int).values

    logging.info(f"Target distribution: {y.mean()*100:.2f}% Default Rate ({y.sum():,} defaulters / {len(y):,} total)")

    mono_constraints = [-1]*11 + [-1, -1, 1, 1, -1, 1]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            monotone_constraints=mono_constraints,
            n_jobs=n_cpu,
            random_state=42 + fold,
            verbose=-1
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_va)[:, 1]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            monotone_constraints=tuple(mono_constraints),
            n_jobs=n_cpu,
            random_state=42 + fold,
            eval_metric="logloss"
        )
        clf_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_va)[:, 1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "pd_scaler.joblib")

    # Fit final base models
    final_lgb = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.025,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=mono_constraints,
        n_jobs=n_cpu,
        random_state=42,
        verbose=-1
    )
    final_lgb.fit(X, y)

    final_xgb = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=tuple(mono_constraints),
        n_jobs=n_cpu,
        random_state=42,
        eval_metric="logloss"
    )
    final_xgb.fit(X, y)

    final_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    final_lr.fit(X_scaled, y)
    oof_lr = final_lr.predict_proba(X_scaled)[:, 1]

    # Stacking meta-learner
    X_meta = np.column_stack([oof_lgb, oof_xgb, oof_lr])
    meta_learner = LogisticRegression(C=1.0, random_state=42)
    meta_learner.fit(X_meta, y)

    joblib.dump(final_lgb, MODELS_DIR / "pd_lgb.joblib")
    joblib.dump(final_xgb, MODELS_DIR / "pd_xgb.joblib")
    joblib.dump(final_lr, MODELS_DIR / "pd_logistic.joblib")
    joblib.dump(meta_learner, MODELS_DIR / "pd_meta_learner.joblib")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "pd_ensemble_features.joblib")

    logging.info("Balanced models successfully persisted to models/")

if __name__ == "__main__":
    train_balanced_calibration()
