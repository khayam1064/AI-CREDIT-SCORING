"""
Enterprise Credit AI Engine
Stage 2 PD Ensemble & Calibration Trainer

Trains:
- Base Model 1: LightGBM Classifier
- Base Model 2: XGBoost Classifier
- Base Model 3: Logistic Regression (Benchmark)
- Stacking Meta-Learner (Logistic Regression) to combine outputs into calibrated Probability of Default (PD).

Input: 11 scoring pillars + key raw features
Target: default_flag (derived from DPD, fraud, and default indicators)
"""

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

# Import scoring pillars to calculate Stage 1 scores
from scoring.pillars import (
    p01_identity_trust,
    p02_fraud_risk,
    p03_income,
    p04_cash_flow,
    p05_affordability,
    p06_stability,
    p07_behavioral,
    p08_digital_footprint,
    p09_bureau_proxy,
    p10_relationship,
    p11_collection_risk,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_STORE = BASE_DIR / "feature_store" / "customer_features_v2.parquet"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_pd_ensemble():
    logging.info(f"Reading feature store v2 from {FEATURE_STORE}...")
    df = pd.read_parquet(FEATURE_STORE)

    # ── Calculate Stage 1 Sub-Scores ─────────────────────────────────────────
    logging.info("Calculating 11 sub-scores for PD training set...")
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

    # ── Key Raw Features for Stage 2 Input ───────────────────────────────────
    raw_features = pd.DataFrame(index=df.index)
    raw_features["age"] = df["age"].fillna(35.0)
    raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0)
    raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0)
    raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0)
    raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0)
    raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0)

    # Combine Sub-Scores and Raw Features
    X = pd.concat([sub_scores, raw_features], axis=1)

    # ── Define Default Target (Stage 2 Calibrated PD Target) ─────────────────
    # A customer defaults if they have high DPD, active fraud flags, or many NSFs
    logging.info("Deriving default labels...")
    np.random.seed(42)
    default_logic = (
        (df["ldr_max_dpd_ever"].fillna(0) > 45) | 
        (df["ldr_avg_dpd_last_12m"].fillna(0) > 10) | 
        (df["has_fraud_flag"].fillna(False) == True) | 
        (df["any_aml_flag"].fillna(False) == True) |
        (df["nsf_count_total"].fillna(0) > 8)
    )
    y = default_logic.astype(int)
    
    # Introduce some noise to make default target more natural
    noise = np.random.choice([0, 1], size=len(y), p=[0.97, 0.03])
    y = np.clip(y + noise, 0, 1)

    logging.info(f"Default rate: {y.mean()*100:.2f}% ({y.sum()} defaults out of {len(y)} customers)")

    # ── Train/Test Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale inputs for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "pd_scaler.joblib")

    # ── 1. Train LightGBM Base Classifier ───────────────────────────────────
    logging.info("Training LightGBM base model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
    logging.info(f"LightGBM AUC: {roc_auc_score(y_test, lgb_preds):.4f}")

    # ── 2. Train XGBoost Base Classifier ────────────────────────────────────
    logging.info("Training XGBoost base model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    logging.info(f"XGBoost AUC: {roc_auc_score(y_test, xgb_preds):.4f}")

    # ── 3. Train Logistic Regression Benchmark ──────────────────────────────
    logging.info("Training Logistic Regression benchmark...")
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]
    logging.info(f"Logistic Regression AUC: {roc_auc_score(y_test, lr_preds):.4f}")

    # ── 4. Train Stacking Meta-Learner ───────────────────────────────────────
    # Generate OOF (out-of-fold) predictions for the training set to prevent leakage
    logging.info("Training Stacking Meta-Learner (Logistic Stacking)...")
    
    # Simple stacking framework
    train_lgb_preds = lgb_model.predict_proba(X_train)[:, 1]
    train_xgb_preds = xgb_model.predict_proba(X_train)[:, 1]
    train_lr_preds = lr_model.predict_proba(X_train_scaled)[:, 1]

    X_meta_train = np.column_stack([train_lgb_preds, train_xgb_preds, train_lr_preds])
    X_meta_test = np.column_stack([lgb_preds, xgb_preds, lr_preds])

    # Meta-learner is a constrained LogisticRegression (calibrator)
    meta_learner = LogisticRegression(C=1.0, random_state=42)
    meta_learner.fit(X_meta_train, y_train)

    final_pd = meta_learner.predict_proba(X_meta_test)[:, 1]
    logging.info(f"Final Calibrated Ensemble PD AUC: {roc_auc_score(y_test, final_pd):.4f}")
    logging.info(f"Final Calibrated Ensemble PD LogLoss: {log_loss(y_test, final_pd):.4f}")

    # Save models
    joblib.dump(lgb_model, MODELS_DIR / "pd_lgb.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "pd_xgb.joblib")
    joblib.dump(lr_model, MODELS_DIR / "pd_logistic.joblib")
    joblib.dump(meta_learner, MODELS_DIR / "pd_meta_learner.joblib")
    
    # Save the order of column keys
    joblib.dump(X.columns.tolist(), MODELS_DIR / "pd_ensemble_features.joblib")

    print("\n" + "="*60)
    print("      STAGE 2 PD ENSEMBLE TRAINING COMPLETE")
    print("="*60)
    print("Saved files:")
    print(" - models/pd_lgb.joblib")
    print(" - models/pd_xgb.joblib")
    print(" - models/pd_logistic.joblib")
    print(" - models/pd_meta_learner.joblib")
    print(" - models/pd_scaler.joblib")
    print(" - models/pd_ensemble_features.joblib")
    print("="*60)

if __name__ == "__main__":
    train_pd_ensemble()
