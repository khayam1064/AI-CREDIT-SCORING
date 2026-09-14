"""
Enterprise Credit AI Engine
Quantile LightGBM Income Prediction Trainer (P10, P50, P90)

Trains 3 separate LightGBM Quantile Regressors on feature_store/customer_features.parquet:
- Alpha 0.10 (P10): Conservative lower-bound income (Underwriting safety limit)
- Alpha 0.50 (P50): Expected median income (Core affordability baseline)
- Alpha 0.90 (P90): Optimistic upper-bound income (Upsell / limit cap)
"""

import os
import joblib
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "total_monthly_income"

EXCLUDE_COLUMNS = [
    "customer_id", "total_monthly_income", "gross_monthly_salary", 
    "net_monthly_salary", "basic_salary", "verified_income"
]

def pinball_loss(y_true, y_pred, alpha):
    err = y_true - y_pred
    return np.mean(np.maximum(alpha * err, (alpha - 1) * err))

def train_quantile_pipeline():
    logging.info(f"Loading feature store dataset from {FEATURE_STORE_PARQUET}...")
    if not FEATURE_STORE_PARQUET.exists():
        raise FileNotFoundError(f"Feature store not found at {FEATURE_STORE_PARQUET}.")

    df = pd.read_parquet(FEATURE_STORE_PARQUET)
    
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
 
    # Pandas string/object type handling for LightGBM
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    # Store categorical feature information for consistent inference
    categorical_info = {
        "categorical_columns": cat_cols,
        "categorical_dtypes": {col: str(X[col].dtype) for col in cat_cols}
    }
    
    for col in cat_cols:
        X[col] = X[col].astype("category")

    logging.info(f"Feature matrix shape: {X.shape} ({len(feature_cols)} features, {len(cat_cols)} categorical)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    quantiles = [0.10, 0.50, 0.90]
    quantile_names = {0.10: "P10", 0.50: "P50", 0.90: "P90"}
    metrics = {}

    for alpha in quantiles:
        q_name = quantile_names[alpha]
        logging.info(f"Training LightGBM Quantile Regressor for {q_name} (Alpha = {alpha})...")

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        preds = model.predict(X_test)
        p_loss = pinball_loss(y_test, preds, alpha)
        mae = mean_absolute_error(y_test, preds)

        metrics[q_name] = {
            "alpha": alpha,
            "pinball_loss": p_loss,
            "mae": mae
        }

        model_path = MODELS_DIR / f"lgb_income_{q_name.lower()}.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Saved {q_name} model to {model_path}")

    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(categorical_info, MODELS_DIR / "categorical_info.joblib")
    logging.info(f"Saved categorical feature information to {MODELS_DIR / 'categorical_info.joblib'}")

    print("\n" + "="*60)
    print("      QUANTILE LIGHTGBM MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Training Records  : {len(X_train):,}")
    print(f"Testing Records   : {len(X_test):,}")
    print(f"Features Trained  : {len(feature_cols)}")
    print("-"*60)
    for q_name, m in metrics.items():
        print(f"[{q_name}] Alpha {m['alpha']:.2f} | Pinball Loss: {m['pinball_loss']:.2f} PKR | MAE: {m['mae']:.2f} PKR")
    print("="*60)

if __name__ == "__main__":
    train_quantile_pipeline()