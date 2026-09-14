"""
Enterprise Credit AI Engine
Batch Underwriting Scanner - Find Declined & High-Risk Applicants
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from engine.underwriting_rules import CreditUnderwritingEngine

# Paths
BASE_DIR = Path(__file__).resolve().parent
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"
MODELS_DIR = BASE_DIR / "models"

print("Loading feature store and models...")
df_features = pd.read_parquet(FEATURE_STORE_PARQUET)
feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")

# Load LightGBM Quantile Models
model_p10 = joblib.load(MODELS_DIR / "lgb_income_p10.joblib")
model_p50 = joblib.load(MODELS_DIR / "lgb_income_p50.joblib")
model_p90 = joblib.load(MODELS_DIR / "lgb_income_p90.joblib")

# Prepare Feature Matrix X
X = df_features[feature_cols].copy()
cat_cols = X.select_dtypes(include=["object", "string"]).columns
for c in cat_cols:
    X[c] = X[c].astype("category")

print("Generating P10, P50, and P90 income predictions for all records...")
preds_p10 = model_p10.predict(X)
preds_p50 = model_p50.predict(X)
preds_p90 = model_p90.predict(X)

# Enforce Monotonic Non-Crossing (P10 <= P50 <= P90)
sorted_preds = np.sort(np.column_stack([preds_p10, preds_p50, preds_p90]), axis=1)

df_features["pred_p10"] = sorted_preds[:, 0]
df_features["pred_p50"] = sorted_preds[:, 1]
df_features["pred_p90"] = sorted_preds[:, 2]

print("Running Underwriting Rules Engine across all records...")
engine = CreditUnderwritingEngine()

results = []
for idx, row in df_features.iterrows():
    cid = row["customer_id"]
    p10 = float(row["pred_p10"])
    p50 = float(row["pred_p50"])
    p90 = float(row["pred_p90"])
    
    # Calculate monthly obligations from utility/loan debits
    obligations = float(row.get("utility_debit_amt_12m", 0.0) / 12.0)
    stability = float(row.get("income_stability_score", 70.0))
    fraud_flag = bool(row.get("has_fraud_flag", False))

    assessment = engine.evaluate_credit(
        customer_id=cid,
        p10_income=p10,
        p50_income=p50,
        p90_income=p90,
        existing_monthly_obligations=obligations,
        income_stability_score=stability,
        has_fraud_flag=fraud_flag
    )
    results.append(assessment)

df_results = pd.DataFrame(results)

# Filter DECLINED or REFER customers
df_declined = df_results[df_results["decision"].isin(["DECLINED", "REFER"])]

print("\n" + "="*60)
print("             UNDERWRITING BATCH RESULTS SUMMARY")
print("="*60)
print(df_results["decision"].value_counts())
print("="*60)

print("\nTop 10 Declined Customer Examples:")
print(df_declined[["customer_id", "decision", "decision_reason", "underwriting_income_p10", "current_foir_percent"]].head(10))

# Save list of unapproved customers to CSV
output_path = BASE_DIR / "declined_customers.csv"
df_declined.to_csv(output_path, index=False)
print(f"\nSaved all unapproved customer records to: {output_path}")