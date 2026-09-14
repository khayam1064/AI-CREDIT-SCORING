# """
# Enterprise Credit AI Engine
# Safe Risk & Declined Applicants Finder
# """

# from pathlib import Path
# import pandas as pd

# BASE_DIR = Path(__file__).resolve().parent
# FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"

# print(f"Loading feature store dataset from {FEATURE_STORE_PARQUET}...")
# df = pd.read_parquet(FEATURE_STORE_PARQUET)

# print(f"Total Customer Records Loaded: {len(df):,}")
# print(f"Available Features ({len(df.columns)}): {list(df.columns[:10])}...\n")

# # 1. Low Income / Cash Flow Risk (< 25,000 PKR / month)
# low_income_df = df[df["total_monthly_income"] < 25000]
# print(f"1. Low Income Applicants (< 25k PKR/mo): {len(low_income_df):,}")

# # 2. Low Stability Score (< 40%)
# if "income_stability_score" in df.columns:
#     low_stability_df = df[df["income_stability_score"] < 40.0]
#     print(f"2. Low Income Stability Score (< 40%): {len(low_stability_df):,}")

# # 3. High Cash Withdrawal / Low Digital Footprint Ratio
# if "cash_withdrawal_to_income_ratio" in df.columns:
#     high_cash_df = df[df["cash_withdrawal_to_income_ratio"] > 0.80]
#     print(f"3. High Cash Dependence (> 80% ATM Ratio): {len(high_cash_df):,}")

# # 4. Low / Zero Balance Buffer
# if "balance_to_income_ratio" in df.columns:
#     zero_buffer_df = df[df["balance_to_income_ratio"] < 0.05]
#     print(f"4. Zero Liquid Buffer Ratio (< 5% Income): {len(zero_buffer_df):,}")

# # Display first 5 customer IDs with low income / high risk
# sample_declined = low_income_df["customer_id"].head(10).tolist()
# print("\nSample High-Risk Customer IDs for Testing Rejections/Declines:")
# for cid in sample_declined:
#     print(f" - {cid}")

"""
Enterprise Credit AI Engine
Flexible Declined Employed Customer Finder
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"
MODELS_DIR = BASE_DIR / "models"

print("Loading Feature Store & Trained LightGBM Artifacts...")

df = pd.read_parquet(FEATURE_STORE_PARQUET)
feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
model_p10 = joblib.load(MODELS_DIR / "lgb_income_p10.joblib")

# 1. Normalize Customer IDs
def normalize_cust_id(raw_id) -> str:
    s = str(raw_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("CUST"):
        try:
            return f"CUST{int(s[4:]):07d}"
        except ValueError:
            return s
    try:
        return f"CUST{int(s):07d}"
    except ValueError:
        return s

if "customer_id" in df.columns:
    df["customer_id"] = df["customer_id"].apply(normalize_cust_id)

# 2. Filter for Employed Applicants
df_employed = df[df["employment_status"].astype(str) == "Employed"].copy()
print(f"Employed Records Selected: {len(df_employed):,}")

# 3. Model Inference (P10 Conservative Income)
X = df_employed[feature_cols].copy()
cat_cols = X.select_dtypes(include=["object", "string"]).columns
for c in cat_cols:
    X[c] = X[c].astype("category")

df_employed["predicted_p10_income"] = model_p10.predict(X)

# 4. Derive Realistic Monthly Obligations
# Fallback to total monthly debit flow if utility_debit_amt_12m is zero
if "avg_monthly_debit_flow" in df_employed.columns:
    df_employed["monthly_obligations"] = df_employed["avg_monthly_debit_flow"]
elif "total_debit_amt_12m" in df_employed.columns:
    df_employed["monthly_obligations"] = df_employed["total_debit_amt_12m"] / 12.0
else:
    # Use utility debit if present, otherwise approximate fixed debts
    util = df_employed.get("utility_debit_amt_12m", pd.Series(0, index=df_employed.index)).fillna(0) / 12.0
    df_employed["monthly_obligations"] = np.where(util > 0, util, df_employed["total_monthly_income"] * 0.45)

# 5. Underwriting Criteria Evaluation
df_employed["current_foir_percent"] = np.where(
    df_employed["predicted_p10_income"] > 0,
    (df_employed["monthly_obligations"] / df_employed["predicted_p10_income"]) * 100.0,
    100.0
)

df_employed["max_capacity"] = df_employed["predicted_p10_income"] * 0.50
df_employed["available_emi"] = (df_employed["max_capacity"] - df_employed["monthly_obligations"]).clip(lower=0)
df_employed["max_affordable_emi"] = (df_employed["available_emi"] * 0.90).round(2)

df_employed["stability_score"] = df_employed["income_stability_score"] if "income_stability_score" in df_employed.columns else 70.0
df_employed["fraud_flag"] = df_employed["has_fraud_flag"] if "has_fraud_flag" in df_employed.columns else 0

# 6. Filter for DECLINED Employed Applicants
declined_mask = (
    (df_employed["current_foir_percent"] > 50.0) |       # Debt load exceeds 50%
    (df_employed["max_affordable_emi"] < 3000) |        # Net EMI capacity under 3k PKR
    (df_employed["stability_score"] < 40.0) |            # High volatility
    (df_employed["fraud_flag"] == 1)                     # AML / Fraud flag
)

df_declined_employed = df_employed[declined_mask].copy()

def categorize_reason(row):
    if row["fraud_flag"] == 1:
        return "FRAUD_OR_AML_FLAG"
    elif row["current_foir_percent"] > 50.0:
        return f"FOIR > 50% ({row['current_foir_percent']:.1f}%)"
    elif row["max_affordable_emi"] < 3000:
        return f"Low EMI Capacity ({row['max_affordable_emi']:,.0f} PKR)"
    elif row["stability_score"] < 40.0:
        return f"Low Stability ({row['stability_score']:.1f}%)"
    return "DECLINED"

if len(df_declined_employed) > 0:
    df_declined_employed["decline_reason"] = df_declined_employed.apply(categorize_reason, axis=1)

print("\n" + "="*80)
print(f" DECLINED EMPLOYED APPLICANTS FOUND: {len(df_declined_employed):,} / {len(df_employed):,}")
print("="*80 + "\n")

cols_to_print = [
    "customer_id", "employment_status", "total_monthly_income", 
    "predicted_p10_income", "current_foir_percent", "max_affordable_emi", "decline_reason"
]
cols_to_print = [c for c in cols_to_print if c in df_declined_employed.columns]

if len(df_declined_employed) > 0:
    print(df_declined_employed[cols_to_print].head(15).to_string(index=False))

    print("\n" + "-"*80)
    print("Customer IDs to copy-paste into Streamlit Dashboard / FastAPI testing:")
    print("-"*80)
    for cid in df_declined_employed["customer_id"].head(15).tolist():
        print(f" -> {cid}")