"""
Enterprise Credit AI Engine
Feature Store Aggregation Pipeline

Processes raw datasets (Customers, Employment, Income, Bank Accounts, Transactions)
and aggregates customer-level ML features. Exports a compressed Snappy Parquet file
to `feature_store/customer_features.parquet` for LightGBM Quantile Regression.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Base Path Config
BASE_DIR = Path(__file__).resolve().parent.parent

if (BASE_DIR / "data" / "raw").exists():
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
else:
    DATA_RAW_DIR = BASE_DIR / "datasets" / "raw"

FEATURE_STORE_DIR = BASE_DIR / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

# 5 Core Input Raw Files
CUSTOMER_FILE = DATA_RAW_DIR / "customers.csv"
EMPLOYMENT_FILE = DATA_RAW_DIR / "employment.csv"
INCOME_FILE = DATA_RAW_DIR / "income.csv"
BANKING_FILE = DATA_RAW_DIR / "bank_accounts.csv"
TRANSACTIONS_FILE = DATA_RAW_DIR / "transactions.csv"

# Output Parquet File
CUSTOMER_FEATURES_PARQUET = FEATURE_STORE_DIR / "customer_features.parquet"

def normalize_cust_id(raw_id):
    s = str(raw_id).strip()
    if s.startswith("CUST"):
        return s
    try:
        return f"CUST{int(s):07d}"
    except ValueError:
        return s

# -----------------------------------------------------------------------------
# 1. Transaction Feature Aggregation
# -----------------------------------------------------------------------------

def aggregate_transaction_features():
    logging.info("Streaming and aggregating 14.8M+ transaction records...")
    
    customer_tx_metrics = {}
    chunk_size = 500000
    total_tx_count = 0

    for chunk in tqdm(pd.read_csv(TRANSACTIONS_FILE, chunksize=chunk_size), desc="Processing Transactions"):
        chunk["customer_id"] = chunk["customer_id"].apply(normalize_cust_id)
        chunk["amount"] = chunk["amount"].astype(float)
        
        grouped = chunk.groupby(["customer_id", "transaction_type", "category"])["amount"].agg(["sum", "count"]).reset_index()

        for _, row in grouped.iterrows():
            cid = row["customer_id"]
            tx_type = row["transaction_type"]
            cat = row["category"]
            amt_sum = row["sum"]
            cnt = row["count"]

            if cid not in customer_tx_metrics:
                customer_tx_metrics[cid] = {
                    "total_credit_amt_12m": 0.0,
                    "total_debit_amt_12m": 0.0,
                    "credit_count_12m": 0,
                    "debit_count_12m": 0,
                    "salary_credit_amt_12m": 0.0,
                    "salary_credit_count_12m": 0,
                    "utility_debit_amt_12m": 0.0,
                    "pos_debit_amt_12m": 0.0,
                    "atm_debit_amt_12m": 0.0,
                    "fee_debit_amt_12m": 0.0
                }

            metrics = customer_tx_metrics[cid]
            
            if tx_type == "Credit":
                metrics["total_credit_amt_12m"] += amt_sum
                metrics["credit_count_12m"] += cnt
                if cat == "Salary Credit":
                    metrics["salary_credit_amt_12m"] += amt_sum
                    metrics["salary_credit_count_12m"] += cnt
            elif tx_type == "Debit":
                metrics["total_debit_amt_12m"] += amt_sum
                metrics["debit_count_12m"] += cnt
                if cat == "Utility Payment":
                    metrics["utility_debit_amt_12m"] += amt_sum
                elif cat == "POS Merchant":
                    metrics["pos_debit_amt_12m"] += amt_sum
                elif cat == "Cash Withdrawal":
                    metrics["atm_debit_amt_12m"] += amt_sum
                elif cat == "Bank Fee":
                    metrics["fee_debit_amt_12m"] += amt_sum

        total_tx_count += len(chunk)

    logging.info(f"Successfully processed {total_tx_count:,} transactions into customer metrics.")
    
    df_tx_features = pd.DataFrame.from_dict(customer_tx_metrics, orient="index")
    df_tx_features.index.name = "customer_id"
    df_tx_features.reset_index(inplace=True)
    return df_tx_features

# -----------------------------------------------------------------------------
# 2. Banking Feature Aggregation
# -----------------------------------------------------------------------------

def aggregate_banking_features():
    logging.info(f"Loading bank accounts dataset from {BANKING_FILE}...")
    df_bank = pd.read_csv(BANKING_FILE)
    df_bank["customer_id"] = df_bank["customer_id"].apply(normalize_cust_id)

    bank_agg = df_bank.groupby("customer_id").agg(
        total_bank_accounts=("account_id", "count"),
        active_bank_accounts=("account_status", lambda s: (s == "Active").sum()),
        dormant_bank_accounts=("account_status", lambda s: (s == "Dormant").sum()),
        total_available_balance=("available_balance", "sum"),
        total_ledger_balance=("ledger_balance", "sum"),
        total_avg_monthly_balance=("average_monthly_balance", "sum"),
        max_account_age_months=("account_age_months", "max"),
        has_priority_banking=("priority_banking", "any"),
        has_salary_account=("is_salary_account", "any"),
        has_credit_card_linked=("linked_credit_card", "any"),
        has_loan_linked=("linked_loan", "any"),
        total_overdraft_limit=("overdraft_limit", "sum"),
        total_overdraft_used=("overdraft_used", "sum")
    ).reset_index()

    return bank_agg

# -----------------------------------------------------------------------------
# 3. Build Core Feature Matrix
# -----------------------------------------------------------------------------

def build_feature_store():
    logging.info("Loading 5 core raw datasets...")
    df_cust = pd.read_csv(CUSTOMER_FILE)
    df_emp = pd.read_csv(EMPLOYMENT_FILE)
    df_inc = pd.read_csv(INCOME_FILE)

    df_cust["customer_id"] = df_cust["customer_id"].apply(normalize_cust_id)
    df_emp["customer_id"] = df_emp["customer_id"].apply(normalize_cust_id)
    df_inc["customer_id"] = df_inc["customer_id"].apply(normalize_cust_id)

    # Demographic Features (Safe Column Selection)
    cust_cols = [c for c in ["customer_id", "age", "gender", "education", "marital_status", 
                            "city", "province", "residential_status", "years_at_address", 
                            "sim_age_days", "email_age_days"] if c in df_cust.columns]
    df_features = df_cust[cust_cols].copy()

    # Employment Features
    emp_cols = [c for c in ["customer_id", "employment_status", "employment_type", "industry", 
                           "company_size", "years_experience", "current_job_tenure", 
                           "previous_jobs_count", "working_hours_per_week", "manager_level", 
                           "is_salary_verified"] if c in df_emp.columns]
    df_features = df_features.merge(df_emp[emp_cols], on="customer_id", how="left")

    # Income Features (Target & Benchmarks)
    inc_cols = [c for c in ["customer_id", "gross_monthly_salary", "net_monthly_salary", 
                           "basic_salary", "total_monthly_income", "verified_income", 
                           "income_confidence_score", "income_stability_score", 
                           "income_source_count", "tax_amount", "salary_variability"] if c in df_inc.columns]
    df_features = df_features.merge(df_inc[inc_cols], on="customer_id", how="left")

    # Bank Account Features
    df_bank_agg = aggregate_banking_features()
    df_features = df_features.merge(df_bank_agg, on="customer_id", how="left")

    # Transaction Features
    df_tx_agg = aggregate_transaction_features()
    df_features = df_features.merge(df_tx_agg, on="customer_id", how="left")

    tx_cols = [
        "total_credit_amt_12m", "total_debit_amt_12m", "credit_count_12m", 
        "debit_count_12m", "salary_credit_amt_12m", "salary_credit_count_12m", 
        "utility_debit_amt_12m", "pos_debit_amt_12m", "atm_debit_amt_12m", "fee_debit_amt_12m"
    ]
    df_features[tx_cols] = df_features[tx_cols].fillna(0.0)

    # 4. Computed Ratios & Signals
    logging.info("Computing derived ML signals...")

    df_features["avg_monthly_credit_flow"] = (df_features["total_credit_amt_12m"] / 12.0).round(2)
    df_features["avg_monthly_debit_flow"] = (df_features["total_debit_amt_12m"] / 12.0).round(2)
    df_features["net_monthly_cash_flow"] = (df_features["avg_monthly_credit_flow"] - df_features["avg_monthly_debit_flow"]).round(2)

    df_features["credit_to_debit_ratio"] = np.where(
        df_features["avg_monthly_debit_flow"] > 0,
        (df_features["avg_monthly_credit_flow"] / df_features["avg_monthly_debit_flow"]).round(3),
        1.0
    )

    df_features["cash_withdrawal_to_income_ratio"] = np.where(
        df_features["total_monthly_income"] > 0,
        ((df_features["atm_debit_amt_12m"] / 12.0) / df_features["total_monthly_income"]).round(3),
        0.0
    )

    df_features["balance_to_income_ratio"] = np.where(
        df_features["total_monthly_income"] > 0,
        (df_features["total_available_balance"] / df_features["total_monthly_income"]).round(3),
        0.0
    )

    # Convert boolean columns to integers
    bool_cols = df_features.select_dtypes(include=["bool"]).columns
    df_features[bool_cols] = df_features[bool_cols].astype(int)

    return df_features

def save_to_feature_store(df_features):
    logging.info(f"Saving feature matrix ({len(df_features):,} rows, {len(df_features.columns)} columns) to Parquet...")
    
    df_features.to_parquet(
        CUSTOMER_FEATURES_PARQUET,
        engine="pyarrow",
        compression="snappy",
        index=False
    )
    
    size_mb = CUSTOMER_FEATURES_PARQUET.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*60)
    print("        FEATURE STORE PIPELINE COMPLETE")
    print("="*60)
    print(f"Total Customer Entities : {len(df_features):,}")
    print(f"Total ML Features       : {len(df_features.columns)}")
    print(f"Feature Store Path      : {CUSTOMER_FEATURES_PARQUET}")
    print(f"Parquet Storage Size    : {size_mb:.2f} MB")
    print("="*60)

def main():
    df_features = build_feature_store()
    save_to_feature_store(df_features)

if __name__ == "__main__":
    main()