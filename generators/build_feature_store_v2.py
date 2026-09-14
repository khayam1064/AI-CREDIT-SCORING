"""
Enterprise Credit AI Engine
Extended Feature Store Builder (v2)

Merges all 9 raw datasets and produces a unified customer-level feature
matrix covering all 15 data families. Output feeds the 12-pillar scorer.

Input CSVs  (datasets/raw/ or data/raw/):
  customers.csv, employment.csv, income.csv, bank_accounts.csv,
  transactions.csv, device_behavioral.csv, telecom_signals.csv,
  wallets.csv, loan_ledger.csv, application_meta.csv

Output:
  feature_store/customer_features_v2.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
if (BASE_DIR / "data" / "raw").exists():
    RAW_DIR = BASE_DIR / "data" / "raw"
else:
    RAW_DIR = BASE_DIR / "datasets" / "raw"

FEATURE_STORE_DIR = BASE_DIR / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET = FEATURE_STORE_DIR / "customer_features_v2.parquet"

TODAY_YR, TODAY_MO = 2026, 8


def normalize_id(s):
    s = str(s).strip()
    if s.startswith("CUST"):
        return s
    try:
        return f"CUST{int(s):07d}"
    except ValueError:
        return s


def load(filename, required=True):
    p = RAW_DIR / filename
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Required file missing: {p}")
        logging.warning(f"Optional file not found, skipping: {p}")
        return None
    logging.info(f"Loading {filename} ...")
    df = pd.read_csv(p, low_memory=False)
    df["customer_id"] = df["customer_id"].apply(normalize_id)
    return df


def build_feature_store():
    # ── 1. Core datasets ─────────────────────────────────────────────────────
    df_cust = load("customers.csv")
    df_emp  = load("employment.csv",  required=False)
    df_inc  = load("income.csv",      required=False)

    # Start with customers as spine
    df = df_cust.copy()
    if df_emp is not None:
        df = df.merge(df_emp, on="customer_id", how="left", suffixes=("", "_emp"))
    if df_inc is not None:
        df = df.merge(df_inc, on="customer_id", how="left", suffixes=("", "_inc"))

    # ── 2. Banking aggregates ─────────────────────────────────────────────────
    df_bank = load("bank_accounts.csv", required=False)
    if df_bank is not None:
        # Primary account only for single-row-per-customer join
        bank_agg = df_bank.groupby("customer_id").agg(
            bank_account_count=("account_id", "count"),
            primary_bank=("bank_name", "first"),
            total_avg_monthly_balance=("average_monthly_balance", "sum"),
            max_overdraft_limit=("overdraft_limit", "sum"),
            total_overdraft_used=("overdraft_used", "sum"),
            nsf_count_total=("nsf_count", "sum"),
            returned_cheques_total=("returned_cheques", "sum"),
            any_fraud_flag=("fraud_flag", "max"),
            any_aml_flag=("aml_flag", "max"),
            has_mobile_banking=("mobile_banking", "max"),
            has_internet_banking=("internet_banking", "max"),
            salary_verified_bank=("salary_verified", "max"),
            total_monthly_credit=("monthly_credit_amount", "sum"),
            total_monthly_debit=("monthly_debit_amount", "sum"),
            max_account_age_months=("account_age_months", "max"),
        ).reset_index()
        bank_agg["overdraft_utilization_rate"] = np.where(
            bank_agg["max_overdraft_limit"] > 0,
            bank_agg["total_overdraft_used"] / bank_agg["max_overdraft_limit"],
            0.0
        )
        bank_agg["net_monthly_cashflow"] = (
            bank_agg["total_monthly_credit"] - bank_agg["total_monthly_debit"]
        )
        df = df.merge(bank_agg, on="customer_id", how="left")

    # ── 3. Transaction aggregates ─────────────────────────────────────────────
    # Read existing feature store parquet if transactions.csv is huge
    existing_fs = FEATURE_STORE_DIR / "customer_features.parquet"
    if existing_fs.exists():
        logging.info("Merging existing feature store (income/cash-flow aggregates)...")
        df_existing = pd.read_parquet(existing_fs)
        df_existing["customer_id"] = df_existing["customer_id"].apply(normalize_id)
        # Take only the transaction-derived columns not already in df
        tx_cols = [c for c in df_existing.columns if c not in df.columns or c == "customer_id"]
        df = df.merge(df_existing[tx_cols], on="customer_id", how="left")

    # ── 4. New Family datasets ────────────────────────────────────────────────
    for fname, prefix in [
        ("device_behavioral.csv", "dev_"),
        ("telecom_signals.csv",   "tel_"),
        ("wallets.csv",           "wal_"),
        ("loan_ledger.csv",       "ldr_"),
        ("application_meta.csv",  "app_"),
    ]:
        df_aux = load(fname, required=False)
        if df_aux is None:
            continue
        rename = {c: f"{prefix}{c}" for c in df_aux.columns if c != "customer_id"}
        df_aux = df_aux.rename(columns=rename)
        df = df.merge(df_aux, on="customer_id", how="left")

    # ── 5. Derived / Engineered Features ─────────────────────────────────────
    logging.info("Engineering derived features...")
    df = df.copy()  # Defragment to avoid PerformanceWarning on insert

    # FOIR proxy
    total_inc = df.get("total_monthly_income", pd.Series(dtype=float))
    if total_inc is None:
        total_inc = df.get("total_monthly_credit", pd.Series(0.0, index=df.index))
    utility_debit = df.get("utility_debit_amt_12m", pd.Series(0.0, index=df.index)).fillna(0) / 12
    foir_pct = np.where(total_inc.fillna(0) > 0, utility_debit / total_inc.fillna(1) * 100, 0)
    df["foir_pct"] = np.round(foir_pct, 2)

    # Cash flow surplus %
    cr12 = df.get("total_credit_amt_12m", df.get("total_monthly_credit", pd.Series(0.0, index=df.index))).fillna(0)
    db12 = df.get("total_debit_amt_12m",  df.get("total_monthly_debit",  pd.Series(0.0, index=df.index))).fillna(0)
    df["cash_flow_surplus_pct"] = np.where(cr12 > 0, (cr12 - db12) / cr12 * 100, 0.0).round(2)

    # Digital maturity index (0-100)
    email_age = df.get("email_age_days", pd.Series(0, index=df.index)).fillna(0)
    sim_age   = df.get("sim_age_days",   pd.Series(0, index=df.index)).fillna(0)
    df["digital_maturity_index"] = np.round(
        (np.minimum(email_age / 3650, 1.0) * 50)
        + (np.minimum(sim_age  / 1825, 1.0) * 50),
        1
    )

    # Income declared vs predicted divergence (fraud signal)
    declared = df.get("total_monthly_income", pd.Series(dtype=float))
    if declared is not None and "verified_income" in df.columns:
        df["income_divergence_pct"] = np.where(
            declared.fillna(0) > 0,
            np.abs(df["verified_income"].fillna(0) - declared.fillna(0)) / declared.fillna(1) * 100,
            0.0
        ).round(2)
    else:
        df["income_divergence_pct"] = 0.0

    # Months as customer (for relationship weight scaling)
    if "customer_since" in df.columns:
        cs = pd.to_datetime(df["customer_since"], errors="coerce")
        df["months_as_customer"] = (
            (TODAY_YR - cs.dt.year) * 12 + (TODAY_MO - cs.dt.month)
        ).clip(lower=0).fillna(0).astype(int)
    else:
        df["months_as_customer"] = 0

    # Savings ratio proxy
    df["savings_ratio"] = np.where(cr12 > 0, (cr12 - db12) / cr12, 0.0).clip(0, 1).round(4)

    logging.info(f"Feature store v2 shape: {df.shape}")
    df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    logging.info(f"Saved → {OUTPUT_PARQUET}")

    print(f"\n{'='*60}")
    print(f"  FEATURE STORE V2 — BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"Total Customers    : {len(df):,}")
    print(f"Total Features     : {df.shape[1]}")
    print(f"Output             : {OUTPUT_PARQUET}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    build_feature_store()
