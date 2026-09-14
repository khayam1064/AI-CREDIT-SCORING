"""
Enterprise Credit AI Engine
Loan Ledger Generator — Internal Relationship History (Family F13)

Generates synthetic internal loan ledger records for existing customers only
(is_existing_customer = True, ~65% of the 250K dataset).

Output: datasets/raw/loan_ledger.csv
"""

import random
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
if (BASE_DIR / "data" / "raw").exists():
    DATASETS_DIR = BASE_DIR / "data" / "raw"
else:
    DATASETS_DIR = BASE_DIR / "datasets" / "raw"

CUSTOMER_FILE = DATASETS_DIR / "customers.csv"
OUTPUT_FILE = DATASETS_DIR / "loan_ledger.csv"

RANDOM_SEED = 42
TODAY = datetime(2026, 8, 5)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LOAN_STATUS_DIST = {
    "Closed_GoodStanding": 0.55,
    "Active_GoodStanding":  0.25,
    "Closed_EarlyRepaid":   0.08,
    "Delinquent":           0.07,
    "Defaulted":            0.05,
}

def weighted_choice(mapping):
    return random.choices(list(mapping.keys()), weights=list(mapping.values()), k=1)[0]

def generate_loan_ledger(row):
    cid = row["customer_id"]
    is_existing = bool(row.get("is_existing_customer", False))
    customer_since = row.get("customer_since", None)
    segment = str(row.get("customer_segment", "Mass"))

    # Non-existing customers → empty ledger
    if not is_existing or pd.isnull(customer_since):
        return {
            "customer_id": cid,
            "has_internal_history": False,
            "prior_loans_count": 0,
            "active_loans_count": 0,
            "last_loan_amount": 0.0,
            "last_loan_status": None,
            "avg_loan_amount": 0.0,
            "on_time_payment_rate": None,
            "avg_dpd_last_12m": None,
            "max_dpd_ever": None,
            "early_repayments_count": 0,
            "collection_contacts_count": 0,
            "limit_utilization_pct": 0.0,
            "top_up_count": 0,
            "app_engagement_recency_days": None,
            "months_as_customer": 0,
            "relationship_score": 0.0,
        }

    # Months as customer
    cs_dt = pd.to_datetime(customer_since)
    months_as_customer = max(1, (TODAY.year - cs_dt.year) * 12 + (TODAY.month - cs_dt.month))

    # Number of prior loans (more loans for longer tenure)
    max_loans = min(8, max(1, months_as_customer // 6))
    prior_loans_count = random.randint(1, max_loans)
    active_loans_count = random.choices([0, 1, 2], weights=[0.50, 0.38, 0.12], k=1)[0]
    active_loans_count = min(active_loans_count, prior_loans_count)

    # Loan amounts based on segment
    loan_range = {
        "Mass":            (5000, 150000),
        "Affluent":        (50000, 750000),
        "Premium":         (200000, 3000000),
        "Private Banking": (500000, 10000000),
    }.get(segment, (5000, 150000))
    last_loan_amount = round(random.uniform(*loan_range), -2)
    avg_loan_amount = round(last_loan_amount * random.uniform(0.6, 1.2), -2)

    # Loan status
    last_loan_status = weighted_choice(LOAN_STATUS_DIST)

    # Payment performance
    if last_loan_status in ("Closed_GoodStanding", "Closed_EarlyRepaid", "Active_GoodStanding"):
        on_time_payment_rate = round(random.uniform(0.78, 1.00), 3)
        avg_dpd_last_12m = round(random.uniform(0.0, 5.0), 1)
        max_dpd_ever = random.choices([0, 1, 3, 7], weights=[0.55, 0.25, 0.15, 0.05], k=1)[0]
        collection_contacts_count = 0
    elif last_loan_status == "Delinquent":
        on_time_payment_rate = round(random.uniform(0.40, 0.78), 3)
        avg_dpd_last_12m = round(random.uniform(5.0, 45.0), 1)
        max_dpd_ever = random.randint(15, 90)
        collection_contacts_count = random.randint(1, 8)
    else:  # Defaulted
        on_time_payment_rate = round(random.uniform(0.10, 0.45), 3)
        avg_dpd_last_12m = round(random.uniform(30.0, 120.0), 1)
        max_dpd_ever = random.randint(90, 365)
        collection_contacts_count = random.randint(5, 25)

    early_repayments_count = random.choices([0, 1, 2, 3], weights=[0.55, 0.28, 0.13, 0.04], k=1)[0]
    if last_loan_status == "Closed_EarlyRepaid":
        early_repayments_count = max(1, early_repayments_count)

    # Credit limit utilization
    limit_utilization_pct = round(random.uniform(0.05, 0.95), 3)
    if last_loan_status in ("Delinquent", "Defaulted"):
        limit_utilization_pct = round(random.uniform(0.65, 1.00), 3)

    # Top-up behavior (revolving / repeat borrowing)
    top_up_count = random.choices([0, 1, 2, 3], weights=[0.45, 0.35, 0.15, 0.05], k=1)[0]

    # App engagement recency
    app_engagement_recency_days = random.randint(0, 60)
    if last_loan_status in ("Delinquent", "Defaulted"):
        app_engagement_recency_days = random.randint(30, 180)

    # Relationship score (0-100)
    rel_score = round(
        (on_time_payment_rate * 45)
        + (min(1.0, months_as_customer / 36.0) * 20)
        + (min(1.0, prior_loans_count / 5.0) * 15)
        + (early_repayments_count * 3)
        + (10 if active_loans_count == 0 else 5)
        - (collection_contacts_count * 2)
        - (max_dpd_ever / 365.0 * 20)
    , 1)
    rel_score = max(0.0, min(100.0, rel_score))

    return {
        "customer_id": cid,
        "has_internal_history": True,
        "prior_loans_count": prior_loans_count,
        "active_loans_count": active_loans_count,
        "last_loan_amount": last_loan_amount,
        "last_loan_status": last_loan_status,
        "avg_loan_amount": avg_loan_amount,
        "on_time_payment_rate": on_time_payment_rate,
        "avg_dpd_last_12m": avg_dpd_last_12m,
        "max_dpd_ever": max_dpd_ever,
        "early_repayments_count": early_repayments_count,
        "collection_contacts_count": collection_contacts_count,
        "limit_utilization_pct": limit_utilization_pct,
        "top_up_count": top_up_count,
        "app_engagement_recency_days": app_engagement_recency_days,
        "months_as_customer": months_as_customer,
        "relationship_score": rel_score,
    }

def main():
    logging.info(f"Loading customers from {CUSTOMER_FILE}...")
    df_cust = pd.read_csv(CUSTOMER_FILE)

    logging.info(f"Generating loan ledger for {len(df_cust):,} customers...")
    records = [generate_loan_ledger(row) for _, row in tqdm(df_cust.iterrows(), total=len(df_cust))]
    df_out = pd.DataFrame(records)

    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_out):,} records → {OUTPUT_FILE}")

    with_history = df_out["has_internal_history"].sum()
    with_def = (df_out["last_loan_status"] == "Defaulted").sum()
    print(f"\n{'='*55}")
    print(f"  LOAN LEDGER — GENERATION SUMMARY")
    print(f"{'='*55}")
    print(f"Total Records         : {len(df_out):,}")
    print(f"Has Internal History  : {with_history:,} ({with_history/len(df_out)*100:.1f}%)")
    print(f"Defaulted Customers   : {with_def:,}")
    print(f"Avg Relationship Score: {df_out.loc[df_out['has_internal_history'], 'relationship_score'].mean():.1f}/100")
    print(f"Avg Months as Customer: {df_out.loc[df_out['has_internal_history'], 'months_as_customer'].mean():.1f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
