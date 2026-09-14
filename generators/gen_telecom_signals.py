"""
Enterprise Credit AI Engine
Telecom Signals Generator (Family F5)

Generates realistic MNO/carrier-derived signals for all 250,000 customers,
correlated with their existing telecom profile (operator, phone_type, sim_age_days).

Output: datasets/raw/telecom_signals.csv
"""

import random
import logging
from pathlib import Path

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
OUTPUT_FILE = DATASETS_DIR / "telecom_signals.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_USAGE_BY_SEGMENT = {
    "Mass":            (0.5, 8.0),
    "Affluent":        (3.0, 25.0),
    "Premium":         (10.0, 60.0),
    "Private Banking": (15.0, 100.0),
}

def generate_telecom(row):
    cid = row["customer_id"]
    sim_age_days = int(row.get("sim_age_days", 365))
    phone_type = str(row.get("phone_type", "Prepaid"))
    mobile_operator = str(row.get("mobile_operator", "Jazz"))
    segment = str(row.get("customer_segment", "Mass"))
    age = int(row.get("age", 30))

    is_postpaid = (phone_type == "Postpaid")
    sim_age_months = max(1, sim_age_days // 30)

    # Recharge (Prepaid) or Bill (Postpaid)
    if is_postpaid:
        recharge_frequency_monthly = 1  # monthly billing cycle
        avg_recharge_amount = random.uniform(800.0, 5000.0)
        top_up_regularity_score = round(random.uniform(65.0, 100.0), 1)
        bill_payment_punctuality_score = round(random.uniform(60.0, 100.0), 1)
        months_paid_late_12m = random.choices([0, 1, 2, 3], weights=[0.65, 0.22, 0.10, 0.03], k=1)[0]
    else:
        recharge_frequency_monthly = random.randint(2, 15)
        avg_recharge_amount = random.uniform(50.0, 500.0)
        top_up_regularity_score = round(random.uniform(30.0, 95.0), 1)
        bill_payment_punctuality_score = None   # N/A for prepaid
        months_paid_late_12m = 0

    # Data usage
    usage_range = DATA_USAGE_BY_SEGMENT.get(segment, (0.5, 8.0))
    data_usage_gb_monthly = round(random.uniform(*usage_range), 2)

    # Roaming events — rare, higher for Premium/Private Banking
    roaming_prob = {"Mass": 0.02, "Affluent": 0.08, "Premium": 0.18, "Private Banking": 0.30}.get(segment, 0.02)
    roaming_events_6m = random.choices([0, 1, 2, 3, 5], weights=[1-roaming_prob, roaming_prob*0.6, roaming_prob*0.25, roaming_prob*0.10, roaming_prob*0.05], k=1)[0]

    # Number porting — rare (switching operator)
    porting_events_ever = random.choices([0, 1, 2], weights=[0.82, 0.15, 0.03], k=1)[0]

    # SIM tenure stability score (0-100)
    # Longer SIM age = higher score; postpaid bonus
    tenure_score = min(100.0, (sim_age_months / 120.0) * 100.0)
    if is_postpaid:
        tenure_score = min(100.0, tenure_score * 1.10)
    sim_tenure_stability_score = round(tenure_score, 1)

    # Telecom risk score (higher = riskier)
    telecom_risk_score = round(
        ((1 - sim_tenure_stability_score / 100) * 30)
        + (porting_events_ever * 10)
        + (months_paid_late_12m * 8)
        + ((1 - top_up_regularity_score / 100) * 20)
        + (roaming_events_6m * 2)
    , 2)
    telecom_risk_score = min(100.0, max(0.0, telecom_risk_score))

    # Telecom credit signal (inverse of risk)
    telecom_credit_signal = round(100.0 - telecom_risk_score, 1)

    return {
        "customer_id": cid,
        "sim_age_months": sim_age_months,
        "is_postpaid": is_postpaid,
        "mobile_operator": mobile_operator,
        "recharge_frequency_monthly": recharge_frequency_monthly,
        "avg_recharge_amount": round(avg_recharge_amount, 2),
        "data_usage_gb_monthly": data_usage_gb_monthly,
        "top_up_regularity_score": top_up_regularity_score,
        "bill_payment_punctuality_score": bill_payment_punctuality_score,
        "months_paid_late_12m": months_paid_late_12m,
        "roaming_events_6m": roaming_events_6m,
        "porting_events_ever": porting_events_ever,
        "sim_tenure_stability_score": sim_tenure_stability_score,
        "telecom_risk_score": telecom_risk_score,
        "telecom_credit_signal": telecom_credit_signal,
    }

def main():
    logging.info(f"Loading customers from {CUSTOMER_FILE}...")
    df_cust = pd.read_csv(CUSTOMER_FILE)

    logging.info(f"Generating telecom signals for {len(df_cust):,} customers...")
    records = [generate_telecom(row) for _, row in tqdm(df_cust.iterrows(), total=len(df_cust))]
    df_out = pd.DataFrame(records)

    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_out):,} records → {OUTPUT_FILE}")

    print(f"\n{'='*55}")
    print(f"  TELECOM SIGNALS — GENERATION SUMMARY")
    print(f"{'='*55}")
    print(f"Total Records           : {len(df_out):,}")
    print(f"Postpaid Customers      : {df_out['is_postpaid'].sum():,} ({df_out['is_postpaid'].mean()*100:.1f}%)")
    print(f"Avg Data Usage (GB/mo)  : {df_out['data_usage_gb_monthly'].mean():.1f}")
    print(f"Avg Telecom Credit Sig  : {df_out['telecom_credit_signal'].mean():.1f}/100")
    print(f"Avg Porting Events      : {df_out['porting_events_ever'].mean():.2f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
