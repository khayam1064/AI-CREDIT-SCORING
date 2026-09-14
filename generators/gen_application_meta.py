"""
Enterprise Credit AI Engine
Application Meta & Velocity Generator (Family F14)

Generates synthetic application event metadata for all 250,000 customers,
simulating velocity checks, channel attribution, and application behavior.

Output: datasets/raw/application_meta.csv
"""

import random
import logging
from pathlib import Path
from datetime import datetime

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
OUTPUT_FILE = DATASETS_DIR / "application_meta.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

REFERRAL_SOURCES = {
    "Organic_App_Store": 0.35,
    "SMS_Campaign":      0.20,
    "Partner_Referral":  0.18,
    "Social_Media":      0.12,
    "Email_Campaign":    0.08,
    "USSD":              0.05,
    "Telemarketing":     0.02,
}

REFERRAL_QUALITY = {
    "Organic_App_Store": (75, 100),
    "Partner_Referral":  (70, 95),
    "Email_Campaign":    (60, 90),
    "Social_Media":      (50, 85),
    "SMS_Campaign":      (40, 75),
    "USSD":              (30, 70),
    "Telemarketing":     (20, 60),
}

TIME_OF_DAY_DIST = {
    "Morning (6-12)":   0.28,
    "Afternoon (12-18)": 0.32,
    "Evening (18-22)":   0.30,
    "Night (22-6)":      0.10,
}

def weighted_choice(mapping):
    return random.choices(list(mapping.keys()), weights=list(mapping.values()), k=1)[0]

def generate_application_meta(row):
    cid = row["customer_id"]
    risk_region = str(row.get("risk_region", "Low"))
    segment = str(row.get("customer_segment", "Mass"))
    is_existing = bool(row.get("is_existing_customer", False))

    # Fraud/velocity risk
    is_high_risk = risk_region == "High" and random.random() < 0.15

    # Velocity checks: apps per device/IP in short windows
    if is_high_risk:
        apps_same_device_7d = random.choices([1, 2, 3, 4, 5], weights=[0.20, 0.20, 0.25, 0.20, 0.15], k=1)[0]
        apps_same_ip_24h = random.choices([1, 2, 3, 4], weights=[0.30, 0.30, 0.25, 0.15], k=1)[0]
        cross_lender_velocity_30d = random.choices([0, 1, 2, 3, 4], weights=[0.15, 0.20, 0.25, 0.25, 0.15], k=1)[0]
    else:
        apps_same_device_7d = random.choices([1, 2, 3], weights=[0.80, 0.17, 0.03], k=1)[0]
        apps_same_ip_24h = 1
        cross_lender_velocity_30d = random.choices([0, 1, 2], weights=[0.70, 0.22, 0.08], k=1)[0]

    referral_source = weighted_choice(REFERRAL_SOURCES)
    ref_quality_range = REFERRAL_QUALITY.get(referral_source, (40, 80))
    referral_quality_score = round(random.uniform(*ref_quality_range), 1)

    time_of_day = weighted_choice(TIME_OF_DAY_DIST)

    # Application behavior
    application_duration_sec = round(random.uniform(45, 600), 0)
    if is_high_risk:
        application_duration_sec = round(random.uniform(10, 60), 0)

    form_abandonment_count = random.choices([0, 1, 2, 3], weights=[0.65, 0.22, 0.10, 0.03], k=1)[0]
    device_fingerprint_match = not is_high_risk or random.random() < 0.60
    ip_geolocation_match = not is_high_risk or random.random() < 0.70
    is_repeat_applicant = is_existing

    # Returning customer gets a positive boost
    returning_customer_bonus = 10.0 if is_existing else 0.0

    # Application velocity risk score (0-100, higher = riskier)
    velocity_risk_score = round(
        (apps_same_device_7d * 8)
        + (apps_same_ip_24h * 10)
        + (cross_lender_velocity_30d * 12)
        + (0 if device_fingerprint_match else 15)
        + (0 if ip_geolocation_match else 10)
        + (form_abandonment_count * 5)
        - returning_customer_bonus
    , 1)
    velocity_risk_score = max(0.0, min(100.0, velocity_risk_score))

    return {
        "customer_id": cid,
        "referral_source": referral_source,
        "referral_quality_score": referral_quality_score,
        "time_of_day": time_of_day,
        "applications_same_device_7d": apps_same_device_7d,
        "applications_same_ip_24h": apps_same_ip_24h,
        "cross_lender_velocity_30d": cross_lender_velocity_30d,
        "application_duration_sec": application_duration_sec,
        "form_abandonment_count": form_abandonment_count,
        "device_fingerprint_match": device_fingerprint_match,
        "ip_geolocation_match": ip_geolocation_match,
        "is_repeat_applicant": is_repeat_applicant,
        "velocity_risk_score": velocity_risk_score,
    }

def main():
    logging.info(f"Loading customers from {CUSTOMER_FILE}...")
    df_cust = pd.read_csv(CUSTOMER_FILE)

    logging.info(f"Generating application meta for {len(df_cust):,} customers...")
    records = [generate_application_meta(row) for _, row in tqdm(df_cust.iterrows(), total=len(df_cust))]
    df_out = pd.DataFrame(records)

    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_out):,} records → {OUTPUT_FILE}")

    print(f"\n{'='*55}")
    print(f"  APPLICATION META — GENERATION SUMMARY")
    print(f"{'='*55}")
    print(f"Total Records              : {len(df_out):,}")
    print(f"High Velocity (device 7d>2): {(df_out['applications_same_device_7d'] > 2).sum():,}")
    print(f"Cross-lender Velocity > 1  : {(df_out['cross_lender_velocity_30d'] > 1).sum():,}")
    print(f"Avg Velocity Risk Score    : {df_out['velocity_risk_score'].mean():.1f}/100")
    print(f"Top Referral Source        : {df_out['referral_source'].value_counts().index[0]}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
