"""
Enterprise Credit AI Engine
Device Intelligence & Behavioral Biometrics Generator (Families F3 + F4)

Generates realistic synthetic device intelligence and in-app behavioral
biometric signals for all 250,000 customers, correlated with fraud flags,
customer segment, and age.

Output: datasets/raw/device_behavioral.csv
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
OUTPUT_FILE = DATASETS_DIR / "device_behavioral.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Device Price Tiers ────────────────────────────────────────────────────────
DEVICE_PRICE_TIER_BY_SEGMENT = {
    "Mass":            {"Budget": 0.50, "Mid": 0.35, "High": 0.12, "Premium": 0.03},
    "Affluent":        {"Budget": 0.15, "Mid": 0.40, "High": 0.35, "Premium": 0.10},
    "Premium":         {"Budget": 0.05, "Mid": 0.20, "High": 0.40, "Premium": 0.35},
    "Private Banking": {"Budget": 0.02, "Mid": 0.10, "High": 0.38, "Premium": 0.50},
}

OS_TYPE_BY_TIER = {
    "Budget":  {"Android": 0.97, "iOS": 0.03},
    "Mid":     {"Android": 0.88, "iOS": 0.12},
    "High":    {"Android": 0.65, "iOS": 0.35},
    "Premium": {"Android": 0.35, "iOS": 0.65},
}

DEVICE_AGE_MONTHS_BY_TIER = {
    "Budget":  (3, 36),
    "Mid":     (3, 30),
    "High":    (3, 24),
    "Premium": (1, 18),
}

def weighted_choice(mapping):
    return random.choices(list(mapping.keys()), weights=list(mapping.values()), k=1)[0]

def generate_device_behavioral(row):
    cid = row["customer_id"]
    segment = str(row.get("customer_segment", "Mass"))
    age = int(row.get("age", 30))
    risk_region = str(row.get("risk_region", "Low"))
    device_count = int(row.get("device_count", 1))

    # ── Base fraud-risk probability tuning ────────────────────────────────
    base_fraud_prob = {"Low": 0.03, "Medium": 0.06, "High": 0.12}.get(risk_region, 0.05)
    is_high_risk = random.random() < base_fraud_prob

    # ── F3: Device Intelligence ────────────────────────────────────────────
    tier_map = DEVICE_PRICE_TIER_BY_SEGMENT.get(segment, DEVICE_PRICE_TIER_BY_SEGMENT["Mass"])
    device_price_tier = weighted_choice(tier_map)

    os_map = OS_TYPE_BY_TIER[device_price_tier]
    os_type = weighted_choice(os_map)

    age_range = DEVICE_AGE_MONTHS_BY_TIER[device_price_tier]
    device_age_months = random.randint(*age_range)

    os_versions = {"Android": ["12", "13", "14", "15"], "iOS": ["15", "16", "17", "18"]}
    os_version = random.choice(os_versions[os_type])

    is_rooted = is_high_risk and random.random() < 0.40
    emulator_detected = is_high_risk and random.random() < 0.25
    vpn_detected = (is_high_risk and random.random() < 0.55) or random.random() < 0.05
    proxy_detected = vpn_detected and random.random() < 0.30
    timezone_gps_mismatch = is_high_risk and random.random() < 0.45

    # SIM swap: higher in fraud/high-risk profiles
    sim_swap_count_6m = 0
    if is_high_risk:
        sim_swap_count_6m = random.choices([0, 1, 2, 3], weights=[0.40, 0.35, 0.18, 0.07], k=1)[0]
    else:
        sim_swap_count_6m = random.choices([0, 1, 2], weights=[0.88, 0.10, 0.02], k=1)[0]

    battery_pct = random.randint(15, 100)
    storage_free_pct = random.uniform(5.0, 80.0)
    app_integrity_passed = not emulator_detected and not is_rooted
    registered_device_count = min(device_count, random.randint(1, 3))
    device_change_count_12m = random.choices([0, 1, 2, 3], weights=[0.65, 0.25, 0.08, 0.02], k=1)[0]

    # GPS / location flags
    location_permission_granted = not is_high_risk or random.random() < 0.60
    impossible_travel_flag = is_high_risk and random.random() < 0.20

    # ── F4: Behavioral Biometrics ──────────────────────────────────────────
    # Genuine users: faster, consistent; fraudsters: slower or copy-paste
    if is_high_risk:
        typing_speed_wpm = random.uniform(5.0, 25.0)
        correction_rate_pct = random.uniform(15.0, 55.0)
        copy_paste_id_field = random.random() < 0.70
        copy_paste_name_field = random.random() < 0.65
        hesitation_score = random.uniform(55.0, 95.0)       # High hesitation = suspicious
        form_completion_time_sec = random.uniform(180.0, 600.0)
        session_replay_anomaly_score = random.uniform(50.0, 100.0)
        swipe_consistency_score = random.uniform(10.0, 50.0)
        scroll_speed_pct = random.uniform(5.0, 30.0)
        avg_session_duration_sec = random.uniform(30.0, 120.0)
        session_count_7d = random.randint(1, 4)
    else:
        typing_speed_wpm = random.uniform(20.0, 65.0)
        correction_rate_pct = random.uniform(2.0, 18.0)
        copy_paste_id_field = random.random() < 0.08
        copy_paste_name_field = random.random() < 0.06
        hesitation_score = random.uniform(5.0, 40.0)
        form_completion_time_sec = random.uniform(45.0, 180.0)
        session_replay_anomaly_score = random.uniform(0.0, 25.0)
        swipe_consistency_score = random.uniform(65.0, 100.0)
        scroll_speed_pct = random.uniform(30.0, 85.0)
        avg_session_duration_sec = random.uniform(120.0, 480.0)
        session_count_7d = random.randint(3, 15)

    # Age-adjusted: older users slightly slower typing
    if age > 50:
        typing_speed_wpm *= 0.80
        form_completion_time_sec *= 1.20

    # Aggregate behavioral risk score (0-100, higher = riskier)
    behavioral_risk_score = round(
        (hesitation_score * 0.30)
        + (session_replay_anomaly_score * 0.25)
        + ((100 - swipe_consistency_score) * 0.20)
        + ((correction_rate_pct / 55.0) * 100 * 0.15)
        + ((1 if copy_paste_id_field else 0) * 10.0)
    , 2)
    behavioral_risk_score = min(100.0, max(0.0, behavioral_risk_score))

    return {
        "customer_id": cid,
        # F3 — Device Intelligence
        "device_price_tier": device_price_tier,
        "device_age_months": device_age_months,
        "os_type": os_type,
        "os_version": os_version,
        "is_rooted": is_rooted,
        "emulator_detected": emulator_detected,
        "vpn_detected": vpn_detected,
        "proxy_detected": proxy_detected,
        "timezone_gps_mismatch": timezone_gps_mismatch,
        "sim_swap_count_6m": sim_swap_count_6m,
        "battery_pct": battery_pct,
        "storage_free_pct": round(storage_free_pct, 1),
        "app_integrity_passed": app_integrity_passed,
        "registered_device_count": registered_device_count,
        "device_change_count_12m": device_change_count_12m,
        "location_permission_granted": location_permission_granted,
        "impossible_travel_flag": impossible_travel_flag,
        # F4 — Behavioral Biometrics
        "typing_speed_wpm": round(typing_speed_wpm, 1),
        "correction_rate_pct": round(correction_rate_pct, 1),
        "copy_paste_id_field": copy_paste_id_field,
        "copy_paste_name_field": copy_paste_name_field,
        "hesitation_score": round(hesitation_score, 1),
        "form_completion_time_sec": round(form_completion_time_sec, 1),
        "session_replay_anomaly_score": round(session_replay_anomaly_score, 1),
        "swipe_consistency_score": round(swipe_consistency_score, 1),
        "scroll_speed_pct": round(scroll_speed_pct, 1),
        "avg_session_duration_sec": round(avg_session_duration_sec, 1),
        "session_count_7d": session_count_7d,
        "behavioral_risk_score": behavioral_risk_score,
    }

def main():
    logging.info(f"Loading customers from {CUSTOMER_FILE}...")
    df_cust = pd.read_csv(CUSTOMER_FILE)

    logging.info(f"Generating device & behavioral signals for {len(df_cust):,} customers...")
    records = [generate_device_behavioral(row) for _, row in tqdm(df_cust.iterrows(), total=len(df_cust))]
    df_out = pd.DataFrame(records)

    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_out):,} records → {OUTPUT_FILE}")

    # Validation summary
    print(f"\n{'='*55}")
    print(f"  DEVICE & BEHAVIORAL DATA — GENERATION SUMMARY")
    print(f"{'='*55}")
    print(f"Total Records        : {len(df_out):,}")
    print(f"Rooted Devices       : {df_out['is_rooted'].sum():,} ({df_out['is_rooted'].mean()*100:.1f}%)")
    print(f"Emulators Detected   : {df_out['emulator_detected'].sum():,}")
    print(f"VPN Detected         : {df_out['vpn_detected'].sum():,}")
    print(f"Copy-Paste ID Field  : {df_out['copy_paste_id_field'].sum():,}")
    print(f"Impossible Travel    : {df_out['impossible_travel_flag'].sum():,}")
    print(f"Avg Behavioral Risk  : {df_out['behavioral_risk_score'].mean():.1f}/100")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
