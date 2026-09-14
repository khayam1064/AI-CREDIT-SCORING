"""
Enterprise Credit AI Engine
Wallet & Payments Generator (Family F7)

Generates JazzCash / Easypaisa / SadaPay / Nayapay wallet signals for
all 250,000 customers, correlated with income segment and banking profile.

Output: datasets/raw/wallets.csv
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
OUTPUT_FILE = DATASETS_DIR / "wallets.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

WALLET_PROVIDERS = {
    "JazzCash":  0.40,
    "Easypaisa": 0.35,
    "SadaPay":   0.12,
    "NayaPay":   0.08,
    "HBL Konnect": 0.05,
}

WALLET_ADOPTION_BY_SEGMENT = {
    "Mass":            0.60,
    "Affluent":        0.80,
    "Premium":         0.88,
    "Private Banking": 0.75,
}

def weighted_choice(mapping):
    return random.choices(list(mapping.keys()), weights=list(mapping.values()), k=1)[0]

def generate_wallet(row):
    cid = row["customer_id"]
    segment = str(row.get("customer_segment", "Mass"))
    age = int(row.get("age", 30))
    sim_age_days = int(row.get("sim_age_days", 365))

    has_wallet = random.random() < WALLET_ADOPTION_BY_SEGMENT.get(segment, 0.60)

    if not has_wallet:
        return {
            "customer_id": cid,
            "has_wallet": False,
            "wallet_provider": None,
            "wallet_tenure_months": 0,
            "avg_monthly_txn_count": 0,
            "avg_monthly_txn_value": 0.0,
            "p2p_network_size": 0,
            "qr_merchant_payment_count_6m": 0,
            "bill_pay_via_wallet_6m": 0,
            "chargeback_count_ever": 0,
            "wallet_linked_to_bank": False,
            "wallet_credit_signal": 0.0,
        }

    wallet_provider = weighted_choice(WALLET_PROVIDERS)

    # Wallet tenure bounded by SIM age (can't have wallet before SIM)
    sim_age_months = max(1, sim_age_days // 30)
    wallet_tenure_months = random.randint(1, min(sim_age_months, 72))

    # Transaction activity scales with segment and tenure
    base_txn_count = {"Mass": (2, 15), "Affluent": (8, 40), "Premium": (15, 80), "Private Banking": (10, 50)}.get(segment, (2, 15))
    avg_monthly_txn_count = random.randint(*base_txn_count)

    base_txn_val = {"Mass": (500, 15000), "Affluent": (3000, 50000), "Premium": (10000, 150000), "Private Banking": (20000, 300000)}.get(segment, (500, 15000))
    avg_monthly_txn_value = round(random.uniform(*base_txn_val), 2)

    # P2P network size
    p2p_network_size = random.randint(0, min(150, avg_monthly_txn_count * 4))

    # QR merchant payments in 6m
    qr_merchant_payment_count_6m = int(avg_monthly_txn_count * 6 * random.uniform(0.1, 0.5))

    # Bill pay via wallet
    bill_pay_via_wallet_6m = random.randint(0, 6)

    # Chargebacks — rare fraud signal
    chargeback_count_ever = 0
    if random.random() < 0.03:
        chargeback_count_ever = random.randint(1, 3)

    # Linked to bank account
    wallet_linked_to_bank = random.random() < 0.70

    # Wallet credit signal (0-100, higher = more creditworthy)
    tenure_factor = min(1.0, wallet_tenure_months / 36.0)
    activity_factor = min(1.0, avg_monthly_txn_count / 40.0)
    chargeback_penalty = chargeback_count_ever * 15.0
    wallet_credit_signal = round(
        max(0.0, min(100.0,
            (tenure_factor * 40)
            + (activity_factor * 35)
            + (bill_pay_via_wallet_6m / 6 * 15)
            + (10 if wallet_linked_to_bank else 0)
            - chargeback_penalty
        )), 1
    )

    return {
        "customer_id": cid,
        "has_wallet": True,
        "wallet_provider": wallet_provider,
        "wallet_tenure_months": wallet_tenure_months,
        "avg_monthly_txn_count": avg_monthly_txn_count,
        "avg_monthly_txn_value": avg_monthly_txn_value,
        "p2p_network_size": p2p_network_size,
        "qr_merchant_payment_count_6m": qr_merchant_payment_count_6m,
        "bill_pay_via_wallet_6m": bill_pay_via_wallet_6m,
        "chargeback_count_ever": chargeback_count_ever,
        "wallet_linked_to_bank": wallet_linked_to_bank,
        "wallet_credit_signal": wallet_credit_signal,
    }

def main():
    logging.info(f"Loading customers from {CUSTOMER_FILE}...")
    df_cust = pd.read_csv(CUSTOMER_FILE)

    logging.info(f"Generating wallet signals for {len(df_cust):,} customers...")
    records = [generate_wallet(row) for _, row in tqdm(df_cust.iterrows(), total=len(df_cust))]
    df_out = pd.DataFrame(records)

    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_out):,} records → {OUTPUT_FILE}")

    has_wallet = df_out["has_wallet"].sum()
    print(f"\n{'='*55}")
    print(f"  WALLET DATA — GENERATION SUMMARY")
    print(f"{'='*55}")
    print(f"Total Records         : {len(df_out):,}")
    print(f"Has Wallet            : {has_wallet:,} ({has_wallet/len(df_out)*100:.1f}%)")
    print(f"Avg Wallet Tenure     : {df_out.loc[df_out['has_wallet'], 'wallet_tenure_months'].mean():.1f} months")
    print(f"Chargebacks > 0       : {(df_out['chargeback_count_ever'] > 0).sum():,}")
    print(f"Avg Wallet Cr Signal  : {df_out.loc[df_out['has_wallet'], 'wallet_credit_signal'].mean():.1f}/100")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
