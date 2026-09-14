import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

LEDGER_PARQUET = Path("data/processed/disbursed_loans.parquet")
LEDGER_CSV = Path("data/processed/disbursed_loans.csv")

def record_disbursement(
    customer_id: str,
    full_name: str,
    amount: float,
    tenor_months: int,
    apr_pct: float,
    monthly_emi: float,
    total_repayable: float,
    e_sign_name: str,
    credit_grade: str,
    calibrated_pd: float
) -> dict:
    LEDGER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now()
    ref_code = f"PK-DISB-{timestamp.strftime('%Y%m%d')}-{np.random.randint(10000, 99999)}"
    
    record = {
        "disbursement_id": ref_code,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": str(customer_id),
        "full_name": str(full_name),
        "principal_pkr": float(amount),
        "tenor_months": int(tenor_months),
        "apr_pct": float(apr_pct),
        "monthly_emi_pkr": float(monthly_emi),
        "total_repayable_pkr": float(total_repayable),
        "credit_grade": str(credit_grade),
        "calibrated_pd_pct": round(float(calibrated_pd) * 100, 2),
        "e_sign_name": str(e_sign_name),
        "status": "DISBURSED_ACTIVE",
        "payment_rail": "SBP_RAAST_INSTANT",
        "settlement_status": "SETTLED_CREDITED"
    }
    
    new_df = pd.DataFrame([record])
    
    if LEDGER_PARQUET.exists():
        try:
            existing_df = pd.read_parquet(LEDGER_PARQUET)
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df
        
    combined_df.to_parquet(LEDGER_PARQUET, index=False)
    combined_df.to_csv(LEDGER_CSV, index=False)
    
    return record

def get_disbursed_loans(limit: int = 50) -> pd.DataFrame:
    if not LEDGER_PARQUET.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(LEDGER_PARQUET)
        return df.head(limit)
    except Exception:
        return pd.DataFrame()

if __name__ == "__main__":
    test_rec = record_disbursement(
        customer_id="CUST0000003",
        full_name="Shahid Hashmi",
        amount=75000.0,
        tenor_months=24,
        apr_pct=25.93,
        monthly_emi=8283.41,
        total_repayable=198801.84,
        e_sign_name="Shahid Hashmi",
        credit_grade="C",
        calibrated_pd=0.0582
    )
    print("Disbursement ledger initialized successfully:")
    print(test_rec)
