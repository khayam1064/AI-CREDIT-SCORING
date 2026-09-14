"""
Enterprise Credit AI Engine
Master Transaction History Generator & Validation Pipeline

Generates 12 months of daily granular transaction streams tied directly to:
- datasets/raw/customers.csv
- datasets/raw/employment.csv
- datasets/raw/income.csv
- datasets/raw/bank_accounts.csv

Powers Cash Flow Analytics, Income Verification, Expense Ratio Estimation,
Underwriting Rule Engines, and Fraud Detection Models.
"""

import os
import re
import sys
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta

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
    DATASETS_DIR = BASE_DIR / "data" / "raw"
else:
    DATASETS_DIR = BASE_DIR / "datasets" / "raw"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOMER_FILE = DATASETS_DIR / "customers.csv"
EMPLOYMENT_FILE = DATASETS_DIR / "employment.csv"
INCOME_FILE = DATASETS_DIR / "income.csv"
BANKING_FILE = DATASETS_DIR / "bank_accounts.csv"
TRANSACTIONS_FILE = DATASETS_DIR / "transactions.csv"

# Global Constants & Seeds
RANDOM_SEED = 42
TODAY = datetime(2026, 8, 5)
START_DATE = TODAY - timedelta(days=365)  # 12 Months Simulation Window

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Business Lookups & Transaction Categories
# -----------------------------------------------------------------------------

UTILITY_PROVIDERS = {
    "Electricity": ["K-Electric", "LESCO", "IESCO", "FESCO", "MEPCO"],
    "Gas": ["SSGC", "SNGPL"],
    "Water": ["KWSC", "WASAL"],
    "Telecom": ["Jazz", "Zong 4G", "Telenor", "PTCL", "Ufone", "StormFiber"]
}

POS_MERCHANTS = [
    "Imtiaz Super Market", "Metro Cash & Carry", "Carrefour", "Naheed Superstore",
    "Khaadi", "Gul Ahmed", "J. Junaid Jamshed", "Al-Fatah", "Hyperstar",
    "Shell Fuel Station", "Total Parco", "PSO Station", "KFC", "McDonald's",
    "Dominos", "Savour Foods", "Tezraftar Courier", "Daraz.pk Online"
]

ATM_LOCATIONS = [
    "HBL ATM - Main Market", "UBL ATM - Commercial Area", "Meezan Bank ATM - DHA",
    "MCB ATM - Mall Road", "Bank Alfalah ATM - Express", "SCB ATM - Airport Road"
]

PAYMENT_CHANNELS = {
    "Salary Credit": ["1LINK 1PAY Payroll", "ACH Salary Transfer", "Corporate IBFT"],
    "Bill Payment": ["Mobile Banking App", "Internet Banking Portal", "ATM Bill Pay"],
    "POS Merchant": ["Debit Card POS", "Contactless NFC", "QR Code Scan"],
    "Cash Withdrawal": ["ATM Withdrawal", "Branch Counter Cash"],
    "Peer Transfer": ["1LINK Raast", "IBFT Mobile App", "ATM Transfer"],
    "Bank Fee": ["System Deduction"]
}

def normalize_cust_id(raw_id):
    s = str(raw_id).strip()
    if s.startswith("CUST"):
        return s
    try:
        return f"CUST{int(s):07d}"
    except ValueError:
        return s

def weighted_choice(mapping):
    choices = list(mapping.keys())
    weights = list(mapping.values())
    return random.choices(choices, weights=weights, k=1)[0]

# -----------------------------------------------------------------------------
# Transaction Stream Generator Engine
# -----------------------------------------------------------------------------

def generate_account_transactions(account_row, global_tx_idx):
    account_id = account_row["account_id"]
    customer_id = account_row["customer_id"]
    account_status = str(account_row["account_status"])
    is_salary_acc = bool(account_row.get("is_salary_account", False))
    
    opening_dt = datetime.strptime(str(account_row["opening_date"]), "%Y-%m-%d")
    sim_start = max(START_DATE, opening_dt)
    
    if account_status in ["Dormant", "Closed"]:
        return [], global_tx_idx

    gross_salary = float(account_row.get("gross_monthly_salary", 0.0))
    if gross_salary <= 0:
        gross_salary = float(account_row.get("total_monthly_income", 0.0)) * 0.70
        
    net_salary = float(account_row.get("net_monthly_salary", gross_salary * 0.88))
    total_income = float(account_row.get("total_monthly_income", gross_salary))
    
    current_balance = float(account_row.get("available_balance", 25000.0))
    salary_credit_day = int(account_row.get("salary_credit_day") if pd.notnull(account_row.get("salary_credit_day")) else random.randint(1, 5))

    transactions = []
    curr_date = sim_start

    while curr_date <= TODAY:
        day_of_month = curr_date.day

        # ---------------------------------------------------------------------
        # 1. Monthly Recurring Salary Deposit (For Salary Accounts)
        # ---------------------------------------------------------------------
        if is_salary_acc and day_of_month == salary_credit_day and net_salary > 0:
            tx_id = f"TXN{global_tx_idx:010d}"
            global_tx_idx += 1
            
            deposit_amt = round(net_salary * random.uniform(0.98, 1.02), 2)
            current_balance += deposit_amt
            
            transactions.append({
                "transaction_id": tx_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "transaction_timestamp": curr_date.strftime("%Y-%m-%d 09:15:00"),
                "transaction_date": curr_date.strftime("%Y-%m-%d"),
                "transaction_type": "Credit",
                "category": "Salary Credit",
                "sub_category": "Direct Deposit",
                "amount": deposit_amt,
                "running_balance": round(current_balance, 2),
                "channel": random.choice(PAYMENT_CHANNELS["Salary Credit"]),
                "description": f"Payroll Deposit - {account_row.get('company_name', 'Employer')}",
                "counterparty_name": str(account_row.get("company_name", "Employer Corp")),
                "counterparty_bank": str(account_row.get("bank_name", "Habib Bank Limited")),
                "reference_number": f"REF-SAL-{random.randint(100000, 999999)}",
                "is_recurring": True,
                "is_payroll": True,
                "status": "Completed"
            })

        # ---------------------------------------------------------------------
        # 2. Monthly Utility Bill Payments (Days 5 to 15)
        # ---------------------------------------------------------------------
        if is_salary_acc and 5 <= day_of_month <= 15 and random.random() < 0.12 and current_balance > 5000:
            tx_id = f"TXN{global_tx_idx:010d}"
            global_tx_idx += 1
            
            util_type = random.choice(list(UTILITY_PROVIDERS.keys()))
            util_provider = random.choice(UTILITY_PROVIDERS[util_type])
            
            if util_type == "Electricity":
                bill_amt = round(random.uniform(8000, 45000), 2)
            elif util_type == "Gas":
                bill_amt = round(random.uniform(1500, 8000), 2)
            elif util_type == "Telecom":
                bill_amt = round(random.uniform(2000, 12000), 2)
            else:
                bill_amt = round(random.uniform(1000, 4000), 2)

            bill_amt = min(bill_amt, current_balance * 0.40)
            current_balance -= bill_amt

            transactions.append({
                "transaction_id": tx_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "transaction_timestamp": curr_date.strftime(f"%Y-%m-%d {random.randint(10, 18):02d}:{random.randint(10, 59):02d}:00"),
                "transaction_date": curr_date.strftime("%Y-%m-%d"),
                "transaction_type": "Debit",
                "category": "Utility Payment",
                "sub_category": util_type,
                "amount": bill_amt,
                "running_balance": round(current_balance, 2),
                "channel": random.choice(PAYMENT_CHANNELS["Bill Payment"]),
                "description": f"Bill Payment - {util_provider}",
                "counterparty_name": util_provider,
                "counterparty_bank": "N/A",
                "reference_number": f"BILL-{random.randint(100000, 999999)}",
                "is_recurring": True,
                "is_payroll": False,
                "status": "Completed"
            })

        # ---------------------------------------------------------------------
        # 3. Daily POS / Merchant Shopping (25% - 40% chance per day)
        # ---------------------------------------------------------------------
        if random.random() < 0.30 and current_balance > 1000:
            tx_id = f"TXN{global_tx_idx:010d}"
            global_tx_idx += 1
            
            pos_amt = round(random.uniform(350, min(15000, max(1000, current_balance * 0.25))), 2)
            current_balance -= pos_amt
            merchant = random.choice(POS_MERCHANTS)

            transactions.append({
                "transaction_id": tx_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "transaction_timestamp": curr_date.strftime(f"%Y-%m-%d {random.randint(11, 21):02d}:{random.randint(10, 59):02d}:00"),
                "transaction_date": curr_date.strftime("%Y-%m-%d"),
                "transaction_type": "Debit",
                "category": "POS Merchant",
                "sub_category": "Retail & Groceries",
                "amount": pos_amt,
                "running_balance": round(current_balance, 2),
                "channel": random.choice(PAYMENT_CHANNELS["POS Merchant"]),
                "description": f"POS Purchase - {merchant}",
                "counterparty_name": merchant,
                "counterparty_bank": "Acquiring Bank",
                "reference_number": f"POS-{random.randint(100000, 999999)}",
                "is_recurring": False,
                "is_payroll": False,
                "status": "Completed"
            })

        # ---------------------------------------------------------------------
        # 4. Weekly ATM Cash Withdrawals
        # ---------------------------------------------------------------------
        if random.random() < 0.15 and current_balance > 3000:
            tx_id = f"TXN{global_tx_idx:010d}"
            global_tx_idx += 1
            
            cash_amt = float(random.choice([2000, 5000, 10000, 15000, 20000]))
            cash_amt = min(cash_amt, current_balance * 0.50)
            current_balance -= cash_amt
            atm_loc = random.choice(ATM_LOCATIONS)

            transactions.append({
                "transaction_id": tx_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "transaction_timestamp": curr_date.strftime(f"%Y-%m-%d {random.randint(8, 20):02d}:{random.randint(10, 59):02d}:00"),
                "transaction_date": curr_date.strftime("%Y-%m-%d"),
                "transaction_type": "Debit",
                "category": "Cash Withdrawal",
                "sub_category": "ATM",
                "amount": cash_amt,
                "running_balance": round(current_balance, 2),
                "channel": random.choice(PAYMENT_CHANNELS["Cash Withdrawal"]),
                "description": f"ATM Cash - {atm_loc}",
                "counterparty_name": atm_loc,
                "counterparty_bank": str(account_row.get("bank_name", "Issuer Bank")),
                "reference_number": f"ATM-{random.randint(100000, 999999)}",
                "is_recurring": False,
                "is_payroll": False,
                "status": "Completed"
            })

        # ---------------------------------------------------------------------
        # 5. Monthly Maintenance / SMS Alert Fees (End of month)
        # ---------------------------------------------------------------------
        if day_of_month == 28 and current_balance > 200:
            tx_id = f"TXN{global_tx_idx:010d}"
            global_tx_idx += 1
            
            fee_amt = 150.0
            current_balance -= fee_amt

            transactions.append({
                "transaction_id": tx_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "transaction_timestamp": curr_date.strftime("%Y-%m-%d 23:59:00"),
                "transaction_date": curr_date.strftime("%Y-%m-%d"),
                "transaction_type": "Debit",
                "category": "Bank Fee",
                "sub_category": "SMS Banking Fee",
                "amount": fee_amt,
                "running_balance": round(current_balance, 2),
                "channel": "System Deduction",
                "description": "Monthly Digital Alert Service Fee",
                "counterparty_name": str(account_row.get("bank_name", "Bank")),
                "counterparty_bank": str(account_row.get("bank_name", "Bank")),
                "reference_number": f"FEE-{random.randint(100000, 999999)}",
                "is_recurring": True,
                "is_payroll": False,
                "status": "Completed"
            })

        curr_date += timedelta(days=1)

    return transactions, global_tx_idx

# -----------------------------------------------------------------------------
# Master Generator Pipeline Execution
# -----------------------------------------------------------------------------

def generate_dataset():
    logging.info(f"Loading master banking profile from {BANKING_FILE}...")
    df_bank = pd.read_csv(BANKING_FILE)
    df_cust = pd.read_csv(CUSTOMER_FILE)
    df_emp = pd.read_csv(EMPLOYMENT_FILE)
    df_inc = pd.read_csv(INCOME_FILE)

    df_bank["customer_id"] = df_bank["customer_id"].apply(normalize_cust_id)
    df_cust["customer_id"] = df_cust["customer_id"].apply(normalize_cust_id)
    df_emp["customer_id"] = df_emp["customer_id"].apply(normalize_cust_id)
    df_inc["customer_id"] = df_inc["customer_id"].apply(normalize_cust_id)

    # Merge master profiles
    df_merged = df_bank.merge(df_cust[["customer_id", "date_of_birth"]], on="customer_id", how="left")
    df_merged = df_merged.merge(df_emp[["customer_id", "company_name"]], on="customer_id", how="left")
    df_merged = df_merged.merge(
        df_inc[["customer_id", "gross_monthly_salary", "net_monthly_salary", "total_monthly_income"]], 
        on="customer_id", 
        how="left"
    )

    logging.info(f"Simulating 12-month transaction streams across {len(df_merged):,} accounts...")
    
    all_transactions = []
    global_tx_idx = 1

    for _, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
        acc_txs, global_tx_idx = generate_account_transactions(row, global_tx_idx)
        all_transactions.extend(acc_txs)

    return pd.DataFrame(all_transactions), df_bank, df_cust

# -----------------------------------------------------------------------------
# Enterprise Validation Audit
# -----------------------------------------------------------------------------

def run_enterprise_validation(df_tx, df_bank, df_cust):
    logging.info("Executing Enterprise Transaction Data Quality Audit...")
    results = {}

    results["Transaction ID Format Check"] = df_tx["transaction_id"].str.match(r"^TXN\d{10}$").all()
    results["Transaction ID Uniqueness"] = df_tx["transaction_id"].duplicated().sum() == 0
    results["Referential Integrity (Account Exists)"] = df_tx["account_id"].isin(df_bank["account_id"]).all()
    results["Referential Integrity (Customer Exists)"] = df_tx["customer_id"].isin(df_cust["customer_id"]).all()

    df_tx["tx_dt"] = pd.to_datetime(df_tx["transaction_date"])
    results["Transaction Timeline (Last 12 Months)"] = (
        (df_tx["tx_dt"] >= pd.to_datetime(START_DATE.strftime("%Y-%m-%d"))) & 
        (df_tx["tx_dt"] <= TODAY)
    ).all()

    results["Non-Negative Transaction Amounts"] = (df_tx["amount"] >= 0.0).all()
    results["Valid Transaction Types (Credit/Debit)"] = df_tx["transaction_type"].isin(["Credit", "Debit"]).all()
    
    payroll_txs = df_tx[df_tx["is_payroll"] == True]
    results["Payroll Transactions Are Credits Only"] = (payroll_txs["transaction_type"] == "Credit").all()

    df_tx.drop(columns=[c for c in df_tx.columns if c.endswith("_dt")], inplace=True)
    return results

def print_validation_report(results, df_tx):
    total_checks = len(results)
    passed_checks = sum(results.values())
    quality_score = (passed_checks / total_checks) * 100

    report = f"""
============================================================
         TRANSACTION HISTORY DATA QUALITY REPORT
============================================================

Records Generated              : {len(df_tx):,}

Transaction ID Format Check    : {"PASS" if results.get("Transaction ID Format Check", False) else "FAIL"}
Transaction ID Uniqueness      : {"PASS" if results.get("Transaction ID Uniqueness", False) else "FAIL"}
Account Mapping Check          : {"PASS" if results.get("Referential Integrity (Account Exists)", False) else "FAIL"}
Customer Mapping Check         : {"PASS" if results.get("Referential Integrity (Customer Exists)", False) else "FAIL"}

Timeline Sequence (12 Months)  : {"PASS" if results.get("Transaction Timeline (Last 12 Months)", False) else "FAIL"}
Non-Negative Transaction Amounts: {"PASS" if results.get("Non-Negative Transaction Amounts", False) else "FAIL"}
Valid Transaction Types        : {"PASS" if results.get("Valid Transaction Types (Credit/Debit)", False) else "FAIL"}
Payroll Direct Deposit Rules   : {"PASS" if results.get("Payroll Transactions Are Credits Only", False) else "FAIL"}

------------------------------------------------------------
Overall Data Quality Score     : {quality_score:.2f}%
Dataset Status                 : {"PRODUCTION READY" if quality_score == 100.0 else "VALIDATION FAILED"}
============================================================
"""
    print(report)
    if quality_score < 100.0:
        failed_tests = [k for k, v in results.items() if not v]
        print("FAILED CHECKS:", failed_tests)
        raise Exception("Transaction Dataset Quality Assurance Failed.")

def save_dataset(df_tx):
    df_tx.to_csv(TRANSACTIONS_FILE, index=False, encoding="utf-8")
    logging.info(f"Master transaction history dataset saved to: {TRANSACTIONS_FILE}")

def main():
    df_tx, df_bank, df_cust = generate_dataset()
    results = run_enterprise_validation(df_tx, df_bank, df_cust)
    print_validation_report(results, df_tx)
    save_dataset(df_tx)

if __name__ == "__main__":
    main()