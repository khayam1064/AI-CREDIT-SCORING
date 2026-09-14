from pathlib import Path

# ==========================
# PROJECT ROOT
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================
# DATA DIRECTORIES
# ==========================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEATURE_STORE_DIR = DATA_DIR / "feature_store"

MODEL_DIR = BASE_DIR / "models"

# ==========================
# FILE PATHS
# ==========================

CUSTOMER_FILE = RAW_DATA_DIR / "customers.csv"

EMPLOYMENT_FILE = RAW_DATA_DIR / "employment.csv"

INCOME_FILE = RAW_DATA_DIR / "income.csv"

BANKING_FILE = RAW_DATA_DIR / "banking.csv"

WALLET_FILE = RAW_DATA_DIR / "wallet.csv"

SPENDING_FILE = RAW_DATA_DIR / "spending.csv"

FINAL_DATASET = DATA_DIR / "synthetic_income.csv"

# ==========================
# RANDOM SEED
# ==========================

RANDOM_SEED = 42

# ==========================
# DATASET SIZE
# ==========================

TOTAL_CUSTOMERS = 250000