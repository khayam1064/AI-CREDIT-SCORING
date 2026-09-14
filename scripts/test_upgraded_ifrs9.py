import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scoring import policy_engine

df_scores = pd.read_parquet("data/processed/credit_scores_250k.parquet").set_index("customer_id")
df_fs = pd.read_parquet("feature_store/customer_features_v2.parquet").set_index("customer_id")

res = policy_engine.calculate_portfolio_ifrs9(df_scores, df_features=df_fs)
print("=== Upgraded IFRS 9 Engine Validation ===")
print("Total Portfolio Exposure (EAD): PKR", f"{res['total_portfolio_exposure']:,.0f}")
print("Total Weighted ECL Provision:   PKR", f"{res['total_ecl_provision']:,.0f}")
print("Coverage Ratio:                    ", f"{res['ecl_coverage_ratio_pct']}%")
print("\nMacroeconomic Scenarios:")
for sc, dat in res['macro_scenarios'].items():
    print(f"  {sc} ({dat['weight_pct']}% weight): PKR {dat['total_ecl']:,.0f} (Coverage: {dat['coverage_pct']}%)")
