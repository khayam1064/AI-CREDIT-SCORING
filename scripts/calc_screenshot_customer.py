import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring import policy_engine

p10 = 28800.0   # Conservative P10 income floor
p50 = 36003.0   # P50 Median Expected income from screenshot
oblg = 1548.0   # Existing monthly debt (4.3% FOIR)
grade = "C"
pd_val = 0.0409 # 4.09% PD from screenshot
prior_loans = 1 # 1 clean repaid loan

res = policy_engine.calculate_limits_and_pricing(
    grade=grade,
    pd_cal=pd_val,
    p10_income=p10,
    p50_income=p50,
    existing_obligations=oblg,
    prior_loans_count=prior_loans
)

print("=== EXACT LIMIT SIZING FOR THIS CUSTOMER ===")
print("Approved Facility Limit: PKR", f"{res['approved_limit']:,.2f}")
print("Revolving Card Limit:    PKR", f"{res['card_limit']:,.2f}")
print("Max Monthly EMI:         PKR", f"{res['max_affordable_emi']:,.2f}")
print("Fixed Annual APR:        ", f"{res['apr_pct']}%")
print("\n--- The 4 Bounds Evaluated by Formula ---")
print("1. Affordability Limit:  PKR", f"{res['affordability_limit']:,.2f}")
print("2. Risk-Adjusted Limit:  PKR", f"{res['risk_limit']:,.2f}")
print("3. Progression Cap:      PKR", f"{res['progression_cap']:,.2f}")
print("4. Institutional Policy: PKR", f"{res['policy_cap']:,.2f}")
