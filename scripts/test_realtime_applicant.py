import sys
from pathlib import Path
import json
import urllib.request

# 1. Prepare raw incoming applicant payload (e.g. from mobile app or credit bureau)
applicant_data = {
    "customer_id": "REAL_TIME_TEST_001",
    "age": 34,
    "gender": "male",
    "education_level": "bachelor",
    "total_monthly_income": 135000.0,
    "verified_income": 125000.0,
    "total_monthly_credit": 130000.0,
    "total_monthly_debit": 85000.0,
    "total_avg_monthly_balance": 48000.0,
    "overdraft_utilization_rate": 0.15,
    "savings_ratio": 0.28,
    "foir_pct": 16.5,
    "ldr_on_time_payment_rate": 1.00,
    "ldr_prior_loans_count": 2,
    
    "dev_is_rooted": False,
    "dev_emulator_detected": False,
    "app_applications_same_device_7d": 1
}

# 2. Test direct model inference via composite_scorer & policy_engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scoring import composite_scorer, policy_engine, shap_explainer

print("=== 1. DIRECT PYTHON INFERENCE TEST ===")
df_in = pd.DataFrame([applicant_data], index=[applicant_data["customer_id"]])
scores = composite_scorer.compute_scores(df_in)
r = scores.iloc[0]

pricing = policy_engine.calculate_limits_and_pricing(
    grade=str(r["credit_grade"]),
    pd_cal=float(r["calibrated_pd"]),
    p10_income=float(r["p10_income"]),
    p50_income=float(r["p50_income"]),
    existing_obligations=5000.0,
    prior_loans_count=applicant_data["ldr_prior_loans_count"]
)

sub_dict = {k: r[k] for k in r.index if k.startswith("p0") or k.startswith("p1")}
xai = shap_explainer.explain_prediction(pd.Series(applicant_data), sub_dict)

print(f"Customer ID:        {applicant_data['customer_id']}")
print(f"Credit Decision:    {r['credit_decision']}")
print(f"Composite Score:    {r['composite_score_1000']} / 900 (Grade {r['credit_grade']})")
print(f"Calibrated PD:      {r['calibrated_pd']*100:.2f}%")
print(f"P10 Floor Income:   PKR {r['p10_income']:,.0f}")
print(f"P50 Median Income:  PKR {r['p50_income']:,.0f}")
print(f"P90 Ceiling Income: PKR {r['p90_income']:,.0f}")
print(f"Approved Limit:     PKR {pricing['approved_limit']:,.0f}")
print(f"Fixed APR:          {pricing['apr_pct']:.2f}% p.a.")
print(f"Max Affordable EMI: PKR {pricing['max_affordable_emi']:,.0f}/mo")
print(f"Top Positive Trust: {xai['top_positive'][0]['name'] if xai['top_positive'] else 'N/A'}")
print(f"Adverse Actions:    {len(xai['top_negative'])} flagged")

# 3. Test HTTP REST API Gateway on Port 8080
print("\n=== 2. LIVE REST API GATEWAY TEST (PORT 8080) ===")
try:
    req = urllib.request.Request(
        "http://localhost:8080/api/v1/score/underwrite",
        data=json.dumps(applicant_data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req, timeout=10)
    api_resp = json.loads(res.read().decode())
    print(f"API HTTP Status:    {res.getcode()} OK")
    print(f"API Latency:        {api_resp.get('execution_time_ms', 'N/A')} ms")
    print(f"API Decision:       {api_resp.get('credit_decision')}")
    print(f"API Score:          {api_resp.get('composite_score')}")
    print(f"API Facility Limit: PKR {api_resp.get('credit_limits', {}).get('approved_facility_limit', 0):,.0f}")
except Exception as e:
    print(f"API Call Failed: {e}")
