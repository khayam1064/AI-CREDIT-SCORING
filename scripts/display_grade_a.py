import pandas as pd

df_scores = pd.read_parquet("data/processed/credit_scores_250k.parquet")
df_fs = pd.read_parquet("feature_store/customer_features_v2.parquet")

# Ensure customer_id is a column in df_scores
if "customer_id" not in df_scores.columns:
    df_scores["customer_id"] = df_fs["customer_id"].values

df_scores.set_index("customer_id", inplace=True)
df_fs.set_index("customer_id", inplace=True)

grade_a = df_scores[df_scores["credit_grade"] == "A"].copy()

print(f"=== GRADE A SUPER-PRIME BORROWERS (SCORE >= 780, PD < 1.5%) ===")
print(f"Total Grade A Customers in Portfolio: {len(grade_a)}\n")

records = []
for cid in grade_a.index:
    row = df_fs.loc[cid]
    name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
    score = int(grade_a.loc[cid, 'composite_score_1000'])
    pd_val = float(grade_a.loc[cid, 'calibrated_pd']) * 100
    income = float(row.get('total_monthly_income', 0))
    foir = float(row.get('foir_pct', 0))
    on_time = float(row.get('ldr_on_time_payment_rate', 1.0) or 1.0) * 100
    city = str(row.get('city', 'N/A'))
    decision = str(grade_a.loc[cid, 'credit_decision'])
    
    records.append({
        "Customer ID": cid,
        "Full Name": name,
        "Score": f"{score} / 900",
        "Grade": "A",
        "12M Default (PD)": f"{pd_val:.2f}%",
        "Monthly Income": f"PKR {income:,.0f}",
        "FOIR": f"{foir:.1f}%",
        "Repayment Rate": f"{on_time:.0f}%",
        "City": city,
        "Decision": decision
    })

df_res = pd.DataFrame(records)
print(df_res.to_string(index=False))
