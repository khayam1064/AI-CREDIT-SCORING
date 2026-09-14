with open("scoring/composite_scorer.py", "r", encoding="utf-8") as f:
    code = f.read()

old_block = """    raw_features = pd.DataFrame(index=df.index)
    raw_features["age"] = df["age"].fillna(35.0)
    raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0)
    raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0)
    raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0)
    raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0)
    raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0)"""

new_block = """    raw_features = pd.DataFrame(index=df.index)
    raw_features["age"] = df["age"].fillna(35.0) if "age" in df.columns else pd.Series(35.0, index=df.index)
    raw_features["savings_ratio"] = df["savings_ratio"].fillna(0.0) if "savings_ratio" in df.columns else pd.Series(0.0, index=df.index)
    raw_features["overdraft_utilization_rate"] = df["overdraft_utilization_rate"].fillna(0.0) if "overdraft_utilization_rate" in df.columns else pd.Series(0.0, index=df.index)
    raw_features["ldr_avg_dpd_last_12m"] = df["ldr_avg_dpd_last_12m"].fillna(0.0) if "ldr_avg_dpd_last_12m" in df.columns else pd.Series(0.0, index=df.index)
    raw_features["ldr_on_time_payment_rate"] = df["ldr_on_time_payment_rate"].fillna(1.0) if "ldr_on_time_payment_rate" in df.columns else pd.Series(1.0, index=df.index)
    raw_features["nsf_count_total"] = df["nsf_count_total"].fillna(0.0) if "nsf_count_total" in df.columns else pd.Series(0.0, index=df.index)"""

code = code.replace(old_block, new_block)

with open("scoring/composite_scorer.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Safely patched raw_features in scoring/composite_scorer.py!")
