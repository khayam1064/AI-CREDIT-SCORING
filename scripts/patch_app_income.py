with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace row_fs.get('p50_income', 0) with r_score.get('p50_income', 0)
text = text.replace(
    'c3.metric("P50 Expected Income", f"PKR {float(row_fs.get(\'p50_income\', 0)):,.0f}")',
    'p50_disp = float(r_score.get(\'p50_income\', row_fs.get(\'total_monthly_income\', 65000)) or 65000)\n        c3.metric("P50 Expected Income", f"PKR {p50_disp:,.0f}")'
)

text = text.replace(
    'c3.metric("Expected Income (P50)", f"PKR {float(row_fs.get(\'p50_income\', 0)):,.0f}")',
    'p50_disp = float(r_score.get(\'p50_income\', row_fs.get(\'total_monthly_income\', 65000)) or 65000)\n        c3.metric("Expected Income (P50)", f"PKR {p50_disp:,.0f}")'
)

# In Quantile income tab
text = text.replace(
    "p10 = float(row_fs.get('p10_income', 75000) or 75000) if row_fs is not None else 75000.0",
    "p10 = float(r_score.get('p10_income', row_fs.get('total_monthly_income', 55000)*0.85) if 'r_score' in locals() else (row_fs.get('total_monthly_income', 55000)*0.85 if row_fs is not None else 55000.0))"
)
text = text.replace(
    "p50 = float(row_fs.get('p50_income', 110000) or 110000) if row_fs is not None else 110000.0",
    "p50 = float(r_score.get('p50_income', row_fs.get('total_monthly_income', 65000)) if 'r_score' in locals() else (row_fs.get('total_monthly_income', 65000) if row_fs is not None else 65000.0))"
)
text = text.replace(
    "p90 = float(row_fs.get('p90_income', 165000) or 165000) if row_fs is not None else 165000.0",
    "p90 = float(r_score.get('p90_income', row_fs.get('total_monthly_income', 75000)*1.2) if 'r_score' in locals() else (row_fs.get('total_monthly_income', 75000)*1.2 if row_fs is not None else 75000.0))"
)

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully patched income metrics in dashboard/app.py!")
