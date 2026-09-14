with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Update IFRS 9 call in Tab 1 to pass df_features=df_fs
text = text.replace(
    "ifrs9 = policy_engine.calculate_portfolio_ifrs9(df_scores)",
    "ifrs9 = policy_engine.calculate_portfolio_ifrs9(df_scores, df_features=df_fs)"
)

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated dashboard/app.py to use dynamic account-level IFRS 9 EAD and Macro Scenarios!")
