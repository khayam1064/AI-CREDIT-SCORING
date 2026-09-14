with open("dashboard/app.py", "r", encoding="utf-8") as f:
    content = f.read()

target = "names_dict = load_customer_names()"
replacement = """df_scores = load_credit_scores()
df_fs = load_feature_store()
names_dict = load_customer_names()"""

if target in content:
    content = content.replace(target, replacement, 1)
    with open("dashboard/app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed data loaders successfully!")
else:
    print("Target not found!")
