with open("dashboard/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "def load_customer_names():" in line:
        new_lines.append(line)
        new_lines.append("    p = PROJECT_ROOT / 'datasets' / 'raw' / 'customers.csv'\n")
        new_lines.append("    if not p.exists(): return {}\n")
        new_lines.append("    try:\n")
        new_lines.append("        df = pd.read_csv(p, usecols=['customer_id', 'first_name', 'last_name'])\n")
        new_lines.append("        df['customer_id'] = df['customer_id'].apply(normalize_id)\n")
        new_lines.append("        df['full_name'] = (df['first_name'].fillna('').astype(str) + ' ' + df['last_name'].fillna('').astype(str)).str.strip()\n")
        new_lines.append("        return dict(zip(df['customer_id'], df['full_name']))\n")
        new_lines.append("    except Exception:\n")
        new_lines.append("        return {}\n")
        skip = True
    elif skip and "names_dict = load_customer_names()" in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Updated dashboard/app.py with robust load_customer_names()!")
