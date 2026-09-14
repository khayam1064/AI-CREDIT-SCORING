with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

old_func = """@st.cache_resource(show_spinner=False)
def load_customer_names():
    p = PROJECT_ROOT / 'datasets' / 'raw' / 'customers.csv'
    if not p.exists(): return {}
    try:
        df = pd.read_csv(p, usecols=['customer_id', 'first_name', 'last_name'])
        df['customer_id'] = df['customer_id'].apply(normalize_id)
        df['full_name'] = (df['first_name'].fillna('').astype(str) + ' ' + df['last_name'].fillna('').astype(str)).str.strip()
        return dict(zip(df['customer_id'], df['full_name']))
    except Exception:
        return {}"""

new_func = """@st.cache_resource(show_spinner=False)
def load_customer_names():
    p_fs = PROJECT_ROOT / 'feature_store' / 'customer_features_v2.parquet'
    if p_fs.exists():
        try:
            df = pd.read_parquet(p_fs, columns=['customer_id', 'first_name', 'last_name'])
            df['customer_id'] = df['customer_id'].apply(normalize_id)
            df['full_name'] = (df['first_name'].fillna('').astype(str) + ' ' + df['last_name'].fillna('').astype(str)).str.strip()
            return dict(zip(df['customer_id'], df['full_name']))
        except Exception:
            pass
    p = PROJECT_ROOT / 'datasets' / 'raw' / 'customers.csv'
    if not p.exists(): return {}
    try:
        df = pd.read_csv(p, usecols=['customer_id', 'first_name', 'last_name'])
        df['customer_id'] = df['customer_id'].apply(normalize_id)
        df['full_name'] = (df['first_name'].fillna('').astype(str) + ' ' + df['last_name'].fillna('').astype(str)).str.strip()
        return dict(zip(df['customer_id'], df['full_name']))
    except Exception:
        return {}"""

if old_func in text:
    text = text.replace(old_func, new_func)
    print("Replaced load_customer_names directly!")
else:
    # Partial fallback replacement
    target = "p = PROJECT_ROOT / 'datasets' / 'raw' / 'customers.csv'"
    replacement = "p_fs = PROJECT_ROOT / 'feature_store' / 'customer_features_v2.parquet'\n    if p_fs.exists():\n        try:\n            df = pd.read_parquet(p_fs, columns=['customer_id', 'first_name', 'last_name'])\n            df['customer_id'] = df['customer_id'].apply(normalize_id)\n            df['full_name'] = (df['first_name'].fillna('').astype(str) + ' ' + df['last_name'].fillna('').astype(str)).str.strip()\n            return dict(zip(df['customer_id'], df['full_name']))\n        except Exception:\n            pass\n    p = PROJECT_ROOT / 'datasets' / 'raw' / 'customers.csv'"
    text = text.replace(target, replacement)
    print("Replaced via target fallback!")

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated dashboard/app.py to read customer names directly from customer_features_v2.parquet!")
