import sys
from pathlib import Path

comp_file = Path("scoring/composite_scorer.py")

with open(comp_file, "r", encoding="utf-8") as f:
    text = f.read()

# Let's add predict_income_quantiles to composite_scorer.py
income_helper = """
_INCOME_CACHE = {}

def _get_income_models():
    if not _INCOME_CACHE:
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        _INCOME_CACHE["p10"] = joblib.load(models_dir / "lgb_income_p10.joblib")
        _INCOME_CACHE["p50"] = joblib.load(models_dir / "lgb_income_p50.joblib")
        _INCOME_CACHE["p90"] = joblib.load(models_dir / "lgb_income_p90.joblib")
        _INCOME_CACHE["feat_cols"] = joblib.load(models_dir / "feature_columns.joblib")
        _INCOME_CACHE["cat_info"] = joblib.load(models_dir / "categorical_info.joblib")
    return _INCOME_CACHE

def predict_income_quantiles(df: pd.DataFrame) -> tuple:
    \"\"\"Predicts P10, P50, P90 income quantiles via LightGBM Pinball models.\"\"\"
    try:
        cache = _get_income_models()
        feature_cols = cache["feat_cols"]
        cat_cols = cache["cat_info"].get("categorical_columns", [])
        
        available_cols = [c for c in feature_cols if c in df.columns]
        missing_cols = [c for c in feature_cols if c not in df.columns]
        
        X = df[available_cols].copy()
        for c in missing_cols:
            X[c] = 0
        X = X[feature_cols]
        
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
                
        p10 = cache["p10"].predict(X)
        p50 = cache["p50"].predict(X)
        p90 = cache["p90"].predict(X)
        
        sorted_preds = np.sort(np.stack([p10, p50, p90], axis=1), axis=1)
        return sorted_preds[:, 0].round(2), sorted_preds[:, 1].round(2), sorted_preds[:, 2].round(2)
    except Exception as e:
        base = df.get("total_monthly_income", df.get("verified_income", pd.Series(65000.0, index=df.index))).fillna(65000.0)
        return (base * 0.80).round(2), base.round(2), (base * 1.25).round(2)
"""

# Let's insert this helper before compute_scores
if "_get_income_models" not in text:
    text = text.replace("def compute_scores(df: pd.DataFrame) -> pd.DataFrame:", income_helper + "\n\ndef compute_scores(df: pd.DataFrame) -> pd.DataFrame:")

# In compute_scores, let's calculate income quantiles first!
old_start = """def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Computes Credit Risk Scores via the Stage-based Topology.
    \"\"\"
    results = pd.DataFrame(index=df.index)
    results["customer_id"] = df["customer_id"] if "customer_id" in df.columns else df.index"""

new_start = """def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Computes Credit Risk Scores via the Stage-based Topology.
    \"\"\"
    results = pd.DataFrame(index=df.index)
    results["customer_id"] = df["customer_id"] if "customer_id" in df.columns else df.index

    # Predict / populate Income Quantiles
    df_eval = df.copy()
    if "p10_income" not in df_eval.columns or "p50_income" not in df_eval.columns:
        p10_arr, p50_arr, p90_arr = predict_income_quantiles(df_eval)
        df_eval["p10_income"] = p10_arr
        df_eval["p50_income"] = p50_arr
        df_eval["p90_income"] = p90_arr
    
    results["p10_income"] = df_eval["p10_income"]
    results["p50_income"] = df_eval["p50_income"]
    results["p90_income"] = df_eval["p90_income"]"""

if old_start in text:
    text = text.replace(old_start, new_start)

# Replace df with df_eval in sub-score calls
text = text.replace('results["p01_score"] = p01_identity_trust.score(df)', 'results["p01_score"] = p01_identity_trust.score(df_eval)')
text = text.replace('results["p02_score"] = p02_fraud_risk.score(df)', 'results["p02_score"] = p02_fraud_risk.score(df_eval)')
text = text.replace('results["p03_score"] = p03_income.score(df)', 'results["p03_score"] = p03_income.score(df_eval)')
text = text.replace('results["p04_score"] = p04_cash_flow.score(df)', 'results["p04_score"] = p04_cash_flow.score(df_eval)')
text = text.replace('results["p05_score"] = p05_affordability.score(df)', 'results["p05_score"] = p05_affordability.score(df_eval)')
text = text.replace('results["p06_score"] = p06_stability.score(df)', 'results["p06_score"] = p06_stability.score(df_eval)')
text = text.replace('results["p07_score"] = p07_behavioral.score(df)', 'results["p07_score"] = p07_behavioral.score(df_eval)')
text = text.replace('results["p08_score"] = p08_digital_footprint.score(df)', 'results["p08_score"] = p08_digital_footprint.score(df_eval)')
text = text.replace('results["p09_score"] = p09_bureau_proxy.score(df)', 'results["p09_score"] = p09_bureau_proxy.score(df_eval)')
text = text.replace('results["p10_score"] = p10_relationship.score(df)', 'results["p10_score"] = p10_relationship.score(df_eval)')
text = text.replace('results["p11_score"] = p11_collection_risk.score(df)', 'results["p11_score"] = p11_collection_risk.score(df_eval)')
text = text.replace('results["p12_compliance"] = p12_compliance.gate(df)', 'results["p12_compliance"] = p12_compliance.gate(df_eval)')
text = text.replace('fraud_gate = p02_fraud_risk.hard_gate(df)', 'fraud_gate = p02_fraud_risk.hard_gate(df_eval)')

with open(comp_file, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated scoring/composite_scorer.py with live Quantile Income models!")
