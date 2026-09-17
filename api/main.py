from scoring import composite_scorer, policy_engine, monitoring_service, disbursement_ledger
"""
Enterprise Credit AI Engine
FastAPI Model Serving & Quantile Scoring Endpoint

Serves real-time P10, P50, and P90 Income Predictions with SHAP Explainability
and Monotonic Non-Crossing Safeguards.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from fastapi.responses import RedirectResponse

def normalize_customer_id(raw_id) -> str:
    """Standardizes raw customer inputs into 'CUST0000001' format."""
    s = str(raw_id).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.startswith("CUST"):
        try:
            return f"CUST{int(s[4:]):07d}"
        except ValueError:
            return s
    try:
        return f"CUST{int(s):07d}"
    except ValueError:
        return s


# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features_v2.parquet" if (BASE_DIR / "feature_store" / "customer_features_v2.parquet").exists() else BASE_DIR / "feature_store" / "customer_features.parquet"

app = FastAPI(
    title="Enterprise Credit AI Engine - Income Prediction Service",
    description="Serves LightGBM Quantile Predictions (P10, P50, P90) with SHAP Explainability",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Global State Containers
models = {}
feature_cols = []
df_feature_store = None
shap_explainer = None
categorical_info = None

@app.on_event("startup")
def load_artifacts():
    global models, feature_cols, df_feature_store, shap_explainer, categorical_info
    
    # Load Models
    for q_name in ["p10", "p50", "p90"]:
        m_path = MODELS_DIR / f"lgb_income_{q_name}.joblib"
        if m_path.exists():
            models[q_name] = joblib.load(m_path)
            
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    
    # Load categorical information if available
    if (MODELS_DIR / "categorical_info.joblib").exists():
        categorical_info = joblib.load(MODELS_DIR / "categorical_info.joblib")
    
    # Load Feature Store Parquet for fast customer_id lookup
    if FEATURE_STORE_PARQUET.exists():
        df_feature_store = pd.read_parquet(FEATURE_STORE_PARQUET)
        df_feature_store["customer_id"] = df_feature_store["customer_id"].apply(normalize_customer_id)
        df_feature_store.set_index("customer_id", inplace=True)

    # Initialize SHAP Explainer
    try:
        from explainability.shap_explainer import IncomeSHAPExplainer
        shap_explainer = IncomeSHAPExplainer()
    except Exception as e:
        print(f"SHAP Explainer startup warning: {e}")

class IncomePredictionResponse(BaseModel):
    customer_id: str
    p10_conservative_income: float
    p50_median_income: float
    p90_optimistic_income: float
    income_bandwidth: float
    stability_confidence_score: float
    explainability: Optional[Dict[str, Any]] = None
    shap_summary: Optional[str] = None

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")    

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "models_loaded": list(models.keys()),
        "feature_count": len(feature_cols),
        "feature_store_records": len(df_feature_store) if df_feature_store is not None else 0
    }

# Append to api/main.py

from engine.underwriting_rules import CreditUnderwritingEngine

underwriting_engine = CreditUnderwritingEngine()

@app.get("/api/v1/predict/customer/{customer_id}")
def predict_customer(customer_id: str):
    # Normalize customer ID
    target_id = normalize_customer_id(customer_id)
    
    if df_feature_store is None or target_id not in df_feature_store.index:
        raise HTTPException(status_code=404, detail=f"Customer ID '{target_id}' not found in Feature Store.")

    row_data = df_feature_store.loc[[target_id]][feature_cols].copy()
    
    # Properly handle categorical features
    if categorical_info and "categorical_columns" in categorical_info:
        cat_cols = categorical_info["categorical_columns"]
        for c in cat_cols:
            if c in row_data.columns:
                row_data[c] = row_data[c].astype("category")
    else:
        # Fallback to auto-detection
        cat_cols = row_data.select_dtypes(include=["object", "string"]).columns
        for c in cat_cols:
            row_data[c] = row_data[c].astype("category")

    if _credit_scores_df is not None and target_id in _credit_scores_df.index:
        p10 = float(_credit_scores_df.loc[target_id, "p10_income"])
        p50 = float(_credit_scores_df.loc[target_id, "p50_income"])
        p90 = float(_credit_scores_df.loc[target_id, "p90_income"])
    else:
        try:
            p10_arr, p50_arr, p90_arr = composite_scorer.predict_income_quantiles(row_data)
            p10, p50, p90 = float(p10_arr[0]), float(p50_arr[0]), float(p90_arr[0])
        except Exception:
            raw_p10 = float(models["p10"].predict(row_data)[0])
            raw_p50 = float(models["p50"].predict(row_data)[0])
            raw_p90 = float(models["p90"].predict(row_data)[0])
            p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])

    # Generate SHAP explanation if available
    explanation = None
    shap_summary = None
    if shap_explainer:
        try:
            explanation = shap_explainer.explain_sample(row_data)
            shap_summary = explanation.get("summary_explanation", "")
        except Exception as e:
            print(f"SHAP explanation error: {e}")

    confidence = float(row_data.get("income_confidence_score", pd.Series([75.0])).values[0])

    return {
        "customer_id": target_id,
        "p10_conservative_income": round(p10, 2),
        "p50_median_income": round(p50, 2),
        "p90_optimistic_income": round(p90, 2),
        "income_bandwidth": round(p90 - p10, 2),
        "stability_confidence_score": round(confidence, 2),
        "explainability": explanation,
        "shap_summary": shap_summary
    }

@app.get("/api/v1/underwrite/customer/{customer_id}")
def underwrite_customer(customer_id: str):
    # Normalize customer ID
    target_id = normalize_customer_id(customer_id)
    
    if df_feature_store is None or target_id not in df_feature_store.index:
        raise HTTPException(status_code=404, detail=f"Customer ID '{target_id}' not found in Feature Store.")

    row_data = df_feature_store.loc[[target_id]][feature_cols].copy()
    
    # Properly handle categorical features
    if categorical_info and "categorical_columns" in categorical_info:
        cat_cols = categorical_info["categorical_columns"]
        for c in cat_cols:
            if c in row_data.columns:
                row_data[c] = row_data[c].astype("category")
    else:
        # Fallback to auto-detection
        cat_cols = row_data.select_dtypes(include=["object", "string"]).columns
        for c in cat_cols:
            row_data[c] = row_data[c].astype("category")

    if _credit_scores_df is not None and target_id in _credit_scores_df.index:
        p10 = float(_credit_scores_df.loc[target_id, "p10_income"])
        p50 = float(_credit_scores_df.loc[target_id, "p50_income"])
        p90 = float(_credit_scores_df.loc[target_id, "p90_income"])
    else:
        try:
            p10_arr, p50_arr, p90_arr = composite_scorer.predict_income_quantiles(row_data)
            p10, p50, p90 = float(p10_arr[0]), float(p50_arr[0]), float(p90_arr[0])
        except Exception:
            raw_p10 = float(models["p10"].predict(row_data)[0])
            raw_p50 = float(models["p50"].predict(row_data)[0])
            raw_p90 = float(models["p90"].predict(row_data)[0])
            p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])

    obligations = float(row_data.get("utility_debit_amt_12m", pd.Series([0.0])).values[0] / 12.0)
    stability = float(row_data.get("income_stability_score", pd.Series([70.0])).values[0])
    fraud_flag = bool(row_data.get("has_fraud_flag", pd.Series([False])).values[0])

    assessment = underwriting_engine.evaluate_credit(
        customer_id=target_id,
        p10_income=p10,
        p50_income=p50,
        p90_income=p90,
        existing_monthly_obligations=obligations,
        income_stability_score=stability,
        has_fraud_flag=fraud_flag
    )

    return assessment


# ─────────────────────────────────────────────────────────────────────────────
#  12-PILLAR CREDIT SCORING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# Global preloaded results for fast lookup
_credit_scores_df: pd.DataFrame = None

CREDIT_SCORES_PARQUET = BASE_DIR / "data" / "processed" / "credit_scores_250k.parquet"
FEATURE_STORE_V2      = BASE_DIR / "feature_store" / "customer_features_v2.parquet"

@app.on_event("startup")
def load_credit_scores():
    global _credit_scores_df
    if CREDIT_SCORES_PARQUET.exists():
        _credit_scores_df = pd.read_parquet(CREDIT_SCORES_PARQUET)
        _credit_scores_df["customer_id"] = _credit_scores_df["customer_id"].apply(normalize_customer_id)
        _credit_scores_df.set_index("customer_id", inplace=True)
        print(f"Credit scores loaded: {len(_credit_scores_df):,} customers")
    else:
        print("WARNING: credit_scores_250k.parquet not found. Run scoring/run_batch_scoring.py first.")


@app.get("/api/v1/score/customer/{customer_id}", tags=["Credit Scoring"])
def get_customer_credit_score(customer_id: str):
    """
    Returns the full 12-pillar credit score breakdown for a single customer.
    Includes: pillar scores (P01–P12), composite score (0-1000), credit grade
    (A+/A/B/C/D/F), final decision (APPROVED/REFER/DECLINED), and decline reason.

    Run `python scoring/run_batch_scoring.py` to generate scores first.
    """
    target_id = normalize_customer_id(customer_id)

    if _credit_scores_df is None:
        raise HTTPException(
            status_code=503,
            detail="Credit scores not loaded. Run: python scoring/run_batch_scoring.py"
        )

    if target_id not in _credit_scores_df.index:
        raise HTTPException(status_code=404, detail=f"Customer '{target_id}' not found in credit scores.")

    row = _credit_scores_df.loc[target_id].to_dict()

    pillar_scores = {
        "P01_Identity_Trust":     round(float(row.get("p01_score", 0)), 1),
        "P02_Fraud_Risk":         round(float(row.get("p02_score", 0)), 1),
        "P03_Income":             round(float(row.get("p03_score", 0)), 1),
        "P04_Cash_Flow":          round(float(row.get("p04_score", 0)), 1),
        "P05_Affordability":      round(float(row.get("p05_score", 0)), 1),
        "P06_Stability":          round(float(row.get("p06_score", 0)), 1),
        "P07_Behavioral":         round(float(row.get("p07_score", 0)), 1),
        "P08_Digital_Footprint":  round(float(row.get("p08_score", 0)), 1),
        "P09_Bureau_Proxy":       round(float(row.get("p09_score", 0)), 1),
        "P10_Relationship":       round(float(row.get("p10_score", 0)), 1),
        "P11_Collection_Risk":    round(float(row.get("p11_score", 0)), 1),
        "P12_Compliance":         str(row.get("p12_compliance", "PASS")),
    }

    return {
        "customer_id":           target_id,
        "composite_score_1000":  int(row.get("composite_score_1000", 0)),
        "composite_score_100":   round(float(row.get("composite_score_100", 0)), 2),
        "credit_grade":          str(row.get("credit_grade", "F")),
        "credit_decision":       str(row.get("credit_decision", "DECLINED")),
        "decline_reason":        None if pd.isna(row.get("decline_reason")) else str(row.get("decline_reason")),
        "bureau_weight_effective": round(float(row.get("w_p09_effective", 0.12)), 4),
        "relationship_weight_effective": round(float(row.get("w_p10_effective", 0.0)), 4),
        "pillar_scores":         pillar_scores,
    }


@app.get("/api/v1/score/batch/summary", tags=["Credit Scoring"])
def get_batch_score_summary():
    """
    Returns portfolio-level statistics across all 250K scored customers:
    decision distribution, grade distribution, score percentiles,
    and pillar score averages — broken down by customer segment.
    """
    if _credit_scores_df is None:
        raise HTTPException(status_code=503, detail="Credit scores not loaded.")

    df = _credit_scores_df.reset_index()
    total = len(df)

    decisions = df["credit_decision"].value_counts().to_dict()
    grades    = df["credit_grade"].value_counts().to_dict()

    scored = df[df["composite_score_1000"] > 0]["composite_score_1000"]

    pillar_avgs = {}
    for col in df.columns:
        if col.endswith("_score") and (col.startswith("p0") or col.startswith("p1")):
            pillar_avgs[col] = round(float(df[col].mean()), 1)

    decline_reasons = df["decline_reason"].dropna().value_counts().head(10).to_dict()

    return {
        "total_customers": total,
        "decisions": {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in decisions.items()},
        "grades":    {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in grades.items()},
        "score_statistics": {
            "mean":   round(float(scored.mean()), 0) if len(scored) else 0,
            "median": round(float(scored.median()), 0) if len(scored) else 0,
            "std":    round(float(scored.std()), 0) if len(scored) else 0,
            "p10":    round(float(scored.quantile(0.10)), 0) if len(scored) else 0,
            "p25":    round(float(scored.quantile(0.25)), 0) if len(scored) else 0,
            "p75":    round(float(scored.quantile(0.75)), 0) if len(scored) else 0,
            "p90":    round(float(scored.quantile(0.90)), 0) if len(scored) else 0,
        },
        "pillar_averages": pillar_avgs,
        "top_decline_reasons": decline_reasons,
    }


@app.get("/api/v1/taxonomy/families", tags=["Credit Scoring"])
def get_data_taxonomy():
    """
    Returns metadata describing all 15 data families, which pillars they feed,
    and the signals collected from each family.
    """
    return {
        "taxonomy_version": "1.0",
        "total_families": 15,
        "families": [
            {"id": "F1",  "name": "Identity & KYC",           "feeds_pillars": ["P01"],               "base_features": "60-80",   "consent": "Mandatory"},
            {"id": "F2",  "name": "Demographics",             "feeds_pillars": ["P01", "P06"],         "base_features": "30-50",   "consent": "Mandatory"},
            {"id": "F3",  "name": "Device Intelligence",      "feeds_pillars": ["P02", "P07"],         "base_features": "200-300", "consent": "Mandatory (disclosed)"},
            {"id": "F4",  "name": "Behavioral Biometrics",    "feeds_pillars": ["P02", "P07"],         "base_features": "150-250", "consent": "Mandatory (disclosed)"},
            {"id": "F5",  "name": "Telecom",                  "feeds_pillars": ["P06", "P08"],         "base_features": "150-220", "consent": "Opt-in"},
            {"id": "F6",  "name": "Banking & Open Banking",   "feeds_pillars": ["P03","P04","P05","P09","P11"], "base_features": "350-450", "consent": "Opt-in"},
            {"id": "F7",  "name": "Wallets & Payments",       "feeds_pillars": ["P07", "P11"],         "base_features": "150-250", "consent": "Opt-in"},
            {"id": "F8",  "name": "Utility, Rent & Subscriptions", "feeds_pillars": ["P04", "P05"],  "base_features": "80-120",  "consent": "Opt-in"},
            {"id": "F9",  "name": "Credit Bureau",            "feeds_pillars": ["P09"],               "base_features": "80-150",  "consent": "Mandatory where available"},
            {"id": "F10", "name": "Location Intelligence",    "feeds_pillars": ["P02", "P12"],         "base_features": "80-120",  "consent": "Mandatory + Opt-in"},
            {"id": "F11", "name": "Digital Footprint",        "feeds_pillars": ["P08"],               "base_features": "40-80",   "consent": "Mandatory (fraud)"},
            {"id": "F12", "name": "Employment & Income",      "feeds_pillars": ["P03", "P06"],         "base_features": "80-150",  "consent": "Opt-in + Derived"},
            {"id": "F13", "name": "Relationship (Internal)",  "feeds_pillars": ["P10", "P11"],         "base_features": "150-250", "consent": "Automatic"},
            {"id": "F14", "name": "Application Meta & Velocity", "feeds_pillars": ["P02", "P12"],      "base_features": "60-100",  "consent": "Automatic"},
            {"id": "F15", "name": "Macro & Portfolio Context","feeds_pillars": ["P05"],               "base_features": "50-100",  "consent": "Automatic"},
        ]
    }

class DisbursePayload(BaseModel):
    customer_id: str
    amount: float
    tenor_months: int
    e_sign_name: str

@app.get("/api/v1/portfolio/ifrs9", tags=["Enterprise Risk"])
def get_portfolio_ifrs9():
    if _credit_scores_df is None:
        raise HTTPException(status_code=503, detail="Credit scores not loaded.")
    res = policy_engine.calculate_portfolio_ifrs9(_credit_scores_df, df_features=df_feature_store)
    return {
        "status": "SUCCESS",
        "portfolio_summary": {
            "total_accounts": len(_credit_scores_df),
            "total_portfolio_exposure_ead_pkr": res.get("total_portfolio_exposure", 0.0),
            "total_weighted_ecl_provision_pkr": res.get("total_ecl_provision", 0.0),
            "provision_coverage_ratio": res.get("ecl_coverage_ratio_pct", 0.0)
        },
        "scenario_provisions": res.get("macro_scenarios", {}),
        "stage_breakdown": res.get("stages", {})
    }

@app.get("/api/v1/portfolio/drift-monitoring", tags=["Enterprise Risk"])
def get_drift_metrics():
    if _credit_scores_df is None:
        raise HTTPException(status_code=503, detail="Credit scores not loaded.")
    res = monitoring_service.monitor_portfolio_stability(_credit_scores_df, df_features=df_feature_store)
    return {
        "status": "SUCCESS",
        "monitoring_report": res
    }

@app.post("/api/v1/loans/disburse", tags=["Core Banking"])
def disburse_loan(payload: DisbursePayload):
    cid = normalize_customer_id(payload.customer_id)
    if df_feature_store is None or cid not in df_feature_store.index:
        raise HTTPException(status_code=404, detail=f"Customer {cid} not registered.")
        
    row_fs = df_feature_store.loc[[cid]]
    live_scores = composite_scorer.compute_scores(row_fs)
    r = live_scores.iloc[0]
    
    if r["credit_decision"] != "APPROVED":
        raise HTTPException(status_code=400, detail=f"Underwriting declined: {r.get('decline_reason', 'Risk policy stop')}")
        
    pricing = policy_engine.calculate_limits_and_pricing(
        grade=r["credit_grade"],
        pd_cal=float(r["calibrated_pd"]),
        p10_income=float(r["p10_income"]),
        p50_income=float(r["p50_income"]),
        existing_obligations=float(row_fs.get("foir_pct", pd.Series([20.0])).iloc[0])/100.0 * float(r["p50_income"]),
        prior_loans_count=int(row_fs.get("ldr_prior_loans_count", pd.Series([0])).iloc[0])
    )
    
    if payload.amount > pricing["approved_limit"]:
        limit_val = pricing['approved_limit']
        raise HTTPException(status_code=400, detail=f"Requested amount PKR {payload.amount:,.0f} exceeds approved limit of PKR {limit_val:,.0f}")
        
    rec = disbursement_ledger.record_disbursement(
        customer_id=cid,
        disbursed_amount=payload.amount,
        tenor_months=payload.tenor_months,
        credit_grade=r["credit_grade"],
        risk_adjusted_apr=pricing["apr_pct"],
        e_sign_name=payload.e_sign_name
    )
    return {
        "status": "DISBURSED_SUCCESSFULLY",
        "loan_record": rec
    }

@app.get("/api/v1/loans/ledger", tags=["Core Banking"])
def get_loan_ledger():
    df = disbursement_ledger.get_disbursed_loans(50)
    return {
        "status": "SUCCESS",
        "total_disbursed_loans": len(df),
        "ledger": df.to_dict(orient="records") if not df.empty else []
    }


# ---------------------------------------------------------------------------
# Real-Time Payload Underwriting Endpoint (Direct Ciihive Integration)
# ---------------------------------------------------------------------------
try:
    from api.schemas import RealtimeApplicantPayload
except ImportError:
    from schemas import RealtimeApplicantPayload

@app.post('/api/v1/score/customer/realtime', tags=['Credit Scoring'])
def score_customer_realtime(payload: RealtimeApplicantPayload):
    import pandas as pd
    from scoring.composite_scorer import compute_scores
    from scoring.policy_engine import calculate_limits_and_pricing

    raw_dict = payload.dict()
    df_in = pd.DataFrame([raw_dict])
    scored_df = compute_scores(df_in)
    r = scored_df.iloc[0]
    existing_debt = float(r.get('total_monthly_debit', 0.0)) * 0.25 if 'total_monthly_debit' in r else 0.0
    p10_val = float(r.get('p10_income', 50000.0))
    p50_val = float(r.get('p50_income', 75000.0))
    pricing = calculate_limits_and_pricing(
        grade=str(r['credit_grade']),
        pd_cal=float(r['calibrated_pd']),
        p10_income=p10_val,
        p50_income=p50_val,
        existing_obligations=existing_debt,
        prior_loans_count=int(payload.ldr_prior_loans_count or 0)
    )
    sub_scores = {}
    for k in r.index:
        if k.startswith('p0') or k.startswith('p1'):
            val = r[k]
            try:
                sub_scores[k] = float(val)
            except (ValueError, TypeError):
                sub_scores[k] = str(val)
    
    return {
        'status': 'SUCCESS',
        'customer_id': payload.customer_id,
        'credit_score': int(r['composite_score_1000']),
        'credit_grade': str(r['credit_grade']),
        'decision': str(r['credit_decision']),
        'calibrated_pd': float(r['calibrated_pd']),
        'decline_reason': str(r['decline_reason']) if pd.notna(r.get('decline_reason')) else None,
        'approved_limit_pkr': float(pricing.get('approved_limit', 0.0)),
        'recommended_apr_pct': float(pricing.get('recommended_apr', 0.0)),
        'max_affordable_emi_pkr': float(pricing.get('max_affordable_emi', 0.0)),
        'bounds': {
            'bound_1_affordability_pkr': float(pricing.get('bound_1_affordability', 0.0)),
            'bound_2_risk_cap_pkr': float(pricing.get('bound_2_risk_cap', 0.0)),
            'bound_3_policy_cap_pkr': float(pricing.get('bound_3_policy_cap', 0.0)),
            'bound_4_progression_pkr': float(pricing.get('bound_4_progression', 0.0))
        },
        'sub_pillar_scores': sub_scores
    }
