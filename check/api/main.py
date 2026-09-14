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


# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"

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

@app.on_event("startup")
def load_artifacts():
    global models, feature_cols, df_feature_store, shap_explainer
    
    # Load Models
    for q_name in ["p10", "p50", "p90"]:
        m_path = MODELS_DIR / f"lgb_income_{q_name}.joblib"
        if m_path.exists():
            models[q_name] = joblib.load(m_path)
            
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    
    # Load Feature Store Parquet for fast customer_id lookup
    if FEATURE_STORE_PARQUET.exists():
        df_feature_store = pd.read_parquet(FEATURE_STORE_PARQUET)
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

@app.get("/api/v1/underwrite/customer/{customer_id}")
def underwrite_customer(customer_id: str):
    if df_feature_store is None or customer_id not in df_feature_store.index:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found in Feature Store.")

    row_data = df_feature_store.loc[[customer_id]][feature_cols].copy()
    cat_cols = row_data.select_dtypes(include=["object", "string"]).columns
    for c in cat_cols:
        row_data[c] = row_data[c].astype("category")

    raw_p10 = float(models["p10"].predict(row_data)[0])
    raw_p50 = float(models["p50"].predict(row_data)[0])
    raw_p90 = float(models["p90"].predict(row_data)[0])
    p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])

    obligations = float(row_data.get("utility_debit_amt_12m", pd.Series([0.0])).values[0] / 12.0)
    stability = float(row_data.get("income_stability_score", pd.Series([70.0])).values[0])
    fraud_flag = bool(row_data.get("has_fraud_flag", pd.Series([False])).values[0])

    assessment = underwriting_engine.evaluate_credit(
        customer_id=customer_id,
        p10_income=p10,
        p50_income=p50,
        p90_income=p90,
        existing_monthly_obligations=obligations,
        income_stability_score=stability,
        has_fraud_flag=fraud_flag
    )

    return assessment