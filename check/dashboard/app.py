"""
Enterprise Credit AI Engine
Interactive Executive & Underwriter Dashboard (Streamlit)

Provides real-time interactive analytics for:
- LightGBM Quantile Income Predictions (P10, P50, P90)
- Monotonic Bandwidth & Confidence Metrics
- TreeSHAP Feature Attribution & Model Explainability
- Automated Rule-Based Underwriting & FOIR Analysis
"""

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. System Path Auto-Resolution
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "engine").exists():
    PROJECT_ROOT = BASE_DIR
elif (BASE_DIR.parent / "engine").exists():
    PROJECT_ROOT = BASE_DIR.parent
else:
    PROJECT_ROOT = BASE_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports after path resolution
import json
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 2. Page Configuration & Global Variables
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Credit AI Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURE_STORE_PARQUET = PROJECT_ROOT / "feature_store" / "customer_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
API_BASE_URL = "http://127.0.0.1:8000"

def normalize_cust_id(raw_id) -> str:
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

# -----------------------------------------------------------------------------
# 3. Direct Local Engine Fallback (If FastAPI Server is Offline)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_local_engine():
    """Fallback loader for direct predictions if FastAPI is offline."""
    if not FEATURE_STORE_PARQUET.exists() or not (MODELS_DIR / "lgb_income_p50.joblib").exists():
        return None, None, None, None

    df_fs = pd.read_parquet(FEATURE_STORE_PARQUET)
    if "customer_id" in df_fs.columns:
        df_fs["customer_id"] = df_fs["customer_id"].apply(normalize_cust_id)
        df_fs.set_index("customer_id", inplace=True)
    else:
        df_fs.index = df_fs.index.map(normalize_cust_id)

    feat_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    
    models = {
        "p10": joblib.load(MODELS_DIR / "lgb_income_p10.joblib"),
        "p50": joblib.load(MODELS_DIR / "lgb_income_p50.joblib"),
        "p90": joblib.load(MODELS_DIR / "lgb_income_p90.joblib")
    }

    try:
        from explainability.shap_explainer import IncomeSHAPExplainer
        explainer = IncomeSHAPExplainer()
    except Exception:
        explainer = None

    return df_fs, feat_cols, models, explainer

def evaluate_local_underwriting(p10, p50, p90, obligations=0.0, stability=70.0, fraud_flag=False):
    """Local fallback execution of underwriting rules."""
    if fraud_flag:
        return {
            "decision": "DECLINED",
            "decision_reason": "FRAUD_OR_AML_FLAG_TRIGGERED",
            "existing_monthly_obligations": 0.0,
            "current_foir_percent": 0.0,
            "projected_foir_percent": 0.0,
            "max_affordable_emi": 0.0,
            "approved_term_loan_limit": 0.0,
            "approved_revolving_card_limit": 0.0
        }

    underwriting_income = p10
    max_capacity = underwriting_income * 0.50
    current_foir = round((obligations / underwriting_income) * 100.0, 2) if underwriting_income > 0 else 100.0

    avail_emi = max(0.0, max_capacity - obligations)
    max_affordable_emi = round(avail_emi * 0.90, 2)
    projected_foir = round(((obligations + max_affordable_emi) / underwriting_income) * 100.0, 2) if underwriting_income > 0 else 100.0

    # Loan Principal PV Formula (24M @ 24% annual interest)
    r = 0.24 / 12.0
    n = 24
    max_loan = round(max_affordable_emi * ((1.0 - (1.0 + r) ** (-n)) / r), -3) if max_affordable_emi > 0 else 0.0
    card_limit = round(min(p50 * 2.0, max_loan * 0.50), -3) if max_loan > 0 else 0.0

    if current_foir > 50.0:
        decision = "DECLINED"
        reason = "EXCEEDS_MAXIMUM_FOIR_CAP_50_PERCENT"
    elif max_affordable_emi < 3000.0:
        decision = "DECLINED"
        reason = "INSUFFICIENT_NET_DISPOSABLE_INCOME"
    elif stability < 40.0:
        decision = "REFER"
        reason = "LOW_INCOME_STABILITY_MANUAL_REVIEW_REQUIRED"
    else:
        decision = "APPROVED"
        reason = "MEETS_ALL_UNDERWRITING_STABILITY_AND_FOIR_CRITERIA"

    return {
        "decision": decision,
        "decision_reason": reason,
        "existing_monthly_obligations": round(obligations, 2),
        "current_foir_percent": current_foir,
        "projected_foir_percent": projected_foir if decision == "APPROVED" else current_foir,
        "max_affordable_emi": max_affordable_emi if decision == "APPROVED" else 0.0,
        "approved_term_loan_limit": max_loan if decision == "APPROVED" else 0.0,
        "approved_revolving_card_limit": card_limit if decision == "APPROVED" else 0.0
    }

# -----------------------------------------------------------------------------
# 4. Streamlit Dashboard Layout
# -----------------------------------------------------------------------------
st.title("💳 Enterprise Credit AI Engine - Underwriting & Income Analytics")
st.markdown("Real-time LightGBM Quantile Income Predictions ($P_{10}, P_{50}, P_{90}$), TreeSHAP Explainability, and Automated Credit Decisioning.")

st.sidebar.header("Customer Inspection Panel")

# Load Feature Store Index for Sample Selection
df_fs_demo, _, _, _ = load_local_engine()
if df_fs_demo is not None and not df_fs_demo.empty:
    sample_ids = list(df_fs_demo.index[:10])
else:
    sample_ids = ["CUST0000001", "CUST0000002", "CUST0000003", "CUST0000010", "CUST0000025"]

selected_sample = st.sidebar.selectbox("Select Sample Customer ID:", sample_ids)
customer_id_input = st.sidebar.text_input("Or Enter Custom ID:", value=selected_sample)
target_id = normalize_cust_id(customer_id_input)

fetch_btn = st.sidebar.button("Run Assessment", type="primary")

# Health Check
api_online = False
try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
    if health_res.status_code == 200:
        api_online = True
        st.sidebar.success("✅ REST API Online (port 8000)")
    else:
        st.sidebar.warning("⚠️ REST API Error. Running in Local Mode.")
except Exception:
    st.sidebar.info("ℹ️ FastAPI offline. Running via Local Parquet Engine.")

if fetch_btn or target_id:
    pred_data = None
    uw_data = None

    with st.spinner(f"Evaluating Customer ID: {target_id}..."):
        # Path A: Try REST API call first
        if api_online:
            try:
                p_res = requests.get(f"{API_BASE_URL}/api/v1/predict/customer/{target_id}")
                u_res = requests.get(f"{API_BASE_URL}/api/v1/underwrite/customer/{target_id}")
                if p_res.status_code == 200 and u_res.status_code == 200:
                    pred_data = p_res.json()
                    uw_data = u_res.json()
                else:
                    st.error(f"Customer ID '{target_id}' was not found in API backend.")
            except Exception as e:
                st.warning(f"API connection failed ({e}). Falling back to local direct engine...")

        # Path B: Fallback to Direct In-Memory Evaluation
        if pred_data is None or uw_data is None:
            df_fs, feat_cols, models, explainer = load_local_engine()
            
            if df_fs is None or target_id not in df_fs.index:
                st.error(f"❌ Customer ID '{target_id}' was not found in the Parquet Feature Store.")
            else:
                row_data = df_fs.loc[[target_id]][feat_cols].copy()
                cat_cols = row_data.select_dtypes(include=["object", "string"]).columns
                for c in cat_cols:
                    row_data[c] = row_data[c].astype("category")

                raw_p10 = float(models["p10"].predict(row_data)[0])
                raw_p50 = float(models["p50"].predict(row_data)[0])
                raw_p90 = float(models["p90"].predict(row_data)[0])
                p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])

                explanation = explainer.explain_sample(row_data) if explainer else None
                confidence = float(row_data.get("income_confidence_score", pd.Series([75.0])).values[0])

                pred_data = {
                    "customer_id": target_id,
                    "p10_conservative_income": round(p10, 2),
                    "p50_median_income": round(p50, 2),
                    "p90_optimistic_income": round(p90, 2),
                    "income_bandwidth": round(p90 - p10, 2),
                    "stability_confidence_score": round(confidence, 2),
                    "explainability": explanation
                }

                # Robust Monthly Obligations Extraction (Fallback to Debit Cash Flow if utility_debit_amt_12m is zero)
                if "avg_monthly_debit_flow" in row_data.columns and float(row_data["avg_monthly_debit_flow"].values[0]) > 0:
                    obligations = float(row_data["avg_monthly_debit_flow"].values[0])
                elif "total_debit_amt_12m" in row_data.columns and float(row_data["total_debit_amt_12m"].values[0]) > 0:
                    obligations = float(row_data["total_debit_amt_12m"].values[0]) / 12.0
                else:
                    util = float(row_data.get("utility_debit_amt_12m", pd.Series([0.0])).values[0]) / 12.0
                    obligations = util if util > 0 else float(row_data.get("total_monthly_income", pd.Series([0.0])).values[0]) * 0.45

                stability = float(row_data.get("income_stability_score", pd.Series([70.0])).values[0])
                fraud_flag = bool(row_data.get("has_fraud_flag", pd.Series([False])).values[0])

                uw_data = evaluate_local_underwriting(
                    p10, p50, p90, obligations, stability, fraud_flag
                )

        # -----------------------------------------------------------------------------
        # 5. Visual Dashboard Rendering
        # -----------------------------------------------------------------------------
        if pred_data and uw_data:
            col1, col2, col3, col4 = st.columns(4)
            
            decision = uw_data.get("decision", "N/A")
            if decision == "APPROVED":
                col1.metric("Underwriting Decision", "APPROVED 🟢", uw_data.get("decision_reason", ""))
            elif decision == "REFER":
                col1.metric("Underwriting Decision", "REFER 🟡", "Manual Review Required")
            else:
                col1.metric("Underwriting Decision", "DECLINED 🔴", uw_data.get("decision_reason", ""))

            col2.metric("Conservative Floor (P10)", f"PKR {pred_data['p10_conservative_income']:,.2f}")
            col3.metric("Expected Median (P50)", f"PKR {pred_data['p50_median_income']:,.2f}")
            col4.metric("Optimistic Ceiling (P90)", f"PKR {pred_data['p90_optimistic_income']:,.2f}")

            st.divider()

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader("📊 Quantile Income Bandwidth")
                fig_band = go.Figure()
                fig_band.add_trace(go.Bar(
                    x=["P10 (Conservative)", "P50 (Median)", "P90 (Optimistic)"],
                    y=[pred_data['p10_conservative_income'], pred_data['p50_median_income'], pred_data['p90_optimistic_income']],
                    marker_color=["#1f77b4", "#2ca02c", "#ff7f0e"]
                ))
                fig_band.update_layout(
                    title=f"Income Estimates (Bandwidth: PKR {pred_data['income_bandwidth']:,.2f})",
                    yaxis_title="Monthly Income (PKR)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_band, use_container_width=True)

            with right_col:
                st.subheader("📋 Approved Credit Limits & FOIR Impact")
                st.write(f"**Existing Obligations:** PKR {uw_data.get('existing_monthly_obligations', 0):,.2f}")
                st.write(f"**Current FOIR:** {uw_data.get('current_foir_percent', 0)}%")
                st.write(f"**Projected FOIR:** {uw_data.get('projected_foir_percent', 0)}%")
                
                st.info(f"**Max Affordable Monthly EMI:** PKR {uw_data.get('max_affordable_emi', 0):,.2f}")
                st.success(f"**Approved Term Loan Limit (24M):** PKR {uw_data.get('approved_term_loan_limit', 0):,.2f}")
                st.success(f"**Approved Credit Card Limit:** PKR {uw_data.get('approved_revolving_card_limit', 0):,.2f}")

            st.divider()

            st.subheader("🔍 TreeSHAP Feature Attribution (P50 Model Drivers)")
            shap_data = pred_data.get("explainability", {})
            if shap_data and "top_drivers" in shap_data:
                df_shap = pd.DataFrame(shap_data["top_drivers"])
                fig_shap = px.bar(
                    df_shap,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="shap_value",
                    color_continuous_scale=["#d62728", "#1f77b4"],
                    title="Top Feature Contributions to Income Estimate (PKR)",
                    labels={"shap_value": "SHAP Impact (PKR)", "feature": "Feature Name"}
                )
                fig_shap.update_layout(template="plotly_white")
                st.plotly_chart(fig_shap, use_container_width=True)
                st.dataframe(df_shap, use_container_width=True)
            else:
                st.info("No SHAP explainability metadata returned.")