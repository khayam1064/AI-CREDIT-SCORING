import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from scoring import composite_scorer
from scoring import policy_engine
from scoring import shap_explainer
from scoring import monitoring_service

app = FastAPI(
    title="APEX CREDIT OS · Commercial REST API",
    description="Tier-1 Enterprise Model-Serving & Underwriting Gateway (Basel III / IFRS 9 / SBP Compliant)",
    version="3.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load feature store and pre-scores into memory for instant millisecond lookups
print("Loading in-memory registry for API gateway...")
DF_FS = pd.read_parquet(BASE_DIR / "feature_store" / "customer_features_v2.parquet")
if "customer_id" in DF_FS.columns:
    DF_FS["customer_id"] = DF_FS["customer_id"].astype(str)
    DF_FS.set_index("customer_id", inplace=True)

DF_SCORES = pd.read_parquet(BASE_DIR / "data" / "processed" / "credit_scores_250k.parquet")
if "customer_id" in DF_SCORES.columns:
    DF_SCORES["customer_id"] = DF_SCORES["customer_id"].astype(str)
    DF_SCORES.set_index("customer_id", inplace=True)
print("API Gateway ready with 250,000 indexed profiles.")

class ApplicantPayload(BaseModel):
    customer_id: Optional[str] = "NEW_APP_001"
    age: Optional[int] = 32
    gender: Optional[str] = "male"
    education_level: Optional[str] = "bachelor"
    total_monthly_income: Optional[float] = 120000.0
    verified_income: Optional[float] = 110000.0
    total_monthly_credit: Optional[float] = 115000.0
    total_monthly_debit: Optional[float] = 85000.0
    total_avg_monthly_balance: Optional[float] = 45000.0
    overdraft_utilization_rate: Optional[float] = 0.15
    savings_ratio: Optional[float] = 0.25
    foir_pct: Optional[float] = 18.5
    ldr_on_time_payment_rate: Optional[float] = 1.00
    ldr_prior_loans_count: Optional[int] = 2
    dev_is_rooted: Optional[bool] = False
    dev_emulator_detected: Optional[bool] = False
    app_applications_same_device_7d: Optional[int] = 1
    features_override: Optional[Dict[str, Any]] = None

class UnderwriteResponse(BaseModel):
    status: str
    execution_time_ms: float
    customer_id: str
    credit_decision: str
    composite_score: int
    credit_grade: str
    calibrated_pd_pct: float
    income_quantiles: Dict[str, float]
    credit_limits: Dict[str, Any]
    pricing: Dict[str, Any]
    adverse_action_codes: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": "APEX Credit Model Server",
        "engine_version": "3.2.0-Production",
        "standard_compliance": ["IFRS 9", "Basel III", "SBP Fair Lending"],
        "indexed_customers": len(DF_FS)
    }

@app.get("/api/v1/customers/{customer_id}")
def get_customer_decision(customer_id: str):
    t0 = time.time()
    cid = customer_id.strip().upper()
    if not cid.startswith("CUST"):
        try:
            cid = f"CUST{int(cid):07d}"
        except:
            pass

    if cid not in DF_FS.index:
        raise HTTPException(status_code=404, detail=f"Customer ID {cid} not found in credit registry")

    row_fs = DF_FS.loc[[cid]]
    live_scores = composite_scorer.compute_scores(row_fs)
    r = live_scores.iloc[0]

    p10 = float(r.get("p10_income", 50000.0))
    p50 = float(r.get("p50_income", 65000.0))
    p90 = float(r.get("p90_income", 80000.0))
    oblg = float(DF_FS.loc[cid].get("utility_debit_amt_12m", 0) or 0) / 12.0

    pricing = policy_engine.calculate_limits_and_pricing(
        grade=str(r["credit_grade"]),
        pd_cal=float(r["calibrated_pd"]),
        p10_income=p10,
        p50_income=p50,
        existing_obligations=oblg,
        prior_loans_count=int(DF_FS.loc[cid].get("ldr_prior_loans_count", 0) or 0)
    )

    sub_dict = {k: r[k] for k in r.index if k.startswith("p0") or k.startswith("p1")}
    xai = shap_explainer.explain_prediction(DF_FS.loc[cid], sub_dict)

    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "SUCCESS",
        "execution_time_ms": latency_ms,
        "customer_id": cid,
        "credit_decision": str(r["credit_decision"]),
        "composite_score": int(r["composite_score_1000"]),
        "credit_grade": str(r["credit_grade"]),
        "calibrated_pd_pct": round(float(r["calibrated_pd"]) * 100, 2),
        "income_quantiles": {
            "p10_conservative_pkr": p10,
            "p50_median_expected_pkr": p50,
            "p90_optimistic_pkr": p90
        },
        "credit_limits": {
            "approved_facility_limit": pricing["approved_limit"],
            "revolving_card_limit": pricing["card_limit"],
            "max_affordable_emi": pricing["max_affordable_emi"],
            "binding_constraint": "Progression Ladder" if pricing["approved_limit"] == pricing["progression_cap"] else "Affordability/FOIR"
        },
        "pricing": {
            "fixed_apr_pct": pricing["apr_pct"],
            "cost_breakdown": pricing["pricing_breakdown"]
        },
        "adverse_action_codes": xai["top_negative"],
        "positive_trust_drivers": xai["top_positive"]
    }

@app.post("/api/v1/score/underwrite")
def live_underwrite_application(payload: ApplicantPayload):
    t0 = time.time()
    # Build feature row
    data = payload.dict()
    cid = data.pop("customer_id", "NEW_APPLICANT")
    overrides = data.pop("features_override", {}) or {}
    data.update(overrides)

    df_applicant = pd.DataFrame([data], index=[cid])
    live_scores = composite_scorer.compute_scores(df_applicant)
    r = live_scores.iloc[0]

    p10 = float(r.get("p10_income", data.get("total_monthly_income", 60000.0) * 0.8))
    p50 = float(r.get("p50_income", data.get("total_monthly_income", 60000.0)))
    p90 = float(r.get("p90_income", data.get("total_monthly_income", 60000.0) * 1.25))

    pricing = policy_engine.calculate_limits_and_pricing(
        grade=str(r["credit_grade"]),
        pd_cal=float(r["calibrated_pd"]),
        p10_income=p10,
        p50_income=p50,
        existing_obligations=float(data.get("existing_obligations", 5000.0)),
        prior_loans_count=int(data.get("ldr_prior_loans_count", 0))
    )

    sub_dict = {k: r[k] for k in r.index if k.startswith("p0") or k.startswith("p1")}
    xai = shap_explainer.explain_prediction(pd.Series(data), sub_dict)

    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "SUCCESS",
        "execution_time_ms": latency_ms,
        "customer_id": cid,
        "credit_decision": str(r["credit_decision"]),
        "composite_score": int(r["composite_score_1000"]),
        "credit_grade": str(r["credit_grade"]),
        "calibrated_pd_pct": round(float(r["calibrated_pd"]) * 100, 2),
        "income_quantiles": {
            "p10_conservative_pkr": p10,
            "p50_median_expected_pkr": p50,
            "p90_optimistic_pkr": p90
        },
        "credit_limits": {
            "approved_facility_limit": pricing["approved_limit"],
            "revolving_card_limit": pricing["card_limit"],
            "max_affordable_emi": pricing["max_affordable_emi"]
        },
        "pricing": {
            "fixed_apr_pct": pricing["apr_pct"],
            "cost_breakdown": pricing["pricing_breakdown"]
        },
        "adverse_action_codes": xai["top_negative"],
        "positive_trust_drivers": xai["top_positive"]
    }

@app.get("/api/v1/portfolio/ifrs9")
def get_portfolio_ifrs9():
    res = policy_engine.calculate_portfolio_ifrs9(DF_SCORES, df_features=DF_FS)
    return {
        "status": "SUCCESS",
        "portfolio_summary": {
            "total_accounts": len(DF_SCORES),
            "total_portfolio_exposure_ead_pkr": res["total_portfolio_exposure"],
            "total_weighted_ecl_provision_pkr": res["total_ecl_provision"],
            "portfolio_ecl_coverage_ratio_pct": res["ecl_coverage_ratio_pct"]
        },
        "macroeconomic_scenarios": res["macro_scenarios"],
        "staging_breakdown": res["stages"]
    }

@app.get("/api/v1/portfolio/drift-monitoring")
def get_drift_metrics():
    res = monitoring_service.monitor_portfolio_stability(DF_SCORES, df_features=DF_FS)
    return {
        "status": "SUCCESS",
        "monitoring_report": res
    }

from scoring import disbursement_ledger

class DisbursePayload(BaseModel):
    customer_id: str
    amount: float
    tenor_months: int
    e_sign_name: str

@app.post("/api/v1/loans/disburse")
def disburse_loan(payload: DisbursePayload):
    cid = payload.customer_id.strip().upper()
    if cid not in DF_FS.index:
        raise HTTPException(status_code=404, detail=f"Customer {cid} not registered.")
        
    row_fs = DF_FS.loc[[cid]]
    live_scores = composite_scorer.compute_scores(row_fs)
    r = live_scores.iloc[0]
    
    if r["credit_decision"] == "DECLINED":
        raise HTTPException(status_code=400, detail="Cannot disburse funds to a declined applicant.")
        
    p10 = float(r.get("p10_income", 50000.0))
    p50 = float(r.get("p50_income", 65000.0))
    oblg = float(DF_FS.loc[cid].get("utility_debit_amt_12m", 0) or 0) / 12.0
    
    pricing = policy_engine.calculate_limits_and_pricing(
        grade=str(r["credit_grade"]),
        pd_cal=float(r["calibrated_pd"]),
        p10_income=p10,
        p50_income=p50,
        existing_obligations=oblg
    )
    
    if payload.amount > pricing["approved_limit"]:
        raise HTTPException(status_code=400, detail=f"Requested amount PKR {payload.amount:,.0f} exceeds maximum approved facility limit of PKR {pricing['approved_limit']:,.0f}.")
        
    r_mo = (pricing["apr_pct"] / 100.0) / 12.0
    n_mo = payload.tenor_months
    emi = round(payload.amount * (r_mo * (1 + r_mo)**n_mo) / ((1 + r_mo)**n_mo - 1), 2)
    tot = round(emi * n_mo, 2)
    
    name = str(DF_FS.loc[cid].get("full_name", "Valued Customer"))
    
    rec = disbursement_ledger.record_disbursement(
        customer_id=cid,
        full_name=name,
        amount=payload.amount,
        tenor_months=payload.tenor_months,
        apr_pct=pricing["apr_pct"],
        monthly_emi=emi,
        total_repayable=tot,
        e_sign_name=payload.e_sign_name,
        credit_grade=str(r["credit_grade"]),
        calibrated_pd=float(r["calibrated_pd"])
    )
    
    return {
        "status": "DISBURSED_SUCCESS",
        "disbursement_details": rec
    }

@app.get("/api/v1/loans/ledger")
def get_loan_ledger():
    df = disbursement_ledger.get_disbursed_loans(50)
    return {
        "status": "SUCCESS",
        "total_disbursed_loans": len(df),
        "ledger": df.to_dict(orient="records") if not df.empty else []
    }
