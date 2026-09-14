with open("api/server.py", "r", encoding="utf-8") as f:
    text = f.read()

endpoint_code = """
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
"""

if "/api/v1/loans/disburse" not in text:
    text += endpoint_code
    with open("api/server.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Added /api/v1/loans/disburse and /api/v1/loans/ledger to FastAPI server!")
else:
    print("Disburse endpoints already present.")
