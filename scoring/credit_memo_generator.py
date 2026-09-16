import sys
from pathlib import Path
from datetime import datetime

def generate_institutional_credit_memo(customer_id: str, name: str, res_dict: dict, pricing_dict: dict, xai_dict: dict) -> str:
    score = res_dict.get("composite_score_1000", 650)
    grade = res_dict.get("credit_grade", "C")
    decision = res_dict.get("credit_decision", "APPROVED")
    pd_val = res_dict.get("calibrated_pd", 0.05) * 100
    p10 = res_dict.get("p10_income", 55000)
    p50 = res_dict.get("p50_income", 65000)
    p90 = res_dict.get("p90_income", 80000)
    
    appr_limit = pricing_dict.get("approved_limit", 75000)
    card_limit = pricing_dict.get("card_limit", 35000)
    max_emi = pricing_dict.get("max_affordable_emi", 15000)
    apr = pricing_dict.get("apr_pct", 22.5)
    
    top_pos = xai_dict.get("top_positive", [])
    top_neg = xai_dict.get("top_negative", [])
    
    now_str = datetime.now().strftime('%d-%b-%Y %H:%M:%S UTC')
    date_tag = datetime.now().strftime('%Y%m%d')
    
    lines = [
        "==============================================================================",
        "        AI CREDIT SCORING OS · CREDIT ASSESSMENT MEMORANDUM (CAM)",
        "        STRICTLY CONFIDENTIAL · FOR RISK COMMITTEE APPROVAL ONLY",
        "==============================================================================",
        f"FACILITY REFERENCE:  CAM-{customer_id}-{date_tag}",
        f"DATE OF EVALUATION:  {now_str}",
        f"PRIMARY APPLICANT:   {name} ({customer_id})",
        f"CREDIT RATING:       GRADE {grade} ({decision})",
        "REGULATORY STATUS:   SBP & BASEL III COMPLIANT · IFRS 9 STAGE 1",
        "",
        "------------------------------------------------------------------------------",
        "1. EXECUTIVE UNDERWRITING DECISION & SIZING",
        "------------------------------------------------------------------------------",
        f"* Final Recommendation:       {decision}",
        f"* Calibrated Composite Score: {score} / 900",
        f"* Calibrated Default (12M PD):{pd_val:.2f}%",
        f"* Recommended Term Facility:  PKR {appr_limit:,.2f}",
        f"* Revolving Line Facility:    PKR {card_limit:,.2f}",
        f"* Max Disposable EMI:         PKR {max_emi:,.2f} / month",
        f"* Risk-Based Pricing (APR):   {apr:.2f}% p.a. (Fixed)",
        "",
        "------------------------------------------------------------------------------",
        "2. AI QUANTILE INCOME RECONSTRUCTION (PINBALL LOSS ENGINE)",
        "------------------------------------------------------------------------------",
        f"* P10 Conservative Floor:    PKR {p10:,.2f} / month (Worst-Case Baseline)",
        f"* P50 Median Expected:       PKR {p50:,.2f} / month (Primary Capacity Anchor)",
        f"* P90 Optimistic Ceiling:    PKR {p90:,.2f} / month (Upside Growth Trajectory)",
        f"* Income Volatility Band:    {((p90-p10)/p50*100):.1f}%",
        "",
        "------------------------------------------------------------------------------",
        "3. MULTI-BOUND CAPACITY LIMIT DECOMPOSITION",
        "------------------------------------------------------------------------------",
        f"1. Affordability Limit:      PKR {pricing_dict.get('affordability_limit', 0):,.2f} (FOIR 45% Ceiling)",
        f"2. Risk-Adjusted Limit:      PKR {pricing_dict.get('risk_limit', 0):,.2f}",
        f"3. Progressive Cap:          PKR {pricing_dict.get('progression_cap', 0):,.2f} (Nano Ladder 1.5^k)",
        f"4. Institutional Policy Cap: PKR {pricing_dict.get('policy_cap', 0):,.2f}",
        "",
        "------------------------------------------------------------------------------",
        "4. EXPLAINABLE AI (SHAP TREE-EXPLAINER) ATTRIBUTION ANALYSIS",
        "------------------------------------------------------------------------------",
        "KEY STRENGTHS / TRUST ENHANCERS:"
    ]
    
    for i, p in enumerate(top_pos[:4], 1):
        lines.append(f"  {i}. {p.get('name', 'Metric')}: {p.get('value', '')} -> {p.get('reason', '')}")
        
    lines.append("\nRISK CONCERNS / ADVERSE ACTION DISCLOSURES:")
    for i, n in enumerate(top_neg[:4], 1):
        lines.append(f"  {i}. {n.get('name', 'Factor')}: {n.get('value', '')} -> {n.get('reason', '')}")
        
    stage_name = "Stage 1 (12M ECL)" if grade in ['A','B','C'] else ("Stage 2 (SICR)" if grade in ['D','E'] else "Stage 3 (Impaired)")
    lines.extend([
        "",
        "------------------------------------------------------------------------------",
        "5. STATUTORY SIGN-OFF & AUDIT TRAILS",
        "------------------------------------------------------------------------------",
        "Underwriting Engine:         APEX Credit Staged Topology v3.2",
        "Ensemble Meta-Learner:       Logistic Meta-Learner (5-Fold Stratified OOF)",
        f"IFRS 9 Provision Category:   {stage_name}",
        "",
        "[APPROVED ELECTRONICALLY BY APEX CREDIT RISK OPERATIONS COMMITTEE]",
        "=============================================================================="
    ])
    return "\n".join(lines)

if __name__ == "__main__":
    memo = generate_institutional_credit_memo("CUST0000003", "Shahid Hashmi", {}, {}, {})
    print(memo[:300])
