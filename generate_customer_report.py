"""
Enterprise Credit AI Engine
Professional Individual Customer Report Generator

Generates comprehensive, professional credit assessment reports for individual customers
explaining approval/rejection decisions based on model results and underwriting criteria.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from engine.underwriting_rules import CreditUnderwritingEngine
from explainability.shap_explainer import IncomeSHAPExplainer

# Paths
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

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

def generate_professional_customer_report(customer_id: str) -> dict:
    """
    Generate a comprehensive professional credit assessment report for a single customer.
    
    Args:
        customer_id: Customer ID (any format - will be normalized)
    
    Returns:
        dict: Complete professional report with all assessment details
    """
    
    # Normalize customer ID
    target_id = normalize_customer_id(customer_id)
    
    # Load feature store
    df_features = pd.read_parquet(FEATURE_STORE_PARQUET)
    
    # Normalize customer IDs in feature store
    df_features["customer_id"] = df_features["customer_id"].apply(normalize_customer_id)
    df_features.set_index("customer_id", inplace=True)
    
    # Check if customer exists
    if target_id not in df_features.index:
        return {
            "success": False,
            "error": f"Customer ID '{target_id}' not found in the system",
            "suggestion": "Please verify the customer ID or check available customers"
        }
    
    # Load models and feature info
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    
    # Load categorical information if available
    categorical_info = None
    if (MODELS_DIR / "categorical_info.joblib").exists():
        categorical_info = joblib.load(MODELS_DIR / "categorical_info.joblib")
    
    # Load LightGBM Quantile Models
    models = {
        "p10": joblib.load(MODELS_DIR / "lgb_income_p10.joblib"),
        "p50": joblib.load(MODELS_DIR / "lgb_income_p50.joblib"),
        "p90": joblib.load(MODELS_DIR / "lgb_income_p90.joblib")
    }
    
    # Get customer data
    customer_data = df_features.loc[[target_id]].copy()
    row_data = customer_data[feature_cols].copy()
    
    # Apply categorical handling
    if categorical_info and "categorical_columns" in categorical_info:
        cat_cols = categorical_info["categorical_columns"]
        for c in cat_cols:
            if c in row_data.columns:
                row_data[c] = row_data[c].astype("category")
    else:
        cat_cols = row_data.select_dtypes(include=["object", "string"]).columns
        for c in cat_cols:
            row_data[c] = row_data[c].astype("category")
    
    # Generate income predictions
    raw_p10 = float(models["p10"].predict(row_data)[0])
    raw_p50 = float(models["p50"].predict(row_data)[0])
    raw_p90 = float(models["p90"].predict(row_data)[0])
    p10, p50, p90 = np.sort([raw_p10, raw_p50, raw_p90])
    
    # Initialize underwriting engine
    engine = CreditUnderwritingEngine()
    
    # Calculate obligations (consistent with dashboard)
    obligations = float(customer_data.get("utility_debit_amt_12m", pd.Series([0.0])).values[0]) / 12.0
    
    stability = float(customer_data.get("income_stability_score", pd.Series([70.0])).values[0])
    fraud_flag = bool(customer_data.get("has_fraud_flag", pd.Series([False])).values[0])
    
    # Run underwriting assessment
    assessment = engine.evaluate_credit(
        customer_id=target_id,
        p10_income=p10,
        p50_income=p50,
        p90_income=p90,
        existing_monthly_obligations=obligations,
        income_stability_score=stability,
        has_fraud_flag=fraud_flag
    )
    
    # Generate SHAP explanation
    shap_explanation = None
    try:
        explainer = IncomeSHAPExplainer()
        shap_explanation = explainer.explain_sample(row_data)
    except Exception as e:
        print(f"SHAP explanation error: {e}")
    
    # Extract customer demographic information
    demographics = {}
    demo_fields = ["age", "gender", "education", "marital_status", "city", "province", 
                   "residential_status", "employment_status", "employment_type", "industry"]
    for field in demo_fields:
        if field in customer_data.columns:
            value = customer_data[field].values[0]
            demographics[field] = str(value) if not pd.isna(value) else "N/A"
    
    # Build professional report
    report = {
        "success": True,
        "report_metadata": {
            "customer_id": target_id,
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_type": "Individual Credit Assessment Report",
            "model_version": "LightGBM Quantile v1.0"
        },
        "customer_profile": {
            "demographics": demographics,
            "income_assessment": {
                "conservative_income_p10": f"PKR {p10:,.2f}",
                "median_income_p50": f"PKR {p50:,.2f}",
                "optimistic_income_p90": f"PKR {p90:,.2f}",
                "income_bandwidth": f"PKR {p90 - p10:,.2f}",
                "income_bandwidth_percentage": f"{((p90 - p10) / p50 * 100):.1f}%",
                "underwriting_income": f"PKR {p10:,.2f} (P10 Conservative)"
            },
            "financial_commitments": {
                "existing_monthly_obligations": f"PKR {obligations:,.2f}",
                "current_foir_ratio": f"{assessment['current_foir_percent']:.2f}%",
                "income_stability_score": f"{stability:.1f}/100",
                "fraud_risk_flag": "HIGH RISK - AUTOMATIC DECLINE" if fraud_flag else "No fraud indicators"
            }
        },
        "underwriting_decision": {
            "final_decision": assessment["decision"],
            "decision_reason": assessment["decision_reason"],
            "decision_rationale": generate_decision_rationale(assessment, p10, obligations, stability),
            "effective_date": datetime.now().strftime("%Y-%m-%d")
        },
        "credit_limits": {
            "status": "APPROVED - Credit Limits Authorized" if assessment["decision"] == "APPROVED" else "NOT AUTHORIZED - Application " + assessment["decision"],
            "maximum_affordable_emi": f"PKR {assessment['max_affordable_emi']:,.2f}" if assessment["decision"] == "APPROVED" else "PKR 0.00",
            "approved_term_loan_limit": f"PKR {assessment['approved_term_loan_limit']:,.2f}" if assessment["decision"] == "APPROVED" else "PKR 0.00",
            "approved_revolving_card_limit": f"PKR {assessment['approved_revolving_card_limit']:,.2f}" if assessment["decision"] == "APPROVED" else "PKR 0.00",
            "loan_terms": {
                "tenure": "24 months",
                "interest_rate": "24% per annum",
                "repayment_structure": "Equal Monthly Installments (EMI)"
            } if assessment["decision"] == "APPROVED" else {}
        },
        "risk_factors": {
            "foir_analysis": {
                "current_ratio": f"{assessment['current_foir_percent']:.2f}%",
                "regulatory_limit": "45%",
                "compliance_status": "COMPLIANT" if assessment['current_foir_percent'] <= 45 else "NON-COMPLIANT",
                "analysis": f"Customer's current debt-to-income ratio is {assessment['current_foir_percent']:.2f}%, which is {'within' if assessment['current_foir_percent'] <= 45 else 'above'} the regulatory limit of 45%."
            },
            "income_stability_analysis": {
                "stability_score": f"{stability:.1f}/100",
                "risk_level": "LOW RISK" if stability >= 70 else "MODERATE RISK" if stability >= 40 else "HIGH RISK",
                "analysis": f"Income stability score of {stability:.1f}/100 indicates {'strong' if stability >= 70 else 'moderate' if stability >= 40 else 'concerning'} income consistency and reliability."
            },
            "disposable_income_analysis": {
                "monthly_disposable_income": f"PKR {max(0, p10 * 0.45 - obligations):,.2f}",
                "stress_buffer_applied": "12% of disposable income",
                "analysis": f"After applying regulatory 45% FOIR limit and 12% stress buffer, customer has {'sufficient' if assessment['max_affordable_emi'] >= 4000 else 'insufficient'} disposable income for loan repayment."
            }
        },
        "model_explainability": {
            "prediction_confidence": "HIGH" if stability >= 70 else "MODERATE" if stability >= 40 else "LOW",
            "key_drivers": shap_explanation["top_drivers"] if shap_explanation else [],
            "summary_explanation": shap_explanation["summary_explanation"] if shap_explanation else "SHAP explanation not available",
            "technical_note": "Income predictions generated using LightGBM Quantile Regression (P10, P50, P90) for conservative, median, and optimistic income estimates."
        } if shap_explanation else {},
        "regulatory_compliance": {
            "foir_compliance": assessment['current_foir_percent'] <= 45,
            "affordability_compliance": assessment['max_affordable_emi'] >= 4000,
            "fraud_clearance": not fraud_flag,
            "overall_compliance_status": "COMPLIANT" if (assessment['current_foir_percent'] <= 45 and assessment['max_affordable_emi'] >= 4000 and not fraud_flag) else "NON-COMPLIANT"
        },
        "next_steps": generate_next_steps(assessment),
        "disclaimer": generate_disclaimer()
    }
    
    return report

def generate_decision_rationale(assessment, p10, obligations, stability) -> str:
    """Generate professional decision rationale."""
    decision = assessment["decision"]
    reason = assessment["decision_reason"]
    
    if decision == "APPROVED":
        return (
            f"Customer application has been APPROVED based on comprehensive credit assessment. "
            f"Conservative income estimate of PKR {p10:,.2f} demonstrates strong repayment capacity. "
            f"Current FOIR of {assessment['current_foir_percent']:.2f}% is well within the 45% regulatory limit. "
            f"Income stability score of {stability:.1f}/100 indicates reliable income source. "
            f"All regulatory requirements met satisfactorily."
        )
    elif decision == "DECLINED":
        if "FOIR" in reason:
            return (
                f"Application DECLINED due to excessive debt burden. "
                f"Current FOIR of {assessment['current_foir_percent']:.2f}% exceeds the regulatory maximum of 45%. "
                f"This indicates customer has insufficient disposable income to accommodate additional credit obligations."
            )
        elif "DISPOSABLE" in reason:
            return (
                f"Application DECLINED due to insufficient net disposable income. "
                f"After applying regulatory 45% FOIR limit and 12% stress buffer, "
                f"customer's maximum affordable EMI of PKR {assessment['max_affordable_emi']:,.2f} is below the minimum threshold of PKR 4,000."
            )
        elif "FRAUD" in reason:
            return (
                f"Application DECLINED due to fraud/AML risk indicators. "
                f"Customer has been flagged for potential fraudulent activity or AML concerns. "
                f"This requires immediate risk mitigation and cannot proceed through automated underwriting."
            )
        else:
            return f"Application DECLINED: {reason}"
    else:  # REFER
        return (
            f"Application REFERRED for manual review. "
            f"Income stability score of {stability:.1f}/100 falls below the automated approval threshold of 45. "
            f"While financial capacity may exist, income consistency concerns require human assessment."
        )

def generate_next_steps(assessment) -> list:
    """Generate appropriate next steps based on decision."""
    decision = assessment["decision"]
    
    if decision == "APPROVED":
        return [
            "Customer should be contacted with approval notification",
            "Credit agreement documentation to be prepared",
            "Loan disbursement process to be initiated",
            "Regular repayment monitoring to be established",
            "Periodic credit review schedule to be set"
        ]
    elif decision == "DECLINED":
        return [
            "Customer to be informed of decline decision with specific reasons",
            "Provide information on re-application eligibility and timing",
            "Suggest financial counseling if appropriate",
            "Document decline reasons for regulatory compliance",
            "Maintain decline record for future reference"
        ]
    else:  # REFER
        return [
            "Application to be forwarded to senior credit officer",
            "Manual review of income documentation required",
            "Additional verification of income sources needed",
            "Final decision to be made within 5 business days",
            "Customer to be informed of review process timeline"
        ]

def generate_disclaimer() -> str:
    """Generate professional disclaimer."""
    return (
        "This report is generated by automated credit assessment systems and is provided for informational purposes only. "
        "Final credit decisions are subject to human review and approval in accordance with institutional policies "
        "and regulatory requirements. The information contained herein is confidential and intended solely for "
        "authorized personnel. Model predictions are based on available data and historical patterns, and may "
        "not account for all relevant factors. This report does not constitute a binding credit offer or guarantee."
    )

def save_report_to_file(report: dict, filename: str = None) -> str:
    """Save report to text file."""
    if filename is None:
        customer_id = report["report_metadata"]["customer_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"credit_assessment_{customer_id}_{timestamp}.txt"
    
    filepath = REPORTS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*15 + "PROFESSIONAL CREDIT ASSESSMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Report Metadata
        f.write("REPORT METADATA\n")
        f.write("-"*40 + "\n")
        for key, value in report["report_metadata"].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Customer Profile
        f.write("CUSTOMER PROFILE\n")
        f.write("-"*40 + "\n")
        f.write("Demographics:\n")
        for key, value in report["customer_profile"]["demographics"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("Income Assessment:\n")
        for key, value in report["customer_profile"]["income_assessment"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("Financial Commitments:\n")
        for key, value in report["customer_profile"]["financial_commitments"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Underwriting Decision
        f.write("UNDERWRITING DECISION\n")
        f.write("-"*40 + "\n")
        f.write(f"Final Decision: {report['underwriting_decision']['final_decision']}\n")
        f.write(f"Decision Reason: {report['underwriting_decision']['decision_reason']}\n")
        f.write(f"Decision Rationale: {report['underwriting_decision']['decision_rationale']}\n")
        f.write(f"Effective Date: {report['underwriting_decision']['effective_date']}\n")
        f.write("\n")
        
        # Credit Limits
        f.write("CREDIT LIMITS\n")
        f.write("-"*40 + "\n")
        f.write(f"Status: {report['credit_limits']['status']}\n")
        f.write(f"Maximum Affordable EMI: {report['credit_limits']['maximum_affordable_emi']}\n")
        f.write(f"Approved Term Loan Limit: {report['credit_limits']['approved_term_loan_limit']}\n")
        f.write(f"Approved Revolving Card Limit: {report['credit_limits']['approved_revolving_card_limit']}\n")
        if report['credit_limits']['loan_terms']:
            f.write("Loan Terms:\n")
            for key, value in report['credit_limits']['loan_terms'].items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Risk Factors
        f.write("RISK FACTORS ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write("FOIR Analysis:\n")
        for key, value in report["risk_factors"]["foir_analysis"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("Income Stability Analysis:\n")
        for key, value in report["risk_factors"]["income_stability_analysis"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("Disposable Income Analysis:\n")
        for key, value in report["risk_factors"]["disposable_income_analysis"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Model Explainability
        if report.get("model_explainability"):
            f.write("MODEL EXPLAINABILITY\n")
            f.write("-"*40 + "\n")
            f.write(f"Prediction Confidence: {report['model_explainability']['prediction_confidence']}\n")
            f.write(f"Summary: {report['model_explainability']['summary_explanation']}\n")
            f.write(f"Technical Note: {report['model_explainability']['technical_note']}\n")
            f.write("\n")
        
        # Regulatory Compliance
        f.write("REGULATORY COMPLIANCE\n")
        f.write("-"*40 + "\n")
        for key, value in report["regulatory_compliance"].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Next Steps
        f.write("RECOMMENDED NEXT STEPS\n")
        f.write("-"*40 + "\n")
        for i, step in enumerate(report["next_steps"], 1):
            f.write(f"{i}. {step}\n")
        f.write("\n")
        
        # Disclaimer
        f.write("DISCLAIMER\n")
        f.write("-"*40 + "\n")
        f.write(report["disclaimer"] + "\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(" "*20 + "END OF REPORT\n")
        f.write("="*80 + "\n")
    
    return str(filepath)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        customer_id = sys.argv[1]
    else:
        customer_id = "CUST0000001"
    
    print(f"Generating professional credit assessment report for customer: {customer_id}")
    print("-" * 60)
    
    report = generate_professional_customer_report(customer_id)
    
    if report["success"]:
        print("Report generated successfully!")
        print(f"Decision: {report['underwriting_decision']['final_decision']}")
        print(f"Reason: {report['underwriting_decision']['decision_reason']}")
        
        # Save to file
        filepath = save_report_to_file(report)
        print(f"\nReport saved to: {filepath}")
        
        # Print key summary
        print("\n" + "="*60)
        print("REPORT SUMMARY")
        print("="*60)
        print(f"Customer ID: {report['report_metadata']['customer_id']}")
        print(f"Decision: {report['underwriting_decision']['final_decision']}")
        print(f"Conservative Income: {report['customer_profile']['income_assessment']['conservative_income_p10']}")
        print(f"Current FOIR: {report['customer_profile']['financial_commitments']['current_foir_ratio']}")
        print(f"Approved Loan Limit: {report['credit_limits']['approved_term_loan_limit']}")
    else:
        print(f"Error: {report['error']}")
        print(f"Suggestion: {report['suggestion']}")