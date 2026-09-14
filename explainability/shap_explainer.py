"""
Enterprise Credit AI Engine
SHAP Explainability Engine

Calculates TreeSHAP feature attributions for LightGBM Quantile Models to provide
transparent, audit-ready explanations for credit decisioning and compliance.
"""

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURE_STORE_PARQUET = BASE_DIR / "feature_store" / "customer_features.parquet"

# Business-friendly feature descriptions for professional explanations
FEATURE_DESCRIPTIONS = {
    # Account Balance Features
    "total_available_balance": "Total available balance across all accounts",
    "total_avg_monthly_balance": "Average monthly balance maintained across accounts",
    "balance_to_income_ratio": "Ratio of account balances to monthly income",
    "total_overdraft_limit": "Total overdraft limit across all accounts",
    
    # Transaction Features
    "total_credit_amt_12m": "Total credit transactions in the last 12 months",
    "total_debit_amt_12m": "Total debit transactions in the last 12 months",
    "avg_monthly_debit_flow": "Average monthly debit outflow",
    "utility_debit_amt_12m": "Utility bill payments over the last 12 months",
    "tax_amount": "Tax payments made",
    
    # Customer Demographics
    "age": "Customer age",
    "gender": "Customer gender",
    "education": "Education level",
    "marital_status": "Marital status",
    "city": "City of residence",
    "province": "Province of residence",
    "residential_status": "Residential ownership status",
    
    # Employment Features
    "employment_status": "Current employment status",
    "employment_type": "Type of employment (full-time, part-time, etc.)",
    "industry": "Industry sector of employment",
    "company_size": "Size of employing company",
    "manager_level": "Management level in organization",
    
    # Income Features
    "total_monthly_income": "Total monthly income",
    "gross_monthly_salary": "Gross monthly salary",
    "net_monthly_salary": "Net monthly salary after deductions",
    "basic_salary": "Basic salary component",
    "income_stability_score": "Score indicating income stability (0-100)",
    "income_confidence_score": "Confidence score in income verification (0-100)",
    
    # Risk Features
    "has_fraud_flag": "Fraud detection flag",
    "number_of_accounts": "Total number of bank accounts",
    "account_tenure_months": "Average account tenure in months",
}

def generate_professional_explanation(feature_name: str, shap_value: float, feature_value: any, impact_type: str) -> str:
    """
    Generate professional, human-readable explanation for SHAP contribution.
    
    Args:
        feature_name: Name of the feature
        shap_value: SHAP value (positive = increases income, negative = decreases income)
        feature_value: Actual value of the feature for this customer
        impact_type: "positive" or "negative"
    
    Returns:
        Professional explanation string
    """
    description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace("_", " ").title())
    
    # Format the feature value for readability
    if isinstance(feature_value, (int, float)):
        if abs(feature_value) >= 1000000:
            formatted_value = f"PKR {feature_value/1000000:.2f}M"
        elif abs(feature_value) >= 1000:
            formatted_value = f"PKR {feature_value/1000:.2f}K"
        else:
            formatted_value = f"PKR {feature_value:.2f}"
    else:
        formatted_value = str(feature_value)
    
    # Format SHAP value
    shap_formatted = f"PKR {abs(shap_value):,.2f}"
    
    if impact_type == "positive":
        direction = "increases"
        business_impact = "supports higher income estimation"
    else:
        direction = "decreases"
        business_impact = "indicates lower income potential"
    
    explanation = (
        f"{description} (Value: {formatted_value}) {direction} predicted income by {shap_formatted}. "
        f"This factor {business_impact} based on historical patterns."
    )
    
    return explanation

class IncomeSHAPExplainer:
    def __init__(self):
        self.model_path = MODELS_DIR / "lgb_income_p50.joblib"
        self.feature_cols_path = MODELS_DIR / "feature_columns.joblib"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train models first.")
            
        self.model = joblib.load(self.model_path)
        self.feature_cols = joblib.load(self.feature_cols_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_sample(self, X_sample: pd.DataFrame, top_k: int = 5) -> dict:
        """
        Computes SHAP values for a given sample DataFrame row.
        Returns top positive and negative contributing features with professional explanations.
        """
        # Ensure categorical formatting alignment
        X_formatted = X_sample[self.feature_cols].copy()
        cat_cols = X_formatted.select_dtypes(include=["object", "string"]).columns
        for c in cat_cols:
            X_formatted[c] = X_formatted[c].astype("category")

        shap_values = self.explainer.shap_values(X_formatted)
        
        # Extract first row SHAP values
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        base_value = float(self.explainer.expected_value)

        feature_contributions = []
        for feat_name, val in zip(self.feature_cols, vals):
            impact_type = "positive" if val > 0 else "negative"
            feature_value = X_formatted[feat_name].values[0]
            
            contribution = {
                "feature": feat_name,
                "feature_description": FEATURE_DESCRIPTIONS.get(feat_name, feat_name.replace("_", " ").title()),
                "shap_value": float(val),
                "shap_value_formatted": f"PKR {abs(val):,.2f}",
                "feature_value": str(feature_value),
                "feature_value_formatted": self._format_feature_value(feature_value),
                "impact_type": impact_type,
                "professional_explanation": generate_professional_explanation(
                    feat_name, val, feature_value, impact_type
                )
            }
            feature_contributions.append(contribution)

        # Sort by magnitude of contribution
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value_pkr": round(base_value, 2),
            "base_value_formatted": f"PKR {base_value:,.2f}",
            "model_baseline_explanation": f"The model's baseline income prediction is PKR {base_value:,.2f} before considering individual customer characteristics.",
            "top_drivers": feature_contributions[:top_k],
            "summary_explanation": self._generate_summary_explanation(feature_contributions[:top_k])
        }
    
    def _format_feature_value(self, value) -> str:
        """Format feature values for human readability."""
        if isinstance(value, (int, float)):
            if abs(value) >= 1000000:
                return f"PKR {value/1000000:.2f}M"
            elif abs(value) >= 1000:
                return f"PKR {value/1000:.2f}K"
            else:
                return f"PKR {value:.2f}"
        return str(value)
    
    def _generate_summary_explanation(self, top_contributions: list) -> str:
        """Generate a professional summary of the top contributing factors."""
        if not top_contributions:
            return "No significant contributing factors identified."
        
        positive_drivers = [c for c in top_contributions if c["impact_type"] == "positive"]
        negative_drivers = [c for c in top_contributions if c["impact_type"] == "negative"]
        
        summary_parts = []
        
        if positive_drivers:
            top_positive = positive_drivers[0]
            summary_parts.append(
                f"The strongest positive factor is {top_positive['feature_description'].lower()} "
                f"({top_positive['feature_value_formatted']}) which increases the income estimate by {top_positive['shap_value_formatted']}."
            )
        
        if negative_drivers:
            top_negative = negative_drivers[0]
            summary_parts.append(
                f"The main limiting factor is {top_negative['feature_description'].lower()} "
                f"({top_negative['feature_value_formatted']}) which reduces the income estimate by {top_negative['shap_value_formatted']}."
            )
        
        return " ".join(summary_parts) 