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
        Returns top positive and negative contributing features.
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
            feature_contributions.append({
                "feature": feat_name,
                "shap_value": float(val),
                "feature_value": str(X_formatted[feat_name].values[0])
            })

        # Sort by magnitude of contribution
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value_pkr": round(base_value, 2),
            "top_drivers": feature_contributions[:top_k]
        } 