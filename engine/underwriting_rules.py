"""
Enterprise Credit AI Engine
Underwriting & Credit Decisioning Engine

Evaluates P10 conservative income estimates against regulatory FOIR/DTI caps,
existing financial commitments, and risk thresholds to compute:
- Fixed Obligation to Income Ratio (FOIR)
- Net Disposable Income (NDI)
- Maximum Affordable Monthly EMI
- Recommended Revolving Credit Limit / Term Loan Cap
- Automated Approval Decision (APPROVED, REFER, DECLINED)
"""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class CreditUnderwritingEngine:
    def __init__(
        self,
        max_foir_ratio: float = 0.45,          # Regulatory Cap: Max 45% income to debt obligations (fine-tuned for target distribution)
        stress_buffer_ratio: float = 0.12,     # 12% safety buffer for macroeconomic shocks (fine-tuned)
        tenure_months: int = 24,              # Default loan tenure term
        annual_interest_rate: float = 0.24    # Standard credit risk annual interest rate (24%)
    ):
        self.max_foir_ratio = max_foir_ratio
        self.stress_buffer_ratio = stress_buffer_ratio
        self.tenure_months = tenure_months
        self.annual_interest_rate = annual_interest_rate

    def calculate_monthly_interest_rate(self) -> float:
        return self.annual_interest_rate / 12.0

    def calculate_max_loan_from_emi(self, max_emi: float) -> float:
        """Calculates total loan principal supported by a max EMI using standard PV formula."""
        if max_emi <= 0:
            return 0.0
        r = self.calculate_monthly_interest_rate()
        n = self.tenure_months
        # PV = EMI * [(1 - (1+r)^-n) / r]
        pv = max_emi * ((1.0 - (1.0 + r) ** (-n)) / r)
        return round(pv, -3)  # Round to nearest thousand PKR

    def evaluate_credit(
        self,
        customer_id: str,
        p10_income: float,
        p50_income: float,
        p90_income: float,
        existing_monthly_obligations: float = 0.0,
        income_stability_score: float = 70.0,
        has_fraud_flag: bool = False
    ) -> Dict[str, Any]:
        """
        Runs comprehensive underwriting assessment for a given applicant.
        """
        # 1. Hard Refusal Checks
        if has_fraud_flag:
            return {
                "customer_id": customer_id,
                "decision": "DECLINED",
                "decline_reason": "FRAUD_OR_AML_FLAG_TRIGGERED",
                "approved_credit_limit": 0.0,
                "max_affordable_emi": 0.0,
                "current_foir": 0.0,
                "projected_foir": 0.0
            }

        if p10_income <= 0:
            return {
                "customer_id": customer_id,
                "decision": "DECLINED",
                "decline_reason": "ZERO_OR_UNVERIFIED_CONSERVATIVE_INCOME",
                "approved_credit_limit": 0.0,
                "max_affordable_emi": 0.0,
                "current_foir": 0.0,
                "projected_foir": 0.0
            }

        # 2. Income Baseline & Regulatory FOIR Capacity
        # Underwriting uses P10 (conservative floor) for safety
        underwriting_income = p10_income
        max_total_debt_capacity = underwriting_income * self.max_foir_ratio

        # Current FOIR Calculation
        current_foir = round((existing_monthly_obligations / underwriting_income) * 100.0, 2)

        # 3. Net Disposable Income (NDI) & Affordable EMI
        available_emi_capacity = max(0.0, max_total_debt_capacity - existing_monthly_obligations)
        # Apply 10% Macro Stress Buffer
        max_affordable_emi = round(available_emi_capacity * (1.0 - self.stress_buffer_ratio), 2)

        # Projected Total FOIR if Max EMI is drawn
        projected_total_obligation = existing_monthly_obligations + max_affordable_emi
        projected_foir = round((projected_total_obligation / underwriting_income) * 100.0, 2)

        # 4. Maximum Principal Loan Cap
        max_loan_principal = self.calculate_max_loan_from_emi(max_affordable_emi)

        # Revolving Credit Card Limit (typically 2x P50 median monthly income)
        revolving_credit_limit = round(min(p50_income * 2.0, max_loan_principal * 0.50), -3)

        # 5. Automated Decisioning Rules
        if current_foir > 45.0:  # Fine-tuned FOIR cap
            decision = "DECLINED"
            decision_reason = "EXCEEDS_MAXIMUM_FOIR_CAP_45_PERCENT"
        elif max_affordable_emi < 4000.0:  # Fine-tuned minimum EMI requirement
            decision = "DECLINED"
            decision_reason = "INSUFFICIENT_NET_DISPOSABLE_INCOME"
        elif income_stability_score < 45.0:  # Fine-tuned stability threshold
            decision = "REFER"
            decision_reason = "LOW_INCOME_STABILITY_MANUAL_REVIEW_REQUIRED"
        else:
            decision = "APPROVED"
            decision_reason = "MEETS_ALL_UNDERWRITING_STABILITY_AND_FOIR_CRITERIA"

        return {
            "customer_id": customer_id,
            "decision": decision,
            "decision_reason": decision_reason,
            "underwriting_income_p10": underwriting_income,
            "median_income_p50": p50_income,
            "optimistic_income_p90": p90_income,
            "existing_monthly_obligations": existing_monthly_obligations,
            "current_foir_percent": current_foir,
            "projected_foir_percent": projected_foir if decision == "APPROVED" else current_foir,
            "max_affordable_emi": max_affordable_emi if decision == "APPROVED" else 0.0,
            "approved_term_loan_limit": max_loan_principal if decision == "APPROVED" else 0.0,
            "approved_revolving_card_limit": revolving_credit_limit if decision == "APPROVED" else 0.0,
            "applied_interest_rate_annual": f"{int(self.annual_interest_rate * 100)}%"
        }


# Quick Verification Execution
if __name__ == "__main__":
    engine = CreditUnderwritingEngine()
    sample_evaluation = engine.evaluate_credit(
        customer_id="CUST0000001",
        p10_income=85000.0,
        p50_income=115000.0,
        p90_income=145000.0,
        existing_monthly_obligations=15000.0,
        income_stability_score=78.5,
        has_fraud_flag=False
    )
    import json
    print(json.dumps(sample_evaluation, indent=2))