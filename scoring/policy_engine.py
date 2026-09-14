# Policy Engine and IFRS 9 Expected Credit Loss (ECL) Calculation Engine
# Implements risk-based pricing, multi-bound limit sizing, and portfolio IFRS 9 provisions.
# Enhanced with account-level dynamic EAD, CCF undrawn limits, and multi-scenario macroeconomic stress testing.

import numpy as np
import pandas as pd

# Multipliers for RiskLimit based on credit grade
GRADE_RISK_MULTIPLIER = {
    'A': 1.50,
    'B': 1.20,
    'C': 0.80,
    'D': 0.50,
    'E': 0.25,
    'F': 0.00,
    'G': 0.00
}

# Standard Basel / IFRS 9 Forward-Looking Macroeconomic Scenarios (GDP & Inflation weights)
MACRO_SCENARIOS = {
    'Baseline': {'weight': 0.50, 'pd_mult': 1.00, 'lgd_mult': 1.00},
    'Downturn': {'weight': 0.30, 'pd_mult': 1.25, 'lgd_mult': 1.15},
    'Upturn':   {'weight': 0.20, 'pd_mult': 0.85, 'lgd_mult': 0.90}
}

def calculate_limits_and_pricing(
    grade: str,
    pd_cal: float,
    p10_income: float,
    p50_income: float,
    existing_obligations: float,
    prior_loans_count: int = 0,
    policy_cap: float = 2000000.0,
    foir_max: float = 0.45,
    lgd: float = 0.70,
    cof: float = 0.08,
    opex: float = 0.04,
    capital_charge: float = 0.04,
    target_margin: float = 0.06
) -> dict:
    p10 = max(0.0, float(p10_income or 0))
    p50 = max(0.0, float(p50_income or 0))
    obligations = max(0.0, float(existing_obligations or 0))
    pd_val = min(1.0, max(0.001, float(pd_cal or 0.05)))

    # 1. Affordability Limit
    max_cap = p10 * foir_max
    avail_emi = max(0.0, max_cap - obligations)
    max_emi = round(avail_emi * 0.88, 2)
    r, n = 0.24 / 12.0, 24
    affordability_limit = round(max_emi * ((1.0 - (1.0 + r)**-n) / r), -3) if max_emi > 0 else 0.0

    # 2. Risk Limit
    k = GRADE_RISK_MULTIPLIER.get(grade, 0.50)
    risk_limit = round(p50 * k * 12.0 * 0.50, -3)

    # 3. Progression Cap
    base_first_loan = 50000.0
    progression_cap = min(policy_cap, base_first_loan * (1.5 ** min(prior_loans_count, 6)))

    # Final Approved Limit
    if grade in ['F', 'G']:
        approved_limit = 0.0
    else:
        approved_limit = min(affordability_limit, risk_limit, policy_cap, progression_cap)
        approved_limit = max(0.0, round(approved_limit, -3))

    # Revolving Card Limit
    card_limit = round(min(p50 * 2.0, approved_limit * 0.50), -3) if approved_limit > 0 else 0.0

    # Risk-Based Pricing (APR)
    ecl_rate = pd_val * lgd
    raw_apr = cof + opex + ecl_rate + capital_charge + target_margin
    apr = min(0.36, max(0.14, round(raw_apr, 4)))

    return {
        'approved_limit': approved_limit,
        'card_limit': card_limit,
        'max_affordable_emi': max_emi,
        'affordability_limit': affordability_limit,
        'risk_limit': risk_limit,
        'progression_cap': progression_cap,
        'policy_cap': policy_cap,
        'apr': apr,
        'apr_pct': round(apr * 100, 2),
        'pricing_breakdown': {
            'Cost of Funds (CoF)': round(cof * 100, 1),
            'Operating Expense (OpEx)': round(opex * 100, 1),
            'ECL Risk Premium': round(ecl_rate * 100, 2),
            'Capital Charge': round(capital_charge * 100, 1),
            'Target Margin': round(target_margin * 100, 1)
        }
    }

def compute_account_ead(df: pd.DataFrame, ccf: float = 0.20) -> pd.Series:
    """
    Computes true account-level Exposure at Default (EAD):
    EAD_i = Drawn_Balance_i + CCF * Undrawn_Limit_i
    If granular ledger fields are missing, intelligently falls back to loan size/utilization signals.
    """
    # 1. Check drawn balance
    if 'total_overdraft_used' in df.columns:
        drawn = df['total_overdraft_used'].fillna(0.0)
    elif 'ldr_last_loan_amount' in df.columns:
        drawn = df['ldr_last_loan_amount'].fillna(0.0)
    else:
        drawn = pd.Series(0.0, index=df.index)

    # 2. Check total limit / undrawn facility
    if 'max_overdraft_limit' in df.columns:
        tot_limit = df['max_overdraft_limit'].fillna(0.0)
    elif 'total_overdraft_limit' in df.columns:
        tot_limit = df['total_overdraft_limit'].fillna(0.0)
    else:
        tot_limit = drawn

    undrawn = (tot_limit - drawn).clip(lower=0.0)
    account_ead = drawn + (ccf * undrawn)

    # Minimum standard baseline floor for active applicants (e.g. 50k nano facility if zero history)
    account_ead = account_ead.where(account_ead > 10000, 150000.0)
    return account_ead.round(2)

def calculate_portfolio_ifrs9(
    df_scores: pd.DataFrame,
    df_features: pd.DataFrame = None,
    avg_exposure_per_borrower: float = 250000.0,
    base_lgd: float = 0.70
) -> dict:
    """
    Full IFRS 9 Impairment Engine with:
    1. Account-level dynamic EAD (Drawn + CCF * Undrawn)
    2. 3-Stage Impairment Lifecycle (12M ECL for Stage 1, Lifetime ECL for Stage 2 & 3)
    3. Multi-Scenario Forward-Looking Macroeconomic Stress Weighting (Baseline, Downturn, Upturn)
    """
    if df_scores is None or df_scores.empty:
        return {}
        
    df = df_scores.copy()
    total_cust = len(df)
    
    # 1. Compute dynamic EAD vector
    if df_features is not None and not df_features.empty:
        # Align index
        common_idx = df.index.intersection(df_features.index)
        if len(common_idx) > 0:
            ead_series = compute_account_ead(df_features.loc[common_idx])
            df.loc[common_idx, 'account_ead'] = ead_series
        else:
            df['account_ead'] = avg_exposure_per_borrower
    elif 'account_ead' not in df.columns:
        df['account_ead'] = avg_exposure_per_borrower
        
    ead_vec = df['account_ead'].fillna(avg_exposure_per_borrower)

    # 2. Classify into IFRS 9 Stages
    # Stage 1: Performing (Grades A, B, C)
    # Stage 2: Underperforming / SICR (Grades D, E)
    # Stage 3: Credit Impaired / Default (Grades F, G)
    stage1_mask = df['credit_grade'].isin(['A', 'B', 'C'])
    stage2_mask = df['credit_grade'].isin(['D', 'E'])
    stage3_mask = df['credit_grade'].isin(['F', 'G'])

    stage1_count = int(stage1_mask.sum())
    stage2_count = int(stage2_mask.sum())
    stage3_count = int(stage3_mask.sum())

    total_portfolio_exposure = float(ead_vec.sum())
    
    s1_ead = float(ead_vec[stage1_mask].sum()) if stage1_count > 0 else 0.0
    s2_ead = float(ead_vec[stage2_mask].sum()) if stage2_count > 0 else 0.0
    s3_ead = float(ead_vec[stage3_mask].sum()) if stage3_count > 0 else 0.0

    s1_pd_mean = float(df.loc[stage1_mask, 'calibrated_pd'].fillna(0.03).mean()) if stage1_count > 0 else 0.03
    s2_pd_mean = float(min(1.0, df.loc[stage2_mask, 'calibrated_pd'].fillna(0.35).mean() * 2.2)) if stage2_count > 0 else 0.40
    s3_pd_mean = 1.0

    # 3. Macroeconomic Scenario Weighting (IFRS 9 Standard Section 5.5.17)
    scenario_ecl = {}
    weighted_total_ecl = 0.0
    weighted_s1_ecl = 0.0
    weighted_s2_ecl = 0.0
    weighted_s3_ecl = 0.0

    for sc_name, sc_params in MACRO_SCENARIOS.items():
        w = sc_params['weight']
        pd_m = sc_params['pd_mult']
        lgd_m = sc_params['lgd_mult']
        
        curr_lgd = min(1.0, base_lgd * lgd_m)
        curr_s1_pd = min(1.0, s1_pd_mean * pd_m)
        curr_s2_pd = min(1.0, s2_pd_mean * pd_m)
        curr_s3_pd = 1.0

        sc_s1 = s1_ead * curr_s1_pd * curr_lgd
        sc_s2 = s2_ead * curr_s2_pd * curr_lgd
        sc_s3 = s3_ead * curr_s3_pd * curr_lgd
        sc_tot = sc_s1 + sc_s2 + sc_s3

        scenario_ecl[sc_name] = {
            'weight_pct': int(w * 100),
            'total_ecl': round(sc_tot, -3),
            'coverage_pct': round(sc_tot / total_portfolio_exposure * 100, 2) if total_portfolio_exposure > 0 else 0
        }

        weighted_total_ecl += w * sc_tot
        weighted_s1_ecl += w * sc_s1
        weighted_s2_ecl += w * sc_s2
        weighted_s3_ecl += w * sc_s3

    ecl_coverage_ratio = (weighted_total_ecl / total_portfolio_exposure * 100) if total_portfolio_exposure > 0 else 0

    return {
        'total_portfolio_exposure': total_portfolio_exposure,
        'total_ecl_provision': round(weighted_total_ecl, -3),
        'ecl_coverage_ratio_pct': round(ecl_coverage_ratio, 2),
        'macro_scenarios': scenario_ecl,
        'stages': {
            'Stage 1 (Performing - 12M ECL)': {
                'count': stage1_count,
                'pct': round(stage1_count / total_cust * 100, 1),
                'exposure_ead': round(s1_ead, -3),
                'avg_pd_pct': round(s1_pd_mean * 100, 2),
                'ecl_pkr': round(weighted_s1_ecl, -3)
            },
            'Stage 2 (SICR - Lifetime ECL)': {
                'count': stage2_count,
                'pct': round(stage2_count / total_cust * 100, 1),
                'exposure_ead': round(s2_ead, -3),
                'avg_pd_pct': round(s2_pd_mean * 100, 2),
                'ecl_pkr': round(weighted_s2_ecl, -3)
            },
            'Stage 3 (Credit Impaired)': {
                'count': stage3_count,
                'pct': round(stage3_count / total_cust * 100, 1),
                'exposure_ead': round(s3_ead, -3),
                'avg_pd_pct': 100.0,
                'ecl_pkr': round(weighted_s3_ecl, -3)
            }
        }
    }
