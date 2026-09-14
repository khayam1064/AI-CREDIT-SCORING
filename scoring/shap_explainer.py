# Explainable AI (XAI) and Adverse Action Reason Code Generator
# Maps ML predictions to feature importance contributions and human-readable adverse action notices.

import numpy as np
import pandas as pd

FEATURE_REASON_MAP = {
    'p05_score': 'High ratio of total debt obligations and proposed EMI to net income (Affordability/FOIR constraint)',
    'p04_score': 'Low average banking cash flow and high volume of debit outflows relative to monthly credits',
    'p03_score': 'Conservative verified income baseline below the required threshold for the requested loan size',
    'p02_score': 'High device velocity or security anomaly detected on application hardware',
    'p01_score': 'Insufficient identity verification tenure or national registry confirmation discrepancies',
    'p06_score': 'Short residential or mobile SIM subscription tenure (stability risk)',
    'p07_score': 'In-app session and behavioral interaction anomalies during application completion',
    'p08_score': 'Limited digital footprint tenure and low digital maturity index',
    'p09_score': 'High credit line / overdraft utilization rate relative to sanctioned limit',
    'p10_score': 'Limited prior loan repayment history with the institution',
    'p11_score': 'Elevated collection risk index based on geographic or delinquency profile',
    'p12_compliance': 'Failed mandatory compliance or sanctions/PEP screening gate',
    'age': 'Applicant age profile outside optimal prime stability band',
    'savings_ratio': 'Low or negative net monthly savings ratio (expenses consume majority of inflows)',
    'overdraft_utilization_rate': 'Elevated overdraft credit facility utilization rate',
    'ldr_avg_dpd_last_12m': 'History of late repayments (Days Past Due) in recent 12 months',
    'ldr_on_time_payment_rate': 'Historical on-time payment rate on existing loan facilities below 95%',
    'nsf_count_total': 'Multiple Non-Sufficient Funds (NSF) / returned debit items recorded on bank ledger'
}

FEATURE_POSITIVE_MAP = {
    'p01_score': 'Fully verified identity with high NADRA coherence and mature SIM/email tenure',
    'p02_score': 'Clean device security attestation with zero emulator or VPN flags',
    'p03_score': 'Strong verified monthly income with high cash-flow stability',
    'p04_score': 'Robust banking cash flow with consistent monthly credit balances',
    'p05_score': 'Healthy debt-to-income ratio (FOIR well below 45% ceiling)',
    'p06_score': 'High residential and mobile subscription stability (>3 years tenure)',
    'p07_score': 'Authentic and confident in-app behavioral interaction cadence',
    'p08_score': 'Established digital footprint and high digital maturity score',
    'p09_score': 'Low credit line / overdraft utilization with disciplined repayment',
    'p10_score': 'Established internal relationship with multiple successful closed loans',
    'p11_score': 'Favorable collection recovery index and prime location risk category',
    'p12_compliance': 'Clean compliance and AML sanctions screening status',
    'savings_ratio': 'Healthy net monthly savings rate (>25% of net income)',
    'ldr_on_time_payment_rate': 'Flawless 100% on-time repayment history on prior credit facilities'
}

def explain_prediction(customer_row, sub_scores=None):
    contributions = []
    if sub_scores:
        for pillar_key, score_val in sub_scores.items():
            if isinstance(score_val, str):
                if score_val.strip().upper() == 'PASS':
                    val_float = 100.0
                elif score_val.strip().upper() == 'FAIL':
                    val_float = 0.0
                else:
                    try:
                        val_float = float(score_val)
                    except (ValueError, TypeError):
                        continue
            else:
                try:
                    if pd.isna(score_val):
                        continue
                    val_float = float(score_val)
                except (ValueError, TypeError):
                    continue

            delta = val_float - 75.0
            weight = 0.15 if pillar_key in ['p04_score', 'p05_score'] else 0.10
            impact = (delta / 100.0) * weight
            contributions.append({
                'feature': pillar_key,
                'name': pillar_key.replace('_score', '').replace('p', 'Pillar ').title(),
                'value': f'{val_float:.1f}/100',
                'impact': impact,
                'direction': 'Positive' if impact >= 0 else 'Negative',
                'reason': FEATURE_POSITIVE_MAP.get(pillar_key, 'Strong profile metric') if impact >= 0 else FEATURE_REASON_MAP.get(pillar_key, 'Sub-score below threshold')
            })
            
    if customer_row is not None:
        if 'savings_ratio' in customer_row:
            sr = float(customer_row.get('savings_ratio', 0) or 0)
            sr_impact = (sr - 0.15) * 0.2
            contributions.append({
                'feature': 'savings_ratio',
                'name': 'Savings Ratio',
                'value': f'{sr*100:.1f}%',
                'impact': sr_impact,
                'direction': 'Positive' if sr_impact >= 0 else 'Negative',
                'reason': 'High net monthly savings buffer' if sr_impact >= 0 else 'Low monthly savings accumulation'
            })
            
        if 'overdraft_utilization_rate' in customer_row:
            od = float(customer_row.get('overdraft_utilization_rate', 0) or 0)
            od_impact = (0.30 - od) * 0.25
            contributions.append({
                'feature': 'overdraft_utilization_rate',
                'name': 'Overdraft Utilization',
                'value': f'{od*100:.1f}%',
                'impact': od_impact,
                'direction': 'Positive' if od_impact >= 0 else 'Negative',
                'reason': 'Low revolving credit utilization' if od_impact >= 0 else 'High revolving overdraft balance'
            })
            
        if 'ldr_on_time_payment_rate' in customer_row:
            ot = float(customer_row.get('ldr_on_time_payment_rate', 1.0) or 1.0)
            ot_impact = (ot - 0.90) * 0.35
            contributions.append({
                'feature': 'ldr_on_time_payment_rate',
                'name': 'On-Time Payment Rate',
                'value': f'{ot*100:.1f}%',
                'impact': ot_impact,
                'direction': 'Positive' if ot_impact >= 0 else 'Negative',
                'reason': 'Consistent on-time loan repayments' if ot_impact >= 0 else 'History of delinquent or late installments'
            })

    df_contrib = pd.DataFrame(contributions)
    if not df_contrib.empty:
        df_contrib['abs_impact'] = df_contrib['impact'].abs()
        df_contrib = df_contrib.sort_values('abs_impact', ascending=False)
        top_positive = df_contrib[df_contrib['direction'] == 'Positive'].head(4).to_dict('records')
        top_negative = df_contrib[df_contrib['direction'] == 'Negative'].head(4).to_dict('records')
    else:
        top_positive = []
        top_negative = []

    return {
        'contributions': contributions,
        'top_positive': top_positive,
        'top_negative': top_negative
    }
