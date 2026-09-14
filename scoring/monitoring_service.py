import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> dict:
    exp = expected[~np.isnan(expected)]
    act = actual[~np.isnan(actual)]
    
    if len(exp) == 0 or len(act) == 0:
        return {'psi': 0.0, 'status': 'GREEN', 'buckets': []}
        
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(exp, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp, bins=bin_edges)
    act_counts, _ = np.histogram(act, bins=bin_edges)

    exp_pct = np.maximum(exp_counts / len(exp), 1e-4)
    act_pct = np.maximum(act_counts / len(act), 1e-4)

    bucket_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    total_psi = float(np.sum(bucket_psi))

    if total_psi < 0.10:
        status = "GREEN (Stable)"
    elif total_psi < 0.25:
        status = "AMBER (Moderate Shift)"
    else:
        status = "RED (Severe Drift - Retraining Required)"

    return {
        'total_psi': round(total_psi, 4),
        'status': status,
        'bin_count': num_buckets,
        'expected_sample_size': len(exp),
        'actual_sample_size': len(act)
    }

def monitor_portfolio_stability(df_scores: pd.DataFrame, df_features: pd.DataFrame = None) -> dict:
    if df_scores is None or df_scores.empty:
        return {}

    n = len(df_scores)
    split_idx = int(n * 0.8)
    
    scores_dev = df_scores['composite_score_1000'].iloc[:split_idx].values
    scores_prod = df_scores['composite_score_1000'].iloc[split_idx:].values
    score_psi = calculate_psi(scores_dev, scores_prod, num_buckets=10)

    pd_dev = df_scores['calibrated_pd'].iloc[:split_idx].values
    pd_prod = df_scores['calibrated_pd'].iloc[split_idx:].values
    pd_psi = calculate_psi(pd_dev, pd_prod, num_buckets=10)

    features_psi = {}
    if df_features is not None and not df_features.empty:
        tracked_feats = ['age', 'savings_ratio', 'overdraft_utilization_rate', 'foir_pct', 'total_monthly_credit']
        for feat in tracked_feats:
            if feat in df_features.columns:
                f_dev = df_features[feat].iloc[:split_idx].astype(float).values
                f_prod = df_features[feat].iloc[split_idx:].astype(float).values
                f_res = calculate_psi(f_dev, f_prod, num_buckets=8)
                features_psi[feat] = f_res

    return {
        'timestamp': pd.Timestamp.now().isoformat(),
        'score_psi': score_psi,
        'pd_psi': pd_psi,
        'feature_drift': features_psi
    }

if __name__ == "__main__":
    df_s = pd.read_parquet("data/processed/credit_scores_250k.parquet")
    df_f = pd.read_parquet("feature_store/customer_features_v2.parquet")
    res = monitor_portfolio_stability(df_s, df_f)
    print("PSI Monitoring initialized:")
    print("Score PSI:", res['score_psi'])
    print("PD PSI:   ", res['pd_psi'])
