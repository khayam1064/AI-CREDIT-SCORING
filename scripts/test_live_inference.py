import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scoring import composite_scorer

df_fs = pd.read_parquet('feature_store/customer_features_v2.parquet')
df_fs.set_index('customer_id', inplace=True)

for cid in ['CUST0000001', 'CUST0000002', 'CUST0000003', 'CUST0000022', 'CUST0000090']:
    row_df = df_fs.loc[[cid]]
    res = composite_scorer.compute_scores(row_df)
    r = res.iloc[0]
    print(f"[{cid}] Score: {r['composite_score_1000']:3d} | Grade: {r['credit_grade']} | Decision: {r['credit_decision']:8s} | PD: {r['calibrated_pd']*100:5.2f}% | P10: PKR {r['p10_income']:8,.0f} | P50: PKR {r['p50_income']:8,.0f} | P90: PKR {r['p90_income']:8,.0f}")
