"""
Enterprise Credit AI Engine
Batch Scoring Runner

Scores all 250,000 customers using the 12-pillar credit scoring system.
Loads the feature store v2, runs the composite scorer in vectorized batches,
and outputs full results to data/processed/.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_STORE_V2 = BASE_DIR / "feature_store" / "customer_features_v2.parquet"
FEATURE_STORE_V1 = BASE_DIR / "feature_store" / "customer_features.parquet"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "credit_scores_250k.parquet"
OUTPUT_CSV     = OUTPUT_DIR / "credit_scores_250k.csv"

import sys
sys.path.insert(0, str(BASE_DIR))
from scoring.composite_scorer import compute_scores


def load_feature_store() -> pd.DataFrame:
    if FEATURE_STORE_V2.exists():
        logging.info(f"Loading feature store v2: {FEATURE_STORE_V2}")
        return pd.read_parquet(FEATURE_STORE_V2)
    logging.warning("feature_store_v2 not found — falling back to v1")
    return pd.read_parquet(FEATURE_STORE_V1)


def run_income_model_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Adds P10/P50/P90 income columns using the trained LightGBM models."""
    try:
        feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
        cat_info     = joblib.load(MODELS_DIR / "categorical_info.joblib")
        m_p10 = joblib.load(MODELS_DIR / "lgb_income_p10.joblib")
        m_p50 = joblib.load(MODELS_DIR / "lgb_income_p50.joblib")
        m_p90 = joblib.load(MODELS_DIR / "lgb_income_p90.joblib")

        available_cols = [c for c in feature_cols if c in df.columns]
        missing_cols   = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            logging.warning(f"Missing {len(missing_cols)} model features; filling with 0.")

        X = df[available_cols].copy()
        # Fill missing columns with 0
        for c in missing_cols:
            X[c] = 0

        X = X[feature_cols]  # Ensure correct column order
        cat_cols = cat_info.get("categorical_columns", [])
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        raw_p10 = m_p10.predict(X)
        raw_p50 = m_p50.predict(X)
        raw_p90 = m_p90.predict(X)

        # Enforce monotonicity P10 ≤ P50 ≤ P90
        sorted_preds = np.sort(np.stack([raw_p10, raw_p50, raw_p90], axis=1), axis=1)
        df["p10_income"] = sorted_preds[:, 0].round(2)
        df["p50_income"] = sorted_preds[:, 1].round(2)
        df["p90_income"] = sorted_preds[:, 2].round(2)
        logging.info("Income model inference complete (P10/P50/P90).")
    except Exception as e:
        logging.warning(f"Income model inference failed: {e}. Using declared income as fallback.")
        if "total_monthly_income" in df.columns:
            df["p50_income"] = df["total_monthly_income"]
            df["p10_income"] = df["total_monthly_income"] * 0.80
            df["p90_income"] = df["total_monthly_income"] * 1.25
    return df


def print_summary(results: pd.DataFrame, elapsed: float):
    total = len(results)
    decisions = results["credit_decision"].value_counts()
    grades = results["credit_grade"].value_counts().sort_index()

    print(f"\n{'='*65}")
    print(f"   ENTERPRISE CREDIT SCORING - BATCH RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"Total Customers Scored   : {total:,}")
    print(f"Processing Time          : {elapsed:.1f} seconds")
    print(f"Throughput               : {total/elapsed:,.0f} customers/sec")
    print(f"{'-'*65}")
    print(f"DECISIONS:")
    for dec in ["APPROVED", "REFER", "DECLINED"]:
        n = decisions.get(dec, 0)
        print(f"  {dec:<12}: {n:>7,}  ({n/total*100:.1f}%)")
    print(f"{'-'*65}")
    print(f"GRADE DISTRIBUTION:")
    for g in ["A+", "A", "B", "C", "D", "F"]:
        n = grades.get(g, 0)
        print(f"  Grade {g}    : {n:>7,}  ({n/total*100:.1f}%)")
    print(f"{'-'*65}")
    print(f"SCORE STATISTICS:")
    scored = results[results["composite_score_1000"] > 0]["composite_score_1000"]
    if len(scored):
        print(f"  Mean Score   : {scored.mean():.0f}")
        print(f"  Median Score : {scored.median():.0f}")
        print(f"  Std Dev      : {scored.std():.0f}")
        print(f"  P10 / P90    : {scored.quantile(0.10):.0f} / {scored.quantile(0.90):.0f}")
    print(f"{'-'*65}")
    print(f"TOP DECLINE REASONS:")
    dr = results["decline_reason"].dropna().value_counts().head(5)
    for reason, count in dr.items():
        print(f"  {reason:<35}: {count:>6,}")
    print(f"{'-'*65}")
    print(f"PILLAR SCORE AVERAGES (non-zero):")
    pillar_cols = [c for c in results.columns if c.startswith("p0") or c.startswith("p1")]
    pillar_score_cols = [c for c in pillar_cols if c.endswith("_score")]
    for pc in sorted(pillar_score_cols):
        mean_v = results[pc].mean()
        print(f"  {pc:<25}: {mean_v:.1f}/100")
    print(f"{'='*65}\n")


def main():
    t0 = time.time()
    logging.info("Loading feature store...")
    df = load_feature_store()
    logging.info(f"Feature store loaded: {len(df):,} customers, {df.shape[1]} features")

    logging.info("Running income model inference (P10/P50/P90)...")
    df = run_income_model_inference(df)

    logging.info("Running 12-pillar composite scorer on all 250K customers...")
    results = compute_scores(df)

    logging.info(f"Saving results → {OUTPUT_PARQUET}")
    results.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")

    logging.info(f"Saving CSV → {OUTPUT_CSV}")
    results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print_summary(results, elapsed)


if __name__ == "__main__":
    main()
