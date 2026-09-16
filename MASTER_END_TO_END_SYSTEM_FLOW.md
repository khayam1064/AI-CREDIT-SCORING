# AI CREDIT SCORING ENGINE: COMPLETE END-TO-END SYSTEM FLOW
**Document Version:** 2.0.0 (Master Architecture Diagram & Execution Trace)
**System Designation:** Enterprise Credit Decisioning & Lakehouse Lending Platform

---

## 1. Master High-Level Architecture Flow

`
[ 15 RAW DATA FAMILIES (Bronze Layer) ]
 (NADRA, Device SDK, Bank Accounts, Telco, Digital Wallets, Core LMS, Sanctions)
                          |
                          v
 [ ETL & FEATURE STORE ENGINE (Silver Layer) ]
  * generators/build_feature_store_v2.py
  * Parquet Columnar Compression + Entity Alignment
                          |
                          v
 [ 12 SPECIALIZED SUB-PILLAR MODELS (P01 - P12) ]
  * P01: Identity Trust (Rules + GBM)
  * P02: Fraud Risk (IsoForest + GNN + GBM)  [Hard Knockout Check]
  * P03: Income Estimation (Quantile Pinball LightGBM: P10, P50, P90)
  * P04: Cash Flow Health (GBM on Volatility)
  * P05: Affordability Engine (SBP FOIR <= 45% + Disposable Margin)
  * P06: Stability (GBM on Tenures)
  * P07: Behavioral Risk (GRU Sequence Typing + Hesitation GBM)
  * P08: Digital Footprint (Alternative Telco/Wallet GBM)
  * P09: Bureau Proxy (WoE Scorecard + Inquiry Velocity GBM)
  * P10: Relationship & Loyalty (Survival Model + Clean Repaid Cycles)
  * P11: Collection Risk (Markov Roll-Rate Transitions: 30->60->90 DPD)
  * P12: Statutory Compliance (NACTA / SBP / AML / PEP Strict Rules)
                          |
                          v (Outputs 11 Normalized Scores [0 - 100])
 [ META-STACKING ENSEMBLE ENGINE ]
  * Base Learners: Logistic Regression + LightGBM + XGBoost
  * Meta-Learner: Constrained Stacking Classifier
  * Calibrated Probability of Default (PD %)
                          |
                          v
 [ CREDIT SCORING & RATING CONVERSION ]
  * Points to Double Odds (PDO = 20, Base = 600 @ 50:1)
  * 3-Digit Score: [300 - 850]
  * Credit Grade: Grade A (Super-Prime) to Grade E (High Risk)
                          |
                          v
 [ 4-BOUND MULTI-CONSTRAINT LIMIT ENGINE ]
  Limit = min(Bound 1: Affordability, Bound 2: Risk Cap, Bound 3: Policy Cap, Bound 4: Progression)
  Risk-Adjusted Pricing: Base Rate (KIBOR) + Credit Spread (Grade A-E) <= SBP 36% Cap
                          |
                          v
 [ EXPLAINABLE AI & ADVERSE ACTION CODES (TreeSHAP) ]
  * Top 5 Positive Drivers (Trust Enhancers)
  * Top 5 Negative Drivers (Adverse Action Regulatory Disclosures)
                          |
                          v
 [ CORE ACCOUNTING LEDGER & IFRS 9 STAGING (Gold Layer) ]
  * Double-Entry Disbursement (Debit Loan Asset, Credit Customer Wallet)
  * IFRS 9 Staging: Stage 1 (12m ECL), Stage 2 (Lifetime ECL / SICR), Stage 3 (Impaired)
                          |
                          v
 [ DUAL-MODE DEPLOYMENT INTERFACES ]
  * Mode 1: Online REST API (FastAPI / Swagger @ :8000) [<15ms latency]
  * Mode 2: Executive Web Console (Streamlit @ :8501)
  * Mode 3: Offline Batch Processing CLI (>5,000 customers / sec)
`

---

## 2. Step-by-Step Detailed Execution Lifecycle

### Step 1: Raw Data Acquisition & Lakehouse Feature Store
* **Input:** Raw unjoined data streams across 15 families (data/raw/transactions.csv, 	elecom_signals.csv, device_behavioral.csv, etc.).
* **Process:** The ingestion script generators/build_feature_store_v2.py resolves entities by customer_id, calculates rolling aggregations, and writes the curated Silver Parquet store: eature_store/customer_features_v2.parquet (250,000 entity rows, 27 model signals, Snappy compression).

### Step 2: The 12 Sub-Pillar Assessment Layer
The orchestrator executes all 12 sub-pillars in parallel:
1. **Compliance & Fraud Knockouts First:**
   * If dev_is_rooted == True, dev_emulator_detected == True, or is_sanctioned == True, the engine immediately halts inference and outputs **DECLINE (Hard Knockout)**.
2. **Quantile Income Modeling:**
   * Evaluates P10 (lower conservative income), P50 (median), and P90 (upper bound) to strip away borrower declaration exaggeration.
3. **P01-P11 Domain Scoring:**
   * Each domain produces an institutional score on a normalized scale of 0.0 to 100.0.

### Step 3: Meta-Stacking Ensemble & PD Calibration
* **Inputs:** The vector of 11 sub-scores: [S_P01, S_P02, ..., S_P11] in [0, 100]^11.
* **Execution:**
  1. Base estimators (Logistic Regression, LightGBM, XGBoost) generate cross-validated default probability estimates.
  2. The meta-learner (models/pd_meta_learner.joblib) uses monotonic constraints to synthesize them into a calibrated 12-month **Probability of Default (PD %)**.
* **Score & Rating Mapping:**
  * Score formula: Score = 600 + (20 / ln(2)) * (ln((1 - PD)/PD) - Base_Odds)
  * Grade Assignment:
    * **Grade A (Super-Prime):** PD <= 1.0% (Score 780 - 850)
    * **Grade B (Prime):** PD 1.0% - 3.0% (Score 700 - 779)
    * **Grade C (Near-Prime):** PD 3.0% - 7.0% (Score 620 - 699)
    * **Grade D (Sub-Prime):** PD 7.0% - 15.0% (Score 550 - 619)
    * **Grade E (High Risk / Decline):** PD > 15.0% (Score 300 - 549)

### Step 4: The 4-Bound Multi-Constraint Limit Engine
The system sizes loan capacity using 4 independent risk boundaries:
1. **Bound 1 (Affordability Bound):**
   * Max EMI = max(0, (P10 * 0.45) - Existing Debt)
   * Bound 1 = PV of Max EMI at APR over Tenure
2. **Bound 2 (Risk-Grade Multiplier Cap):**
   * Bound 2 = k(Grade) * P50 * 12 * 0.50
   * (where k = 1.5 for Grade A, 1.2 for Grade B, 0.8 for Grade C, 0.5 for Grade D, 0.0 for Grade E).
3. **Bound 3 (Institutional Policy Cap):**
   * Bound 3 = PKR 2,000,000
4. **Bound 4 (Progression / Anti-Bust-Out Ladder):**
   * Bound 4 = PKR 50,000 * 1.5^(min(Prior Repaid Cycles, 6))

**Final Approved Limit:** Approved Limit = min(Bound 1, Bound 2, Bound 3, Bound 4)
**Risk-Based APR Pricing:** APR = min(36.0%, KIBOR (18.5%) + Grade Spread (4% to 16%))

### Step 5: Explainable AI & Adverse Action (TreeSHAP)
* Computes exact Shapley feature attributions.
* Outputs:
  * **Top 5 Positive Drivers:** Strengths that boosted the score (e.g., zero historical DPD, high salary stability).
  * **Top 5 Negative Drivers:** Formal Adverse Action reason codes required by banking regulators to give applicants clear justification if declined or capped.

### Step 6: Core Ledgering & IFRS 9 Staging (Gold Layer)
1. **Loan Disbursement:**
   When accepted, the engine writes an immutable transaction into data/processed/disbursed_loans.parquet.
2. **IFRS 9 Expected Credit Loss (ECL):**
   * ECL = PD * LGD * EAD * Discount Factor
   * **Stage 1 (Normal):** 12-Month ECL.
   * **Stage 2 (Significant Increase in Credit Risk - SICR):** Lifetime ECL.
   * **Stage 3 (Default / >90 DPD):** Specific Provisioning.

### Step 7: Dual-Mode Deployment Serving
* **Online Mode:** Served via FastAPI (pi/main.py) and Streamlit (dashboard/app.py). Sub-15ms response.
* **Offline Mode:** Executed via 
un_offline_demo.bat or python -m scoring.run_batch_scoring with zero internet dependencies.
