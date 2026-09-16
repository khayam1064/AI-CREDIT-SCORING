# ENTERPRISE CLIENT ONBOARDING & DATA INGESTION GUIDE
## APEX CREDIT AI DECISIONING ENGINE
**Document Version:** 1.0.0 (Production Blueprint)  
**Target Audience:** Client Engineering Leads, Chief Risk Officers (CRO), Data Engineers, and Core Banking Developers (Ciihive / T24 / Finacle).

---

## 1. Executive Summary & Architecture Paradigm

This platform is a **portable, zero-vendor-lock-in AI Credit Underwriting Engine**. When a new financial institution (bank, microfinance bank, or digital lender) adopts this codebase, they replace our simulated data with **their own real-world customer telemetry and loan ledgers**.

The engine operates on a **Decoupled Architecture**:
`
  [ Client's Real Data Sources ]
  (CBS Core, Mobile App SDK, Telco API, Credit Bureau)
                          │
                          ▼
             [ 1. Ingestion & Mapping Layer ]
     (Transforms client schema to Canonical Lakehouse Store)
                          │
                          ▼
            [ 2. Feature Store (Silver Layer) ]
        eature_store/customer_features_v2.parquet
                          │
                          ▼
      ┌───────────────────┴───────────────────┐
      ▼                                       ▼
[ Mode A: Real-Time Serving ]           [ Mode B: Retraining / Tuning ]
• FastAPI REST / gRPC                   • Retrain P01-P11 Sub-Pillars
• Sub-15ms Live Scoring                 • Calibrate Meta-Stacking PD
• 4-Bound Limits & APR                  • Adjust SBP 36% Cap & FOIR 45%
• TreeSHAP Adverse Codes                • Generate IFRS 9 ECL Matrix
`

---

## 2. Step-by-Step Implementation Roadmap for a New Company

### Phase 1: Ingesting the Client's Real Data
A new company does **not** need to modify the ML mathematical core. They only need to provide the features specified in our canonical contract:

1. **Format:** Apache Parquet (customer_features_v2.parquet) or real-time JSON payload.
2. **Primary Key:** customer_id (alphanumeric string, e.g., CUST0008856 or CNIC hash).
3. **Storage Location:** eature_store/customer_features_v2.parquet.

#### Mapping Client Columns to the Canonical Schema:
| Client Real World Source | Canonical Feature Column | Data Type | Notes / Value Rules |
| :--- | :--- | :---: | :--- |
| **NADRA / KYC Registration** | ge | int | Valid borrower age (18 to 75). |
| | city, province | str | City tier mapping for regional risk. |
| | education | str | Categorical: Secondary, Bachelor, Master. |
| **Mobile Banking App SDK** | dev_is_rooted | ool | Hard knockout: If True, application declined instantly. |
| | dev_emulator_detected| ool | Hard knockout: Emulator fraud block. |
| | dev_typing_speed_wpm | loat | Telemetry signal for bot detection. |
| **Payroll / Bank Statement** | monthly_salary | loat | Base gross monthly income (PKR). |
| | 
et_monthly_cashflow | loat | Monthly inflow minus outflows. |
| | oir_pct | loat | Fixed Obligations to Income Ratio (Max 45%). |
| **Telco Provider (Jazz/Zong)** | 	el_sim_age_months | int | SIM tenure (proxy for identity stability). |
| | 	el_is_postpaid | ool | Postpaid lines receive higher trust factor. |
| **Core LMS / Ledger** | ldr_prior_loans_count | int | **Prior Repaid Clean Cycles** (drives Bound 4). |
| | ldr_on_time_payment_rate| loat | Repayment compliance ratio (0.0 to 1.0). |
| | ldr_avg_dpd_last_12m | loat | Average days past due over past 1 year. |
| **Sanctions / AML Watchlist**| is_pep | ool | Politically Exposed Person indicator. |
| | is_sanctioned | ool | Absolute knockout: Anti-Terrorism / NACTA. |

---

## 3. Retraining the Models on the Client's Historical Default Data

If the client has **historical loan default outcomes** (e.g. 2 years of performance data with default_flag 0 or 1), they can retrain the engine to fit their specific risk profile:

### Step 1: Retrain the Quantile Income Models (P10, P50, P90)
`ash
python training/train_quantile_models.py
`
* **What it does:** Trains three LightGBM pinball loss regressors to estimate lower-bound ({10}$), median ({50}$), and upper-bound ({90}$) income, eliminating self-reported salary bias.
* **Output:** Saved into models/lgb_income_p10.joblib, p50.joblib, p90.joblib.

### Step 2: Retrain the 11 Sub-Pillars
`ash
python training/train_all_pillar_models.py
`
* **What it does:** Fits domain-specific LightGBM/XGBoost models for Identity Trust, Fraud Risk, Affordability, Cash Flow, Stability, Behavioral, Digital Footprint, Bureau Proxy, Relationship, and Collection Risk.
* **Output:** Saved into models/lgb_pillar_*.joblib.

### Step 3: Retrain the Calibrated Meta-Stacking Ensemble
`ash
python training/train_pd_ensemble.py
`
* **What it does:** Combines the predictions of base models into a constrained **Logistic Regression Meta-Learner** with strict monotonic weights.
* **Output:** Produces un-skewed 12-month Probability of Default (PD) and maps to Credit Grades A through E.

---

## 4. Policy Configuration & Regulatory Tuning

Every institution has custom risk appetite and central bank mandates. All regulatory parameters are centralized in [scoring/policy_engine.py](file:///d:/setups/income_ai_demo/scoring/policy_engine.py):

`python
# 1. Usury Interest Cap (State Bank of Pakistan / Central Bank)
MAX_REGULATORY_APR = 0.36  # Hard ceiling: 36.0% per annum

# 2. Maximum Debt Burden / FOIR Ceiling
MAX_REGULATORY_FOIR = 0.45  # Hard ceiling: 45.0%

# 3. Four-Bound Limit Engine Policy Ceiling
POLICY_MAX_LOAN_CAP = 2000000.0  # PKR 2,000,000 maximum single exposure

# 4. Clean Repaid Cycle Step-up Ladder
CYCLE_BASE_LIMIT = 50000.0  # PKR 50,000 for brand new customers
CYCLE_STEP_MULTIPLIER = 1.5  # 1.5x multiplier per cleanly settled loan
`
* **To customize:** The client simply updates these values to match their local credit policy.

---

## 5. Deployment Options for the Client's Infrastructure

The engine is engineered for zero-dependency portability across 3 enterprise patterns:

### Pattern A: Dockerized Microservice (Recommended)
Deliver docker-compose.yml to the client's DevOps team:
`yaml
version: '3.8'
services:
  underwriting-api:
    image: client-credit-ai:latest
    ports:
      - 8000:8000
    volumes:
      - ./feature_store:/app/feature_store
      - ./models:/app/models
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

  underwriter-portal:
    image: client-credit-ai:latest
    ports:
      - 8501:8501
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
`

### Pattern B: Standalone On-Premise (Air-Gapped Linux / Windows Server)
1. Install Python 3.11 or 3.13.
2. Install dependencies: pip install -r requirements.txt.
3. Start local server:
   * Windows: Double-click un_offline_demo.bat.
   * Linux: systemctl start apex-credit-ai.service.

---

## 6. Integration Checklist for the Client's Technical Team

- [ ] **Step 1:** Confirm column names in client's database match [Ciihive_Canonical_Feature_Contract.json](file:///d:/setups/income_ai_demo/Ciihive_Canonical_Feature_Contract.json).
- [ ] **Step 2:** Place their baseline customer dataset into eature_store/customer_features_v2.parquet.
- [ ] **Step 3:** Run the automated QA stress test suite:
  `ash
  python scripts/qa_stress_test_suite.py
  `
  *(Must achieve 10/10 tests passed: 100% compliance on knockouts, usury cap, FOIR, TreeSHAP, and ledger).*
- [ ] **Step 4:** Point their Mobile App / Core Banking System (CBS) to:
  * **Real-time Scoring:** POST http://<server-ip>:8000/score/customer/realtime
  * **Loan Disbursement:** POST http://<server-ip>:8000/loans/disburse
  * **Portfolio IFRS 9 ECL:** GET http://<server-ip>:8000/portfolio/ifrs9
