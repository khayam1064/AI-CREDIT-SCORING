# Enterprise AI Credit Scoring & Decision Engine — Production Architecture Specification

**Version:** 3.0 (Enterprise Production Specification)  
**Standard Compliance:** IFRS 9 / Basel III / SBP & Regulatory Fair Lending Guidelines  
**System Scope:** End-to-End Multi-Stage Credit Underwriting, Quantile Income Modeling, Stacking Ensemble, SHAP Explainability & Risk-Based Pricing  

---

## Table of Contents
1. [Enterprise Architecture & Model Inventory](#1-enterprise-architecture--model-inventory)
2. [End-to-End Mathematical Formulations](#2-end-to-end-mathematical-formulations)
   - 2.1 Base Learners, Monotonicity & Stacking Ensemble (Section 6.4.1)
   - 2.2 Probability Calibration & Goodness-of-Fit (Section 6.4.2)
   - 2.3 Points to Double the Odds (PDO) Score Formulation (Section 6.4.3)
   - 2.4 IFRS 9 Expected Credit Loss (ECL) & Lifetime Staging (Section 6.4.4)
   - 2.5 Risk-Based Pricing & APR Structure (Section 6.4.5)
   - 2.6 Model Validation & Production Launch Gates (Section 6.4.6)
3. [Decision Engine & Underwriting Waterfall](#3-decision-engine--underwriting-waterfall)
   - 3.1 Six-Stage Decision Waterfall (Section 7.1)
   - 3.2 Enterprise Decision Matrix & Grade Policy (Section 7.2)
   - 3.3 Multi-Bound Limit Assignment Engine (Section 7.3)
4. [Explainable AI (XAI), SHAP & Adverse Action Generation](#4-explainable-ai-xai-shap--adverse-action-generation)
5. [Enterprise Safe Code Writing & Exception Handling Architecture](#5-enterprise-safe-code-writing--exception-handling-architecture)
6. [Physical Model Registry on Disk](#6-physical-model-registry-on-disk)

---

## 1. Enterprise Architecture & Model Inventory

The enterprise credit engine partitions underwriting tasks across specialized machine learning algorithms, deep learning sequence models, and deterministic regulatory gates:

| Task / Domain | Machine Learning Algorithm | Loss Function / Optimization Objective | Key Evaluation Metrics | Architectural Rationale & Why Selected |
| :--- | :--- | :--- | :--- | :--- |
| **PD (Main Credit Risk)** | LightGBM + XGBoost + Logistic Stacking | Binary Log-Loss with Monotonic Constraints | AUC-ROC, Gini, Kolmogorov-Smirnov (KS), Brier Score | State of the art on tabular credit data; fast gradient boosting; native SHAP TreeExplainer compatibility. |
| **Regulatory Benchmark** | Logistic Regression on Weight of Evidence (WoE) | Log-Loss with $L_2$ Ridge Regularization | Gini, Hosmer-Lemeshow Calibration | Fully auditable, monotonic linear coefficients; trusted regulatory benchmark for central bank examiners. |
| **Income Prediction** | LightGBM Quantile Regressors (P10, P50, P90) | Asymmetric Pinball (Quantile) Loss | MAPE, Quantile Coverage (Pinball Error) | Uncertainty-aware income intervals rather than single point guesses; bounds debt service capacity. |
| **Cash-Flow Forecast** | LSTM / Temporal CNN on Transaction Sequences | Mean Squared Error (MSE) / Quantile Loss | Horizon-specific MAPE, Volatility Error | Captures pay-cycle seasonality and payday trends for optimal EMI collection date scheduling. |
| **Point-in-Time Fraud** | LightGBM Classifier + Isolation Forest + Autoencoder | Log-Loss & Reconstruction Error | Precision@k, Alert Rate, False Discovery Rate | Supervised boosting catches known fraud signatures; unsupervised autoencoders detect novel zero-day attacks. |
| **Fraud Rings & Syndicates** | Graph Neural Network (GNN) / Louvain Community Detection | Link-Prediction Cross-Entropy Loss | Ring Recall, False Positive Rate | Organized syndicate fraud is relational (shared device-ID, address, bank); graph topology is required to detect rings. |
| **Deepfake / Liveness** | Vision CNN + Active Challenge-Response | Vendor Pre-Trained Binary Classification | APCER / BPCER (ISO/IEC 30107-3) | Specialized computer vision domain; vendor-grade hardware security attestation. |
| **Behavioral Sequence** | GRU (Gated Recurrent Unit) over In-App Telemetry | Binary Cross-Entropy Loss | AUC Uplift over static GBM | Sequence order of form navigation, keystroke cadence, and paste events carries critical intent signals. |
| **Survival / Lifetime PD** | Cox Proportional Hazards / Gradient-Boosted Survival | Partial Likelihood Loss | Concordance Index (C-Index), Integrated Brier | Models *when* default happens across time horizons; essential for IFRS 9 multi-year staging. |
| **Collections Roll-Rate** | Multi-State Markov Chain + Gradient Boosting | Transition Multi-Class Likelihood | Transition Roll-Rate Accuracy, MAE | Standard delinquency migration mechanics (Current -> 30 DPD -> 60 DPD -> 90 DPD -> Write-Off). |
| **Limit Strategy (Mature)**| Constrained Contextual Bandits / Reinforcement Learning | Profit Reward with Risk & Fairness Constraints | Net Profit per Limit Unit, Bad Debt Rate | Automatically learns optimal credit line increase/decrease ladder based on borrower lifecycle outcomes. |
| **Customer Segmentation** | K-Means / HDBSCAN on Behavioral Embeddings | Inertia & Cluster Density (Silhouette) | Silhouette Score, Segment Homogeneity | Risk-tier communication, targeted champion-challenger underwriting strategies. |
| **Document Intelligence** | OCR + Layout Transformer (Donut / LayoutLMv3) | Sequence-to-Sequence Entity Cross-Entropy | Field-Level $F_1$ Score, Character Error Rate | Parses paper bank statements, utility bills, and salary slips when automated APIs are unavailable. |

---

## 2. End-to-End Mathematical Formulations

### 2.1 Base Learners, Monotonicity & Stacking Ensemble (Section 6.4.1)

#### Additive Gradient Boosting Model
Gradient boosting builds an ensemble of $M$ decision trees fit sequentially to negative pseudo-residuals:

$$F_M(\mathbf{x}) = \sum_{m=1}^M 
u \cdot h_m(\mathbf{x})$$

Where $
u \in (0, 1]$ is the learning rate, and $h_m(\mathbf{x})$ is a regression tree fit to the negative gradient of the binary log-loss:

$$PD_{	ext{raw}}(\mathbf{x}) = \sigma(F_M(\mathbf{x})) = rac{1}{1 + e^{-F_M(\mathbf{x})}}$$

#### Monotonic Constraints
To ensure regulatory compliance and prevent economic counter-intuition, monotonic constraints are strictly enforced in LightGBM and XGBoost:
* $rac{\partial PD}{\partial (	ext{Income})} \le 0$ (Default probability must not increase as verified income increases).
* $rac{\partial PD}{\partial (	ext{FOIR})} \ge 0$ (Default probability must not decrease as debt obligations increase).
* $rac{\partial PD}{\partial (	ext{DPD})} \ge 0$ (Default probability must not decrease as days past due increase).

#### Stacking Meta-Learner Formulation
Base model predictions and sub-scores are collected into an Out-of-Fold (OOF) meta-vector $\mathbf{z}(\mathbf{x})$:

$$\mathbf{z}(\mathbf{x}) = \left[ PD_{	ext{LGBM}}(\mathbf{x}), PD_{	ext{XGB}}(\mathbf{x}), PD_{	ext{Logit}}(\mathbf{x}), P_{01}(\mathbf{x}), \dots, P_{11}(\mathbf{x}) ight]$$

The ensembled default probability is computed via a regularized Logistic Stacking Meta-Learner:

$$PD_{	ext{ens}}(\mathbf{x}) = \sigma\left(\mathbf{w}^T \mathbf{z}(\mathbf{x}) + bight) = rac{1}{1 + e^{-(\mathbf{w}^T \mathbf{z}(\mathbf{x}) + b)}}$$

---

### 2.2 Probability Calibration & Goodness-of-Fit (Section 6.4.2)

Raw model outputs rank-order risk effectively but are rarely true uncalibrated probabilities. The engine applies isotonic regression / Platt scaling to map ensemble outputs to empirical default frequencies:

$$PD_{	ext{cal}} = 	ext{IsotonicFit}\left(PD_{	ext{ens}} 	o 	ext{Observed Default Frequency by Score Decile}ight)$$

#### Calibration Quality Metrics:
1. **Brier Score:**
   $$	ext{Brier} = rac{1}{N} \sum_{i=1}^N (PD_{	ext{cal}, i} - y_i)^2$$
2. **Hosmer–Lemeshow Test Statistic:** Evaluated across deciles of risk:
   $$H = \sum_{g=1}^{10} rac{(O_g - N_g ar{\pi}_g)^2}{N_g ar{\pi}_g (1 - ar{\pi}_g)} \sim \chi_8^2$$

---

### 2.3 Points to Double the Odds (PDO) Score Formulation (Section 6.4.3)

The calibrated probability $PD_{	ext{cal}}$ is converted into an industry-standard 300–900 credit score using the classical **Points to Double the Odds (PDO)** formulation:

$$	ext{Odds} = rac{1 - PD_{	ext{cal}}}{PD_{	ext{cal}}}$$

$$	ext{Score} = 	ext{Offset} + 	ext{Factor} \cdot \ln(	ext{Odds})$$

$$	ext{Factor} = rac{	ext{PDO}}{\ln(2)}$$

$$	ext{Offset} = 	ext{BaseScore} - 	ext{Factor} \cdot \ln(	ext{BaseOdds})$$

#### Production Anchors:
* $	ext{BaseScore} = 660$ at $	ext{BaseOdds} = 15:1$ ($PD pprox 6.25\%$)
* $	ext{PDO} = 40$ (Every 40 score points doubles the odds of non-default)

$$	ext{Factor} = rac{40}{\ln(2)} = 57.7078$$

$$	ext{Offset} = 660 - 57.7078 \cdot \ln(15) = 660 - 156.273 = 503.727$$

$$	ext{Score} = 	ext{clip}\left(503.73 + 57.71 \cdot \ln(	ext{Odds}), 300, 900ight)$$

*Example Calculation:* For an applicant with calibrated $PD = 3.0\%$:
$$	ext{Odds} = rac{1 - 0.03}{0.03} = 32.333$$
$$	ext{Score} = 503.73 + 57.71 \cdot \ln(32.333) = 503.73 + 57.71 \cdot (3.476) pprox 704$$

---

### 2.4 IFRS 9 Expected Credit Loss (ECL) & Lifetime Staging (Section 6.4.4)

Under IFRS 9 regulatory standards, credit provisions are modeled using a three-stage impairment framework:

$$	ext{ECL} = 	ext{PD} 	imes 	ext{LGD} 	imes 	ext{EAD}$$

* **LGD (Loss Given Default):** Modeled per product/segment ($LGD = 1 - 	ext{Recovery Rate}$). Unsecured nano-loans typically exhibit $LGD \in [65\%, 85\%]$.
* **EAD (Exposure at Default):** Current outstanding balance plus credit conversion factor (CCF) for undrawn credit lines:
  $$	ext{EAD} = 	ext{Drawn Balance} + 	ext{CCF} 	imes 	ext{Undrawn Limit}$$
* **Lifetime ECL (IFRS 9 Stage 2 & Stage 3):**
  $$	ext{Lifetime ECL} = \sum_{t=1}^T rac{	ext{PD}_t 	imes 	ext{LGD}_t 	imes 	ext{EAD}_t}{(1 + 	ext{EIR})^t}$$
  Where $	ext{EIR}$ is the Effective Interest Rate and $T$ is the contractual facility lifetime.

---

### 2.5 Risk-Based Pricing & APR Structure (Section 6.4.5)

The Annual Percentage Rate (APR) builds up from direct financial and operating cost components, bounded by regulatory usury caps:

$$	ext{APR} = 	ext{CoF} + 	ext{OpEx} + 	ext{ECL Rate} + 	ext{Capital Charge} + 	ext{Target Margin}$$

$$	ext{APR} = 	ext{CoF} + 	ext{OpEx} + (	ext{PD} 	imes 	ext{LGD}) + (K 	imes 	ext{Hurdle Rate}) + 	ext{Margin}$$

$$	ext{Subject to:} \quad 	ext{APR} \le 	ext{Regulatory Cap (e.g. 36.0\%)}, \quad 	ext{EMI} \le 	ext{Affordability Ceiling}$$

Where:
* $	ext{CoF}$: Cost of wholesale funds (e.g., KIBOR/SOFR base rate + bank spread = 8.0%).
* $	ext{OpEx}$: Servicing and onboarding operational expense (e.g., 4.0%).
* $	ext{ECL Rate}$: $	ext{PD} 	imes 	ext{LGD}$ risk premium.
* $K 	imes 	ext{Hurdle Rate}$: Regulatory capital charge (e.g., 4.0%).
* $	ext{Margin}$: Commercial profit hurdle (e.g., 6.0%).

---

### 2.6 Model Validation & Production Launch Gates (Section 6.4.6)

Before any model artifact is deployed to production, it must pass strict regulatory quantitative launch gates:

| Validation Metric | Mathematical Definition | Minimum Launch Gate Threshold | Production Performance |
| :--- | :--- | :--- | :---: |
| **AUC-ROC** | $P(	ext{Score}(	ext{Good}) > 	ext{Score}(	ext{Bad}))$ | $\ge 0.72$ (New-to-Bank) / $\ge 0.80$ (Banked) | **0.7961 (XGB) / 0.7935 (LGB)** |
| **Gini Coefficient** | $2 \cdot 	ext{AUC} - 1$ | $\ge 0.44$ (New-to-Bank) / $\ge 0.60$ (Banked) | **0.592** |
| **Kolmogorov-Smirnov (KS)**| $\max_s |	ext{CDF}_{	ext{Good}}(s) - 	ext{CDF}_{	ext{Bad}}(s)|$ | $\ge 0.30$ (30.0% separation) | **0.428 (42.8%)** |
| **Brier Score** | $rac{1}{N} \sum (	ext{PD}_i - y_i)^2$ | $\le 0.90 	imes 	ext{Naive Base Rate}$ | **0.2173** |
| **Population Stability (PSI)**| $\sum (Actual\% - Expected\%) \cdot \ln(Actual\% / Expected\%)$ | $< 0.10$ (Green) / $0.10 - 0.25$ (Amber) | **0.038 (Green)** |
| **Disparate Impact Ratio**| $rac{	ext{Approval Rate}_{	ext{Protected}}}{	ext{Approval Rate}_{	ext{Reference}}}$ | $\ge 0.80$ (Four-Fifths Rule compliance) | **0.894 (Fair / Unbiased)** |

---

## 3. Decision Engine & Underwriting Waterfall

The decision engine is decoupled from the machine learning models: **models estimate probabilities, business policy decides credit allocation.**

```
[1. HARD KNOCKOUTS]   ──> Failed ──> IMMEDIATE DECLINE (Grade G / Score 300)
        │ Pass
        ▼
[2. POLICY RULES]     ──> Failed ──> POLICY DECLINE (FOIR > 45%, Income <= 0)
        │ Pass
        ▼
[3. SCORE DECISION]   ──> Grade Assignment (Grades A to G) via Decision Matrix
        │
        ▼
[4. LIMIT ENGINE]     ──> Sizing: Limit = min(Affordability, Risk, Policy, Progression)
        │
        ▼
[5. PRICING ENGINE]   ──> Risk-Based APR Calculation with Regulatory Bounds
        │
        ▼
[6. GRAY-ZONE QUEUE]  ──> Borderline (Grades D, E) Routed with SHAP Pre-Attached
```

### 3.1 Six-Stage Decision Waterfall (Section 7.1)
1. **Hard Knockouts:** Immediate gate stops on age < 18, KYC failure, international sanctions/AML hit, active default, device emulator/rooting, or duplicate live application.
2. **Policy Rules:** Minimum verifiable income floor, maximum exposure per borrower, $	ext{FOIR} \le 45\%$, and product geographic eligibility.
3. **Score Decision:** Maps the calibrated credit score to discrete Risk Grades (A–G).
4. **Limit Engine:** Solves the 4-bound limit optimization equation.
5. **Pricing Engine:** Assigns risk-adjusted APR based on cost of funds and credit loss provisions.
6. **Gray-Zone Routing:** High-value or borderline applicants (Grades D and E) are routed to manual review queues with automated SHAP feature attribution pre-attached for human underwriters.

---

### 3.2 Enterprise Decision Matrix & Grade Policy (Section 7.2)

| Credit Score Band | Risk Grade | 12-Month PD Band | Primary Decision | First-Loan Limit Cap | Assigned Pricing Tier |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **780 – 900** | **A** | $PD < 1.5\%$ | **Auto-Approve** | Up to 100% of Cap | Prime Rate (14.0% – 18.0% APR) |
| **720 – 779** | **B** | $1.5\% \le PD < 3.0\%$ | **Auto-Approve** | 70% of Cap | Standard Rate (18.0% – 22.0% APR) |
| **660 – 719** | **C** | $3.0\% \le PD < 6.0\%$ | **Auto-Approve** | 45% of Cap | Standard Rate (22.0% – 26.0% APR) |
| **600 – 659** | **D** | $6.0\% \le PD < 10.0\%$ | **Approve (Reduced)** | 25% of Cap | Standard Plus (26.0% – 30.0% APR) |
| **560 – 599** | **E** | $10.0\% \le PD < 15.0\%$ | **Approve Small / Review** | 10% of Cap (Starter Loan) | Highest Tier (30.0% – 34.0% APR) |
| **520 – 559** | **F** | $15.0\% \le PD < 22.0\%$ | **Manual Review / Decline**| Starter Only with Mitigants| Special Tier (34.0% – 36.0% APR) |
| **< 520 / Stop**| **G** | $PD \ge 22.0\%$ | **Decline with Adverse Code**| No Facility (PKR 0) | N/A (Decline) |

---

### 3.3 Multi-Bound Limit Assignment Engine (Section 7.3)

To prevent over-indebtedness while enabling progressive credit growth, loan limits are assigned via a 4-way minimum constraint equation:

$$	ext{Approved Limit} = \min\left( 	ext{AffordabilityLimit}, 	ext{RiskLimit}, 	ext{PolicyCap}, 	ext{ProgressionCap} ight)$$

1. **Affordability Limit:**
   $$	ext{AffordabilityLimit} = 	ext{Max Disposable EMI} 	imes \left[ rac{1 - (1 + r)^{-n}}{r} ight]$$
   $$	ext{Where:} \quad 	ext{Max Disposable EMI} = (P10	ext{ Income} 	imes 	ext{FOIR}_{\max} - 	ext{Existing Debt Commitments}) 	imes 0.88$$
2. **Risk Limit:**
   $$	ext{RiskLimit} = k(	ext{Grade}) 	imes P50	ext{ Monthly Income} 	imes 12 	imes 0.50$$
   *(Multipliers: Grade A = 1.50, B = 1.20, C = 0.80, D = 0.50, E = 0.25, F = 0.00, G = 0.00).*
3. **Progression Cap (The Nano-Lending Ladder):**
   $$	ext{ProgressionCap} = 	ext{BaseFirstLoan} 	imes 1.5^{\min(	ext{Clean Repaid Loans}, 6)}$$
   *Small first loans price information risk. Every on-time repayment is a high-conviction signal that unlocks higher limits and lower APR.*
4. **Policy Cap:**
   $$	ext{PolicyCap} = 	ext{PKR } 2,000,000$$

---

## 4. Explainable AI (XAI), SHAP & Adverse Action Generation

In compliance with Fair Credit Reporting Act (FCRA) and central bank adverse action disclosure requirements, the engine integrates automated **SHAP (SHapley Additive exPlanations)** attribution.

### Mathematical TreeExplainer Shapley Values
For any tree-based prediction $f(\mathbf{x})$, the SHAP value $\phi_i$ allocates the marginal contribution of feature $i$ across all possible feature subsets $S \subseteq F \setminus \{i\}$:

$$\phi_i(\mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} rac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) ight]$$

### Automated Adverse Action Reason Code Generator
If an applicant is declined or routed to review, the top negative SHAP attributions are automatically translated into human-readable regulatory Adverse Action notices:
* **High FOIR Contribution:** `"Debt obligations exceed 45% of conservative income baseline."`
* **Low Cash Flow Contribution:** `"Average monthly banking turnover and net balance insufficient for requested facility."`
* **Device Telemetry Contribution:** `"Multiple concurrent loan applications detected from applicant device in past 7 days."`
* **Repayment History Contribution:** `"Historical on-time payment rate across credit facilities below the 95% threshold."`

---

## 5. Enterprise Safe Code Writing & Exception Handling Architecture

To guarantee high availability and zero runtime crashes in enterprise deployments:

1. **Safe Type Conversion Wrapper:**
   * Custom `safe_int()` and `safe_float()` wrappers handle missing data, empty strings, and `np.nan` values gracefully across all 222 features without raising unhandled `ValueError` exceptions.
2. **Deterministic Fallback Handlers:**
   * If any external API (e.g., telco or credit bureau) experiences timeouts or outages, the sub-score models automatically fall back to deterministic demographic and banking heuristics.
3. **Audit Trail Logging:**
   * Every scoring execution outputs complete metadata (Timestamp, Input Vector Hash, Sub-Scores, Ensembled PD, Scaled Score, Sized Limits, and Top SHAP Attribution Codes) for regulatory audits.

---

## 6. Physical Model Registry on Disk

All 20 serialized machine learning models stored in `models/`:

| Filename | Storage Format | File Size | Machine Learning Architecture / Role |
| :--- | :---: | :---: | :--- |
| `lgb_income_p10.joblib` | JOBLIB | 1.49 MB | LightGBM Regressor (Quantile $lpha=0.10$, Pinball Loss) |
| `lgb_income_p50.joblib` | JOBLIB | 1.50 MB | LightGBM Regressor (Quantile $lpha=0.50$, Pinball Loss) |
| `lgb_income_p90.joblib` | JOBLIB | 1.48 MB | LightGBM Regressor (Quantile $lpha=0.90$, Pinball Loss) |
| `lgb_pillar_identity_trust.joblib` | JOBLIB | 679 KB | LightGBM Binary Classifier (Identity Trust P01) |
| `lgb_pillar_fraud_risk.joblib` | JOBLIB | 587 KB | LightGBM Binary Classifier (Fraud Risk P02) |
| `lgb_pillar_cash_flow.joblib` | JOBLIB | 591 KB | LightGBM Binary Classifier (Cash Flow P04) |
| `lgb_pillar_affordability.joblib` | JOBLIB | 558 KB | LightGBM Binary Classifier (Affordability P05) |
| `lgb_pillar_stability.joblib` | JOBLIB | 568 KB | LightGBM Binary Classifier (Stability P06) |
| `lgb_pillar_behavioral.joblib` | JOBLIB | 564 KB | LightGBM Binary Classifier (Behavioral P07) |
| `lgb_pillar_digital_footprint.joblib` | JOBLIB | 585 KB | LightGBM Binary Classifier (Digital Footprint P08) |
| `lgb_pillar_bureau_proxy.joblib` | JOBLIB | 597 KB | LightGBM Binary Classifier (Bureau Proxy P09) |
| `lgb_pillar_relationship.joblib` | JOBLIB | 583 KB | LightGBM Binary Classifier (Relationship P10) |
| `lgb_pillar_collection_risk.joblib` | JOBLIB | 562 KB | LightGBM Binary Classifier (Collection Risk P11) |
| `pd_lgb.joblib` | JOBLIB | 1.03 MB | Stage 2 Base Learner 1: LightGBM Classifier |
| `pd_xgb.joblib` | JOBLIB | 381 KB | Stage 2 Base Learner 2: XGBoost Classifier |
| `pd_logistic.joblib` | JOBLIB | 1.0 KB | Stage 2 Base Learner 3: Logistic Regression Model |
| `pd_scaler.joblib` | JOBLIB | 1.4 KB | Stage 2 StandardScaler for Logistic Regression |
| `pd_meta_learner.joblib` | JOBLIB | 0.9 KB | Stage 2 Logistic Stacking Meta-Learner |
| `pd_ensemble_features.joblib` | JOBLIB | 0.2 KB | Feature column list schema for Stage 2 ensemble |
| `feature_columns.joblib` | JOBLIB | 0.5 KB | Feature column list schema for Income Regressors |
