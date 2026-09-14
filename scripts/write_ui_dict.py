import os

doc2_content = """# Enterprise Credit AI Dashboard — UI Layout, Visualizations & Metrics Dictionary

**Author:** Enterprise AI Underwriting Team  
**Scope:** Complete User Interface Specification, Metric Definitions, Color Codings, and Statistical Interpretations  
**Dashboard Access:** `http://localhost:8502`  

---

## Table of Contents
1. [Global Sidebar & Customer Selection Controls](#1-global-sidebar--customer-selection-controls)
2. [Page 1: Portfolio Overview & IFRS 9 Credit Intelligence](#2-page-1-portfolio-overview--ifrs-9-credit-intelligence)
3. [Page 2: Customer Deep Dive & Feature Profile](#3-page-2-customer-deep-dive--feature-profile)
4. [Page 3: Underwriting, Limits Sizing & Risk-Based Pricing](#4-page-3-underwriting-limits-sizing--risk-based-pricing)
5. [Page 4: Explainable AI (XAI) & Regulatory Adverse Action Generator](#5-page-4-explainable-ai-xai--regulatory-adverse-action-generator)
6. [Page 5: Model Mechanics & Calibration](#6-page-5-model-mechanics--calibration)
7. [Page 6: Data Taxonomy](#7-page-6-data-taxonomy)
8. [Glossary of Abbreviations & Regulatory Reference Standards](#8-glossary-of-abbreviations--regulatory-reference-standards)

---

## 1. Global Sidebar & Customer Selection Controls

The global sidebar (left panel) persists across all dashboard views, controlling applicant selection and real-time execution.

### UI Controls & Features:
1. **Search Input Bar (`🔍 Search Customer (ID or Name)`):**
   * **Exact ID Input:** Accepts full customer IDs (e.g. `CUST0000001`, `CUST0000022`) or raw numeric integers (e.g. typing `22` automatically normalizes to `CUST0000022`).
   * **Fuzzy Name Search:** If a non-numeric string is entered (e.g., `Hassan`, `Tariq`, `Naqvi`), the engine performs a fuzzy sub-string lookup against the customer database name registry and automatically loads the matching record.
2. **Quick Presets (`💡 Quick Presets & Filters`):**
   * Pre-configured representative underwriting edge cases for one-click review:
     * `CUST0000001`: Approved (Grade A) — Prime corporate earner with low DPD and strong cash flow.
     * `CUST0000002`: Declined (Grade G) — Failed Stage 0 FOIR Gate (>45% obligations).
     * `CUST0000003`: Approved (Grade A) — Long-standing account with flawless on-time payment rate.
     * `CUST0000004`: Declined (Grade G) — Failed Stage 0 Zero Income Gate.
     * `CUST0000022`: Declined (Grade G) — Pure Zero Income profile.
     * `CUST0000090`: Approved (Grade A) — Prime profile with high stability.
     * `CUST0000100`: Refer (Grade C) — Borderline score requiring manual underwriting review.
3. **Dynamic Database Filters:**
   * **Segment Selector:** Filter by `Mass`, `Affluent`, `Premium`, `Private Banking`.
   * **Decision Selector:** Filter by `APPROVED`, `REFER`, `DECLINED`.
   * **Grade Selector:** Filter by risk grades `A`, `B`, `C`, `D`, `E`, `F`, `G`.

---

## 2. Page 1: Portfolio Overview & IFRS 9 Credit Intelligence

Provides executive-level monitoring across all **250,000 scored applicants** and portfolio-wide IFRS 9 ECL provisions.

### 2.1 Top KPI Metrics Cards

| Metric Tile | Typical Value | Mathematical Definition / Meaning |
| :--- | :---: | :--- |
| **Total Customers** | **250,000** | Total applicant population scored in the database. |
| **✅ Approved** | **150,668 (60.3%)** | Total applicants receiving an `APPROVED` decision (Grades A, B, C, D). |
| **⚠️ Refer** | **4,607 (1.8%)** | Borderline applicants (Grades E, F) routed for manual underwriting review. |
| **❌ Declined** | **94,725 (37.9%)** | Total rejected applicants (Grade G: Stage 0 hard stops and extreme credit risk). |
| **Median Score** | **659 / 900** | The 50th percentile credit score across all scored applicants (Mean is 528). |

### 2.2 Visualizations & Charts
1. **Decision Distribution (Donut Chart):**
   * Visualizes the proportion of Approved (Green: `#10b981`), Refer (Amber: `#f59e0b`), and Declined (Red: `#ef4444`) with the total applicant volume in the center hole.
2. **Credit Grade Distribution (Bar Chart):**
   * Displays customer counts binned into the 7 risk grades: **A** (780+), **B** (720-779), **C** (660-719), **D** (600-659), **E** (560-599), **F** (520-559), and **G** (<520 / Stage 0 Refusals).
3. **Credit Score Distribution (Histogram):**
   * Continuous frequency distribution of scores from 300 to 900. Shows a peak at 300 (hard-stop declines) and a normal distribution across approved applicants (600–780).
4. **Top Decline Reasons Breakdown (Bar Chart):**
   * Quantifies the primary reasons for application rejection:
     * `FOIR_EXCEEDS_45PCT`: Debt obligations exceed 45% of conservative income.
     * `FRAUD_FLAG`: Device rooting, emulator, or velocity fraud detected.
     * `COMPLIANCE_FAIL`: Failed PEP, AML, or Sanctions screening.
     * `ZERO_INCOME`: No detectable or verifiable income inflow.
5. **IFRS 9 Portfolio Provisioning Overview:**
   * **Total Portfolio Exposure:** Total aggregate exposure under management ($EAD$).
   * **Total ECL Provision:** Total estimated credit loss across all stages in PKR.
   * **ECL Coverage Ratio:** $\text{Total ECL} / \text{Total Exposure} \times 100$.
   * **Staging Breakdown Table:**
     * **Stage 1 (Performing):** 12-Month ECL for low-risk active loans (Grades A, B, C).
     * **Stage 2 (SICR):** Lifetime ECL for accounts with Significant Increase in Credit Risk (Grades D, E).
     * **Stage 3 (Credit Impaired):** Full default provisions for defaulted/refused accounts (Grades F, G).

---

## 3. Page 2: Customer Deep Dive & Feature Profile

Executes **live, on-the-fly machine learning scoring** for the selected applicant and renders a complete 360-degree risk profile.

### 3.1 Decision Banner & High-Level KPIs
* **Decision Banner:**
  * **Approved:** Green banner with badge `✅ Credit Decision: APPROVED (Score: 818, Grade: A)`.
  * **Refer:** Amber banner with badge `⚠️ Credit Decision: REFER (Score: 540, Grade: E)`.
  * **Declined:** Red banner with badge `❌ Credit Decision: DECLINED (Score: 300, Grade: G — Reason: ZERO_INCOME)`.
* **Summary KPI Cards (5 Cards):**
  1. **Composite Score:** 300 to 900 score and risk grade badge (A–G).
  2. **Calibrated Default Probability (PD):** Ensembled default probability (e.g. `4.6%`).
  3. **Expected Monthly Income (P50):** Median predicted monthly income in PKR.
  4. **Max Affordable EMI:** Maximum monthly installment applicant can pay safely.
  5. **FOIR / DSR Utilization:** Existing debt obligations as a % of conservative income.

### 3.2 Visual Risk Diagnostics
* **12-Pillar Radar Chart:**
  * A 12-axis spider web chart plotting the applicant's score (0–100) across all 12 dimensions against the outer 100 benchmark.
* **Pillar Breakdown Horizontal Bar Chart:**
  * Color-coded horizontal bars:
    * **Green (`>= 70`):** Strong, low-risk signal.
    * **Amber (`50 – 69`):** Moderate risk signal.
    * **Red (`< 50`):** High risk / deficient signal.

### 3.3 The 5 Data Family Tab Panels

#### Tab 1: 🪪 Identity & Demographics (Families F1, F2, F11)
* **Age:** Applicant age in years.
* **Gender:** Male / Female / Other.
* **Education Level:** Matric / Intermediate / Bachelor / Master / Doctorate.
* **Marital Status:** Single / Married / Divorced / Widowed.
* **Nationality:** Country of citizenship (Pakistani / Non-Resident).
* **CNIC Verification:** Automated NADRA algorithm format and check-digit validity status.
* **Email Age (Days):** Tenure of the email account (older emails indicate authentic identity).
* **SIM Age (Days):** Tenure of mobile phone subscription (proxy for residential stability).
* **Email Provider Domain:** Domain classification (e.g. `gmail.com`, `yahoo.com`, custom corporate).
* **Digital Maturity Index (0–100):** Composite score evaluating digital engagement depth.

#### Tab 2: 🏦 Banking & Income (Families F6, F8, F12)
* **Total Bank Accounts:** Number of verified open bank accounts across the banking sector.
* **Max Bank Account Tenure (Months):** Duration of oldest active bank relationship.
* **Avg Monthly Credit (Inflows):** Mean monthly credit turnover in PKR.
* **Avg Monthly Debit (Outflows):** Mean monthly debit turnover in PKR.
* **Utility Bills Paid (12M):** Total utility expenditure processed through bank/wallet in past year.
* **Total Average Monthly Balance:** 6-month average daily balance maintained across accounts.
* **Overdraft Limit / Used:** Overdraft credit facility sanctioned vs. utilized in PKR.
* **Overdraft Utilization Rate (%):** Proportion of overdraft line currently drawn.
* **NSF (Insufficient Funds) Counts:** Number of bounced cheques / failed auto-debits.

#### Tab 3: 📡 Telecom & Wallet (Families F5, F7)
* **Telecom Type:** Prepaid vs. Postpaid cellular subscription.
* **Avg Monthly Recharge:** Average monthly cellular top-up spend in PKR.
* **Telecom Credit Signal (0–100):** Machine learning score predicting creditworthiness from telco usage.
* **Telecom Porting Events:** Number of times the mobile number switched operators (frequent porting indicates instability).
* **Has Mobile Wallet:** Presence of an active EasyPaisa / JazzCash / Nayapay account.
* **Wallet Provider & Tenure:** Wallet operator and relationship age in months.
* **Wallet Avg Monthly Txn Value:** Monthly transaction turnover through mobile wallet.
* **Wallet Credit Signal (0–100):** Creditworthiness score derived from wallet transaction velocity.

#### Tab 4: 📲 Device & Biometrics (Families F3, F4, F10)
* **Device Manufacturer & OS:** Smartphone hardware brand and operating system version.
* **Device Rooted / Jailbroken:** Binary flag checking if OS security sandbox has been compromised.
* **Emulator Detected:** Binary flag checking if applicant is using an Android PC emulator (high fraud risk).
* **VPN / Proxy Active:** Anonymizer / IP proxy detection.
* **Copy-Paste ID Fields:** Biometric flag checking if national ID was typed manually or pasted from clipboard.
* **Session Count (7d):** Application interactions logged in past 7 days.
* **Session Replay Anomaly Score (0–100):** Machine learning score identifying bot/scripted interactions.
* **Behavioral Risk Score (0–100):** In-app hesitation, mouse trajectory, and form interaction risk.
* **Location Region Risk:** Geo-spatial default risk index for applicant's residence district.

#### Tab 5: 🤝 Repayment & Velocity (Families F13, F14)
* **Is Existing Loan Customer:** Presence of prior borrowing history with the financial institution.
* **Prior Loans Count / Active Loans Count:** Number of closed vs. ongoing loan facilities.
* **On-Time Payment Rate (%):** Historical percentage of EMI installments paid on or before due date.
* **Max DPD (Days Past Due) Ever:** Highest delinquency recorded in applicant history (0 = clean).
* **Relationship Score (0–100):** Internal customer loyalty and repayment score.
* **Apps from Same Device (7d):** Total loan applications submitted from this device in past week.
* **Apps from Same IP (24h):** Total loan applications submitted from this IP address in past 24 hours.
* **Cross-Lender Velocity (30d):** Applications submitted across multi-lender credit network in 30 days.
* **Form Abandonment Count:** Number of times applicant dropped off before completing application.
* **Application Velocity Risk Score (0–100):** Risk score quantifying loan stacking or desperation behavior.

---

## 4. Page 3: Underwriting, Limits Sizing & Risk-Based Pricing

Integrates **Quantile LightGBM Income Modeling**, the **4-Bound Limit Engine**, and **Risk-Based APR Pricing**.

### 4.1 Income Quantiles Display
* **P10 Conservative Income (PKR):** 10th percentile downside income. Used to calculate FOIR to prevent predatory over-lending.
* **P50 Median Expected Income (PKR):** 50th percentile median expected income. Used to size primary loan limits.
* **P90 Optimistic Income (PKR):** 90th percentile upside earning potential.

### 4.2 Sizing Engine & Pricing Breakdown Cards
1. **Existing Obligations (PKR/mo):** Monthly debt and utility commitments.
2. **FOIR % (Fixed Obligation to Income Ratio):** Obligations / P10 Income (Max Cap: 45%).
3. **Max Disposable EMI (PKR/mo):** Maximum safe monthly installment capacity.
4. **Approved Term Loan Limit (PKR):** Sized via $\min(\text{Affordability}, \text{Risk}, \text{Policy}, \text{Progression})$.
5. **Revolving Credit Card Limit (PKR):** Sized credit card line capped at $2 \times P50$ income.
6. **Risk-Based APR (%):** Calculated as $\text{CoF} + \text{OpEx} + (\text{PD} \times \text{LGD}) + \text{Capital Charge} + \text{Margin}$.

### 4.3 Pricing Cost Waterfall Table
* **Cost of Funds (CoF):** Wholesale cost of capital (8.0%).
* **Operating Expense (OpEx):** Servicing and customer acquisition cost (4.0%).
* **ECL Risk Premium:** Expected credit loss provision ($\text{PD} \times \text{LGD}$).
* **Capital Charge:** Regulatory capital charge for risk-weighted assets (4.0%).
* **Target Margin:** Hurdle profit margin (6.0%).

---

## 5. Page 4: Explainable AI (XAI) & Regulatory Adverse Action Generator

Dedicated to model transparency, feature attributions, and FCRA-compliant adverse action disclosures.

### UI Features:
1. **Top Positive Drivers (Trust Enhancers):**
   * Green card indicators highlighting the top 4 factors improving the applicant's credit score (e.g. NADRA verified identity, low overdraft utilization, clean on-time repayment history).
2. **Top Risk Drivers (Adverse Action Factors):**
   * Red card indicators highlighting the top 4 factors contributing to elevated default risk.
   * Directly maps to regulatory Adverse Action Reason Codes suitable for customer notification letters.
3. **SHAP Feature Attribution Waterfall:**
   * Interactive chart detailing how each sub-score and raw variable shifts the model output relative to the baseline population average.

---

## 6. Page 5: Model Mechanics & Calibration

Provides technical transparency into the underlying **AI Architecture, Physical Model Registry, and Calibration Curves**.

### 6.1 Physical Model Registry Table
* Displays all 20 `.joblib` model artifacts, file formats, disk sizes, and assigned machine learning tasks.

### 6.2 Stage 2 PD Stacking Ensemble Performance Breakdown
* **LightGBM Base Learner:** ROC-AUC `0.7935`
* **XGBoost Base Learner:** ROC-AUC `0.7961`
* **Logistic Regression Base Learner:** ROC-AUC `0.7948`
* **Logistic Stacking Meta-Learner:** Final ROC-AUC `0.7893` | LogLoss `0.2173`

### 6.3 Points to Double the Odds (PDO) Calibration Curve
* Visual plot of the mathematical mapping: $\text{Score} = 503.73 + 57.71 \cdot \ln(\text{Odds})$.

---

## 7. Page 6: Data Taxonomy

Provides data governance and feature cataloging for all **15 Data Families (F1–F15)** and **222 Features**.

---

## 8. Glossary of Abbreviations & Regulatory Reference Standards

| Abbreviation | Full Term | Regulatory Definition |
| :--- | :--- | :--- |
| **PD** | Probability of Default | Statistical 12-month default probability (Basel III / IFRS 9). |
| **LGD** | Loss Given Default | Percentage of economic exposure lost upon borrower default. |
| **EAD** | Exposure at Default | Total expected financial exposure at the moment of default. |
| **ECL** | Expected Credit Loss | IFRS 9 required impairment reserve ($\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$). |
| **SICR** | Significant Increase in Credit Risk | IFRS 9 trigger transitioning loans from Stage 1 (12M ECL) to Stage 2 (Lifetime ECL). |
| **FOIR** | Fixed Obligation to Income Ratio | Regulatory debt burden ceiling ($\le 45.0\%$). |
| **PDO** | Points to Double the Odds | Scaling parameter where score delta doubles odds of non-default ($PDO = 40$). |
| **SHAP** | SHapley Additive exPlanations | Game-theoretic feature attribution method for machine learning explainability. |
| **STP** | Straight-Through Processing | Automated credit approval without manual underwriter intervention. |
"""

with open('DASHBOARD_UI_AND_METRICS_DICTIONARY.md', 'w', encoding='utf-8') as f:
    f.write(doc2_content)

print("Updated DASHBOARD_UI_AND_METRICS_DICTIONARY.md successfully!")
