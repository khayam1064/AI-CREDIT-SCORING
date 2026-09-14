import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    '?? Borrower Application Journey (5-Screen Low-Friction Digital Lending Flow)': '📱 Borrower Application Journey (5-Screen Low-Friction Digital Lending Flow)',
    '??? Enterprise Underwriter & Risk Operations Console (7 Comprehensive Tabs)': '💼 Enterprise Underwriter & Risk Operations Console (7 Comprehensive Tabs)',
    'page_icon="??"': 'page_icon="🏦"',
    '<div style=\'font-size:2rem;\'>??</div>': '<div style=\'font-size:2rem;\'>🏦</div>',
    '"?? Borrower Application Journey (5 Screens)"': '"📱 Borrower Application Journey (5 Screens)"',
    '"??? Enterprise Underwriter Console"': '"💼 Enterprise Underwriter Console"',
    '"?? Portfolio Intelligence & IFRS 9"': '"📊 Portfolio Intelligence & IFRS 9"',
    '"?? Real-Time Underwriting & Deep Dive"': '"🔍 Real-Time Underwriting & Deep Dive"',
    '"?? Quantile Income & Capacity Sizing"': '"💵 Quantile Income & Capacity Sizing"',
    '"?? Explainable AI (SHAP & Adverse Action)"': '"🧠 Explainable AI (SHAP & Adverse Action)"',
    '"?? Model Mechanics & Physical Registry"': '"⚙️ Model Mechanics & Physical Registry"',
    '"?? Policy Simulator & Champion/Challenger"': '"🎛️ Policy Simulator & Champion/Challenger"',
    '"??? 15-Family Master Data Taxonomy"': '"🗂️ 15-Family Master Data Taxonomy"',
    '"?? Search ID or Full Name:"': '"🔍 Search ID or Full Name:"',
    '"?? Quick Presets & Filters"': '"💡 Quick Presets & Filters"',
    '### ?? Demo Applicant Profile': '### 👤 Demo Applicant Profile',
    '?? Digital Lending Application Journey': '📱 Digital Lending Application Journey',
    '"?? Mobile Number:"': '"📱 Mobile Number:"',
    '"?? 6-Digit SMS Code:"': '"🔢 6-Digit SMS Code:"',
    '"Verify OTP & Continue ??"': '"Verify OTP & Continue ➡️"',
    '**?? Silent Telemetry Readout:**': '**🛡️ Silent Telemetry Readout:**',
    '**?? National ID (CNIC) Front & Back:**': '**🪪 National ID (CNIC) Front & Back:**',
    '?? Simulated ID Capture:': '✅ Simulated ID Capture:',
    '**?? OCR Extracted Fields:**': '**📋 OCR Extracted Fields:**',
    '**?? Face Match & Passive Liveness:**': '**📸 Face Match & Passive Liveness:**',
    '"Confirm Identity Details ??"': '"Confirm Identity Details ➡️"',
    '"Save Answers & Continue ??"': '"Save Answers & Continue ➡️"',
    '**?? In-App Behavioral Telemetry:**': '**⚡ In-App Behavioral Telemetry:**',
    '"?? Connect Bank Account / Open Banking (Unlocks up to 5x Limit Multiplier)"': '"🏦 Connect Bank Account / Open Banking (Unlocks up to 5x Limit Multiplier)"',
    '"?? Connect Telecom Data API (Unlocks 24-Month Flexible Tenors)"': '"📡 Connect Telecom Data API (Unlocks 24-Month Flexible Tenors)"',
    '"?? Connect Utility & Rent History (Unlocks up to 3% APR Discount)"': '"💡 Connect Utility & Rent History (Unlocks up to 3% APR Discount)"',
    '?? LIVE UNLOCKED CREDIT LIMIT': '💰 LIVE UNLOCKED CREDIT LIMIT',
    '"Generate Final Loan Offer ??"': '"Generate Final Loan Offer 🚀"',
    '"?? Choose Loan Amount (PKR):"': '"💵 Choose Loan Amount (PKR):"',
    '"?? Repayment Tenor:"': '"📅 Repayment Tenor:"',
    '**??? Cost & Pricing Transparency:**': '**📊 Cost & Pricing Transparency:**',
    '**?? Digital e-Signature:**': '**✍️ Digital e-Signature:**',
    '"?? Accept Offer & Disburse Funds Now"': '"🎉 Accept Offer & Disburse Funds Now"',
    '?? Loan of PKR': '🎉 Loan of PKR',
    'k3.metric("?? Refer"': 'k3.metric("⚠️ Refer"',
    '### ??? IFRS 9 Expected Credit Loss (ECL) Provisioning Breakdown': '### 📉 IFRS 9 Expected Credit Loss (ECL) Provisioning Breakdown',
    'b_icon = {"APPROVED": "?", "REFER": "??", "DECLINED": "?"}': 'b_icon = {"APPROVED": "✅", "REFER": "⚠️", "DECLINED": "❌"}',
    '### ??? Raw & Derived Data Family Signals': '### 📑 Raw & Derived Data Family Signals',
    'st.tabs(["?? Identity & Demo", "?? Banking & CashFlow", "?? Telco & Wallet", "?? Device & Biometrics", "?? Repayment & Velocity"])': 'st.tabs(["🪪 Identity & Demo", "🏦 Banking & CashFlow", "📡 Telco & Wallet", "📲 Device & Biometrics", "🤝 Repayment & Velocity"])',
    '### ?? Interactive Underwriting Sizing Simulator': '### 🎛️ Interactive Underwriting Sizing Simulator',
    '### ?? Top Positive Drivers (Trust Enhancers)': '### 🟢 Top Positive Drivers (Trust Enhancers)',
    '### ?? Top Risk Drivers (Adverse Action Factors)': '### 🔴 Top Risk Drivers (Adverse Action Factors)',
    '### ?? Formal Regulatory Adverse Action Notice': '### 📜 Formal Regulatory Adverse Action Notice',
    '### ?? Score Calibration Math: Log-Odds Transformation': '### 📈 Score Calibration Math: Log-Odds Transformation',
    'st.info(f"?? Under policy cutoff': 'st.info(f"💡 Under policy cutoff',
    '? `data/processed/credit_scores_250k.parquet` not found.': '❌ `data/processed/credit_scores_250k.parquet` not found.'
}

for k, v in replacements.items():
    text = text.replace(k, v)

# Also replace any stray ?? or ??? in common markdown headings or strings
text = text.replace('### ??? ', '### 📊 ')
text = text.replace('### ?? ', '### 🔍 ')
text = text.replace('**?? ', '**⚡ ')
text = text.replace('?? ', '🔹 ')

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully cleaned and replaced all corrupted question mark characters in dashboard/app.py!')
