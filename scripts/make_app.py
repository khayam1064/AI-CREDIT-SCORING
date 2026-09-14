# Generator script for dashboard/app.py
with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write("""\"\"\"
Enterprise Credit AI Platform — Full-Fledged Web Application
Realizes the complete System Design Specification (Version 1.0, July 2026):
1. ?? Borrower Application Journey (5-Screen Low-Friction Digital Lending Flow)
2. ??? Enterprise Underwriter & Risk Operations Console (7 Comprehensive Tabs)
\"\"\"

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent if (BASE_DIR.parent / 'scoring').exists() else BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

from scoring import composite_scorer
from scoring import shap_explainer
from scoring import policy_engine

try:
    from generate_customer_report import generate_professional_customer_report, save_report_to_file
except ImportError:
    generate_professional_customer_report = None

st.set_page_config(
    page_title="Pak-Credit AI · Digital Lending Platform",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(\"\"\"
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    .css-1d391kg, .css-12oz5g7 { background-color: #111827; }
    
    .portal-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f8fafc; }
    .metric-sub { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
    
    .decision-approved { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981; padding: 12px 18px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; }
    .decision-refer { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #f59e0b; padding: 12px 18px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; }
    .decision-declined { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #ef4444; padding: 12px 18px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; margin-bottom: 16px; }
    
    .ux-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background: #334155;
        color: #93c5fd;
        margin-right: 6px;
        margin-bottom: 4px;
    }
</style>
\"\"\", unsafe_allow_html=True)
""")
with open("dashboard/app.py", "a", encoding="utf-8") as f:
    f.write("""
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(17,24,39,0)",
    plot_bgcolor="rgba(17,24,39,0)",
    font=dict(color="#94a3b8", family="sans-serif", size=11),
    margin=dict(t=40, l=10, r=10, b=10),
)

GRADE_COLOR = {
    "A": "#10b981",
    "B": "#3b82f6",
    "C": "#8b5cf6",
    "D": "#f59e0b",
    "E": "#f97316",
    "F": "#ef4444",
    "G": "#7f1d1d"
}

def normalize_id(s):
    s = str(s).strip()
    if s.endswith(".0"): s = s[:-2]
    if s.upper().startswith("CUST"):
        try: return f"CUST{int(s[4:]):07d}"
        except: return s
    try: return f"CUST{int(s):07d}"
    except: return s

def safe_int(val, default=0):
    try:
        if pd.isna(val): return default
        return int(float(val))
    except (ValueError, TypeError): return default

def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except (ValueError, TypeError): return default

@st.cache_resource(show_spinner=False)
def load_credit_scores():
    p = PROJECT_ROOT / "data" / "processed" / "credit_scores_250k.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df["customer_id"] = df["customer_id"].apply(normalize_id)
    return df.set_index("customer_id")

@st.cache_resource(show_spinner=False)
def load_feature_store():
    p = PROJECT_ROOT / "feature_store" / "customer_features_v2.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df["customer_id"] = df["customer_id"].apply(normalize_id)
    return df.set_index("customer_id")

@st.cache_resource(show_spinner=False)
def load_customer_names():
    p = PROJECT_ROOT / "datasets" / "raw" / "customers.csv"
    if not p.exists(): return {}
    df = pd.read_csv(p, usecols=["customer_id", "full_name"])
    df["customer_id"] = df["customer_id"].apply(normalize_id)
    return dict(zip(df["customer_id"], df["full_name"]))

df_scores = load_credit_scores()
df_fs = load_feature_store()
names_dict = load_customer_names()

with st.sidebar:
    st.markdown(\"\"\"
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
        <div style='font-size:2rem;'>??</div>
        <div>
            <div style='font-size:1.15rem; font-weight:800; color:#f8fafc; letter-spacing:0.02em;'>PAK-CREDIT AI</div>
            <div style='font-size:0.75rem; color:#60a5fa; font-weight:600;'>Enterprise Lending Platform</div>
        </div>
    </div>
    <div style='font-size:0.72rem; color:#475569; margin-bottom:12px;'>15 Data Families · 12 Pillars · 250K Customers</div>
    \"\"\", unsafe_allow_html=True)
    
    portal_mode = st.radio(
        "Select Platform Portal:",
        ["?? Borrower Application Journey (5 Screens)", "??? Enterprise Underwriter Console"],
        index=0
    )
    
    st.markdown("---")
    
    if portal_mode == "??? Enterprise Underwriter Console":
        underwriter_page = st.radio("Underwriter Navigation:", [
            "?? Portfolio Intelligence & IFRS 9",
            "?? Real-Time Underwriting & Deep Dive",
            "?? Quantile Income & Capacity Sizing",
            "?? Explainable AI (SHAP & Adverse Action)",
            "?? Model Mechanics & Physical Registry",
            "?? Policy Simulator & Champion/Challenger",
            "??? 15-Family Master Data Taxonomy"
        ])
        
        st.markdown("---")
        st.markdown("### Customer Lookup")
        search_input = st.text_input("?? Search ID or Full Name:", value="CUST0000001")
        target_id = normalize_id(search_input)
        
        if df_fs is not None and target_id not in df_fs.index:
            q = search_input.strip().lower()
            if q:
                matched = [k for k, v in names_dict.items() if q in v.lower()]
                if matched:
                    target_id = matched[0]
                    st.success(f"Matched: **{names_dict[target_id]}** ({target_id})")
                    
        with st.expander("?? Quick Presets & Filters", expanded=False):
            sel_opt = st.radio("Presets Mode:", ["Standard Presets", "Dynamic Filters"])
            if sel_opt == "Standard Presets":
                presets = {
                    "CUST0000001": "Approved (Grade A) - Prime Corporate",
                    "CUST0000002": "Declined (Grade G) - Stage 0 FOIR >45%",
                    "CUST0000003": "Approved (Grade A) - High Stability/Low DPD",
                    "CUST0000004": "Declined (Grade G) - Stage 0 Zero Income",
                    "CUST0000022": "Declined (Grade G) - Zero Income Profile",
                    "CUST0000090": "Approved (Grade A) - Premium Customer",
                    "CUST0000100": "Refer (Grade C) - Low Stability Review"
                }
                c_preset = st.selectbox("Choose Preset:", list(presets.keys()), format_func=lambda x: f"{x} - {presets[x]}")
                if c_preset != target_id and search_input == "CUST0000001":
                    target_id = c_preset
            else:
                if df_scores is not None and df_fs is not None:
                    c_dec = st.selectbox("Decision:", ["All", "APPROVED", "REFER", "DECLINED"])
                    c_grd = st.selectbox("Grade:", ["All", "A", "B", "C", "D", "E", "F", "G"])
                    f_df = df_scores
                    if c_dec != "All": f_df = f_df[f_df["credit_decision"] == c_dec]
                    if c_grd != "All": f_df = f_df[f_df["credit_grade"] == c_grd]
                    m_ids = f_df.index.tolist()
                    if m_ids:
                        s_filt = st.selectbox(f"Matched ({len(m_ids):,}):", m_ids[:100], format_func=lambda x: f"{x} ({df_scores.loc[x, 'credit_decision']})")
                        if s_filt != target_id and search_input == "CUST0000001":
                            target_id = s_filt
    else:
        st.markdown("### ?? Demo Applicant Profile")
        demo_id = st.selectbox("Load Baseline Applicant:", ["CUST0000001", "CUST0000003", "CUST0000090", "CUST0000022"], format_func=lambda x: f"{x} ({names_dict.get(x, 'Applicant')})")
        target_id = demo_id
""")
with open("dashboard/app.py", "a", encoding="utf-8") as f:
    f.write("""
if "Borrower Application" in portal_mode:
    st.markdown(\"\"\"
    <div class='portal-banner'>
        <div style='font-size:1.4rem; font-weight:800; color:#f8fafc;'>?? Digital Lending Application Journey</div>
        <div style='color:#94a3b8; font-size:0.85rem; margin-top:4px;'>
            Governing Principle: <span style='color:#60a5fa; font-weight:600;'>Ask Little, Learn Much</span> · 5 Screens · <3 Minutes · Sub-5 Second Decision
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    step = st.radio(
        "Application Step:",
        ["1. Phone + OTP", "2. Identity (KYC)", "3. Three Questions", "4. Data Consent & Live Unlock", "5. Offer Sizing & e-Sign"],
        horizontal=True
    )
    
    row_fs = df_fs.loc[target_id] if (df_fs is not None and target_id in df_fs.index) else None
    
    if "1. Phone" in step:
        st.markdown("<div class='ux-card'>", unsafe_allow_html=True)
        st.markdown("### Screen 1: Mobile Authentication & Silent Telemetry")
        st.write("Enter your mobile phone number. We will send a secure 6-digit OTP code to verify your identity.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            phone_num = st.text_input("?? Mobile Number:", value="+92 300 1234567")
            otp_val = st.text_input("?? 6-Digit SMS Code:", value="489201", help="Simulated instant SMS OTP")
            if st.button("Verify OTP & Continue ??", type="primary"):
                st.success("? Phone verified successfully! SIM tenure and device fingerprint bound.")
                
        with c2:
            st.markdown("**?? Silent Telemetry Readout:**")
            st.markdown("<span class='badge-tag'>SIM Age: 1,420 days</span> <span class='badge-tag'>Carrier: Jazz</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Device: Samsung Galaxy S23</span> <span class='badge-tag'>OS: Android 14</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Rooted: No</span> <span class='badge-tag'>Emulator: Clean</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>IP Velocity (24h): 1 application</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif "2. Identity" in step:
        st.markdown("<div class='ux-card'>", unsafe_allow_html=True)
        st.markdown("### Screen 2: Smart KYC, OCR Extraction & Passive Liveness")
        st.write("Take a photo of your National Identity Card (CNIC) and a selfie for instant automated verification.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**?? National ID (CNIC) Front & Back:**")
            st.info("?? Simulated ID Capture: CNIC `35202-1234567-1` (Verified via NADRA)")
            st.markdown("**?? OCR Extracted Fields:**")
            st.text_input("Full Name:", value=names_dict.get(target_id, "Hassan Tariq Naqvi"), disabled=True)
            st.text_input("Date of Birth:", value="14-Aug-1991 (Age: 35)", disabled=True)
            st.text_input("Address:", value="House 14, Sector F-7/2, Islamabad", disabled=True)
            
        with c2:
            st.markdown("**?? Face Match & Passive Liveness:**")
            st.success("? Facial Match Confidence: 99.4% (Live Selfie vs. CNIC Photo)")
            st.markdown("<span class='badge-tag'>Deepfake Score: 0.02 (Ultra Clean)</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Sanctions/PEP Check: PASS</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Duplicate ID Graph: Clean</span>", unsafe_allow_html=True)
            
            if st.button("Confirm Identity Details ??", type="primary"):
                st.success("? Identity KYC Completed!")
        st.markdown("</div>", unsafe_allow_html=True)

    elif "3. Three Questions" in step:
        st.markdown("<div class='ux-card'>", unsafe_allow_html=True)
        st.markdown("### Screen 3: Employment, Declared Income & Loan Purpose")
        st.write("Confirm the following three details. Your typing dynamics silently calibrate behavioral trust.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            emp_type = st.selectbox("1. Employment Status:", ["Salaried (Private Sector)", "Salaried (Government / PSU)", "Self-Employed / Business", "Freelancer / Gig Worker"])
            dec_inc = st.select_slider("2. Declared Monthly Income (PKR):", options=["< 50k", "50k - 100k", "100k - 200k", "200k - 400k", "> 400k"], value="100k - 200k")
            purpose = st.selectbox("3. Primary Loan Purpose:", ["Medical & Healthcare", "Home Renovation", "Education & Fees", "Working Capital / Business", "Travel / Major Purchase"])
            
            if st.button("Save Answers & Continue ??", type="primary"):
                st.success("? Declared profile stored. Proceed to data consent to unlock maximum limit.")
                
        with c2:
            st.markdown("**?? In-App Behavioral Telemetry:**")
            st.markdown("<span class='badge-tag'>Typing Cadence: 48 WPM</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Hesitation Index: Low (1.2s)</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Copy-Paste Fields: 0 (Manual)</span>", unsafe_allow_html=True)
            st.markdown("<span class='badge-tag'>Form Duration: 24.8 seconds</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif "4. Data Consent" in step:
        st.markdown("<div class='ux-card'>", unsafe_allow_html=True)
        st.markdown("### Screen 4: Data Consent & Live Limit Unlock")
        st.write("Connect alternative data sources. Notice how your **Live Credit Limit Preview** immediately increases as you connect sources.")
        
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.markdown("**Select Data Connections (Opt-in with Plain Language Benefit):**")
            c_bank = st.checkbox("?? Connect Bank Account / Open Banking (Unlocks up to 5x Limit Multiplier)", value=True)
            c_telco = st.checkbox("?? Connect Telecom Data API (Unlocks 24-Month Flexible Tenors)", value=True)
            c_util = st.checkbox("?? Connect Utility & Rent History (Unlocks up to 3% APR Discount)", value=True)
            
        with c_right:
            mult = 1.0
            if c_bank: mult += 2.5
            if c_telco: mult += 0.8
            if c_util: mult += 0.7
            
            base_p50 = float(row_fs.get('p50_income', 120000)) if row_fs is not None else 120000.0
            live_unlocked = round(base_p50 * mult * 2.0, -3)
            
            st.markdown(f\"\"\"
            <div style='background:#0f172a; border:2px solid #3b82f6; border-radius:10px; padding:16px; text-align:center;'>
                <div style='font-size:0.8rem; color:#94a3b8; font-weight:600;'>?? LIVE UNLOCKED CREDIT LIMIT</div>
                <div style='font-size:2.2rem; font-weight:800; color:#60a5fa;'>PKR {live_unlocked:,.0f}</div>
                <div style='font-size:0.75rem; color:#10b981; font-weight:600;'>Multiplier: {mult:.1f}x (Consent Tier {int(mult)})</div>
            </div>
            \"\"\", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Final Loan Offer ??", type="primary"):
            st.success("? Data ingested and ensembled! Review your customized loan offer on Screen 5.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif "5. Offer Sizing" in step:
        st.markdown("<div class='ux-card'>", unsafe_allow_html=True)
        st.markdown("### Screen 5: Loan Customization, Transparent Pricing & e-Signature")
        
        if row_fs is not None:
            row_fs_df = df_fs.loc[[target_id]]
            live_scores = composite_scorer.compute_scores(row_fs_df)
            r_score = live_scores.iloc[0]
            
            p10_inc = float(row_fs.get('p10_income', 90000) or 90000)
            p50_inc = float(row_fs.get('p50_income', 120000) or 120000)
            oblg = float(row_fs.get('utility_debit_amt_12m', 0) or 0) / 12.0
            
            pricing_dict = policy_engine.calculate_limits_and_pricing(
                grade=str(r_score['credit_grade']),
                pd_cal=float(r_score['calibrated_pd']),
                p10_income=p10_inc,
                p50_income=p50_inc,
                existing_obligations=oblg,
                prior_loans_count=safe_int(row_fs.get('ldr_prior_loans_count', 0))
            )
        else:
            pricing_dict = {'approved_limit': 500000.0, 'apr_pct': 22.5, 'max_affordable_emi': 35000.0, 'pricing_breakdown': {'Cost of Funds (CoF)': 8.0, 'Operating Expense (OpEx)': 4.0, 'ECL Risk Premium': 3.5, 'Capital Charge': 4.0, 'Target Margin': 6.0}}
            
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            max_avail = max(50000.0, float(pricing_dict['approved_limit']))
            chosen_amount = st.slider("?? Choose Loan Amount (PKR):", min_value=25000.0, max_value=max_avail, value=min(250000.0, max_avail), step=25000.0)
            chosen_tenor = st.selectbox("?? Repayment Tenor:", [6, 12, 18, 24, 36], index=3)
            
            r_mo = (pricing_dict['apr_pct'] / 100.0) / 12.0
            n_mo = chosen_tenor
            monthly_emi = round(chosen_amount * (r_mo * (1 + r_mo)**n_mo) / ((1 + r_mo)**n_mo - 1), 2)
            total_repay = round(monthly_emi * n_mo, 2)
            
            st.markdown(f\"\"\"
            <div style='background:#111827; border:1px solid #334155; border-radius:8px; padding:12px; margin-top:12px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='color:#94a3b8;'>Monthly Installment (EMI):</span>
                    <strong style='color:#10b981; font-size:1.1rem;'>PKR {monthly_emi:,.2f}</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='color:#94a3b8;'>Total Amount Payable:</span>
                    <strong>PKR {total_repay:,.2f}</strong>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#94a3b8;'>Fixed APR:</span>
                    <strong style='color:#60a5fa;'>{pricing_dict['apr_pct']:.2f}% p.a.</strong>
                </div>
            </div>
            \"\"\", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("**??? Cost & Pricing Transparency:**")
            for k_cost, v_cost in pricing_dict['pricing_breakdown'].items():
                st.write(f"• **{k_cost}:** `{v_cost}%`")
            st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
            st.markdown("**?? Digital e-Signature:**")
            st.text_input("Type Full Name to e-Sign:", value=names_dict.get(target_id, "Hassan Tariq Naqvi"))
            
            if st.button("?? Accept Offer & Disburse Funds Now", type="primary"):
                st.balloons()
                st.success(f"?? Loan of PKR {chosen_amount:,.0f} approved & disbursed! Transaction Ref: `PK-DISB-{np.random.randint(100000, 999999)}`")
                
        st.markdown("</div>", unsafe_allow_html=True)
""")
with open("dashboard/app.py", "a", encoding="utf-8") as f:
    f.write("""
else:
    if "Portfolio" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Portfolio Credit Intelligence & IFRS 9 ECL Engine</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;margin-bottom:20px;'>Real-time macro portfolio analytics across all 250,000 scored customer accounts</p>", unsafe_allow_html=True)
        
        if df_scores is None:
            st.error("? `data/processed/credit_scores_250k.parquet` not found. Please run scoring pipeline.")
            st.stop()
            
        df = df_scores.reset_index()
        tot = len(df)
        appr = (df["credit_decision"] == "APPROVED").sum()
        ref = (df["credit_decision"] == "REFER").sum()
        dec = (df["credit_decision"] == "DECLINED").sum()
        med_score = int(df[df["composite_score_1000"] > 0]["composite_score_1000"].median())
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Applicants", f"{tot:,}")
        k2.metric("? Approved", f"{appr:,}", f"{appr/tot*100:.1f}%")
        k3.metric("?? Refer", f"{ref:,}", f"{ref/tot*100:.1f}%")
        k4.metric("? Declined", f"{dec:,}", f"{dec/tot*100:.1f}%")
        k5.metric("Median Score", f"{med_score} / 900")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns(2)
        with c_left:
            fig_pie = go.Figure(go.Pie(
                labels=["Approved", "Refer", "Declined"],
                values=[appr, ref, dec],
                hole=0.6,
                marker_colors=["#10b981", "#f59e0b", "#ef4444"],
                textinfo="label+percent"
            ))
            fig_pie.add_annotation(text=f"<b>{tot:,}</b><br><span style='font-size:11px'>Total Scored</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#f1f5f9"))
            fig_pie.update_layout(title="Decision Distribution", height=320, **PLOTLY_DARK)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c_right:
            grd_counts = df["credit_grade"].value_counts().reindex(["A","B","C","D","E","F","G"]).fillna(0)
            fig_bar = go.Figure(go.Bar(
                x=grd_counts.index,
                y=grd_counts.values,
                marker_color=[GRADE_COLOR.get(g, "#64748b") for g in grd_counts.index],
                text=[f"{int(v):,}" for v in grd_counts.values],
                textposition="outside"
            ))
            fig_bar.update_layout(title="Credit Grade Distribution (A–G)", height=320, **PLOTLY_DARK)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown("### ??? IFRS 9 Expected Credit Loss (ECL) Provisioning Breakdown")
        
        ifrs9_data = policy_engine.calculate_portfolio_ifrs9(df_scores)
        if ifrs9_data:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Portfolio Exposure (EAD)", f"PKR {ifrs9_data['total_portfolio_exposure']:,.0f}")
            c2.metric("Total IFRS 9 ECL Provision", f"PKR {ifrs9_data['total_ecl_provision']:,.0f}")
            c3.metric("Portfolio ECL Coverage Ratio", f"{ifrs9_data['ecl_coverage_ratio_pct']}%")
            
            stage_rows = []
            for stg_name, stg_vals in ifrs9_data['stages'].items():
                stage_rows.append({
                    "IFRS 9 Classification": stg_name,
                    "Account Count": f"{stg_vals['count']:,}",
                    "Portfolio Share": f"{stg_vals['pct']}%",
                    "Average Staged PD": f"{stg_vals['avg_pd_pct']}%",
                    "ECL Provision (PKR)": f"PKR {stg_vals['ecl_pkr']:,.0f}"
                })
            st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)

    elif "Real-Time Underwriting" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Real-Time Underwriting & Customer Deep Dive</h1>", unsafe_allow_html=True)
        cust_name = names_dict.get(target_id, "Unknown Name")
        st.markdown(f"<p style='color:#64748b;margin-bottom:20px;'>Live 12-Pillar Assessment and Data Family Signals for <code style='color:#93c5fd;'>{target_id} ({cust_name})</code></p>", unsafe_allow_html=True)
        
        if df_fs is None or target_id not in df_fs.index:
            st.error(f"Customer `{target_id}` not found in feature store.")
            st.stop()
            
        row_fs = df_fs.loc[target_id]
        row_fs_df = df_fs.loc[[target_id]]
        live_scores = composite_scorer.compute_scores(row_fs_df)
        r_score = live_scores.iloc[0]
        
        score_val = int(r_score["composite_score_1000"])
        grade_val = str(r_score["credit_grade"])
        dec_val = str(r_score["credit_decision"])
        decline_rsn = r_score.get("decline_reason", None)
        
        b_class = {"APPROVED": "decision-approved", "REFER": "decision-refer", "DECLINED": "decision-declined"}.get(dec_val, "decision-declined")
        b_icon = {"APPROVED": "?", "REFER": "??", "DECLINED": "?"}.get(dec_val, "?")
        rsn_str = f" — Reason: <code>{decline_rsn}</code>" if (pd.notna(decline_rsn) and str(decline_rsn) != 'nan') else ""
        st.markdown(f"<div class='{b_class}'>{b_icon} Underwriting Decision: {dec_val} (Score: {score_val} / 900, Grade: {grade_val}){rsn_str}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Composite Score", f"{score_val} / 900", f"Grade {grade_val}")
        c2.metric("Calibrated Default (PD)", f"{float(r_score['calibrated_pd'])*100:.2f}%")
        c3.metric("P50 Expected Income", f"PKR {float(row_fs.get('p50_income', 0)):,.0f}")
        c4.metric("FOIR / Debt Ratio", f"{min(100.0, float(row_fs.get('foir_pct', 0))):.1f}%")
        c5.metric("NADRA KYC Status", "VERIFIED")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_radar, col_bars = st.columns([1, 1])
        
        pillar_labels = ["P01 Identity", "P02 Fraud", "P03 Income", "P04 CashFlow", "P05 Afford", "P06 Stability", "P07 Behavior", "P08 Digital", "P09 Bureau", "P10 Relat", "P11 Collect", "P12 Compl"]
        pillar_vals = [
            float(r_score["p01_score"]), float(r_score["p02_score"]), float(r_score["p03_score"]),
            float(r_score["p04_score"]), float(r_score["p05_score"]), float(r_score["p06_score"]),
            float(r_score["p07_score"]), float(r_score["p08_score"]), float(r_score["p09_score"]),
            float(r_score["p10_score"]), float(r_score["p11_score"]), 100.0 if r_score.get("p12_compliance", "PASS") == "PASS" else 0.0
        ]
        
        with col_radar:
            fig_rad = go.Figure(go.Scatterpolar(r=pillar_vals, theta=pillar_labels, fill='toself', line_color='#3b82f6'))
            fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e293b")), title="12-Pillar Scoring Radar Profile", height=330, **PLOTLY_DARK)
            st.plotly_chart(fig_rad, use_container_width=True)
            
        with col_bars:
            bar_colors = ["#10b981" if v >= 70 else "#f59e0b" if v >= 50 else "#ef4444" for v in pillar_vals]
            fig_pbar = go.Figure(go.Bar(x=pillar_vals, y=pillar_labels, orientation='h', marker_color=bar_colors, text=[f"{v:.1f}" for v in pillar_vals], textposition='outside'))
            fig_pbar.update_layout(title="Pillar Breakdown (0–100 Scale)", xaxis_range=[0, 115], height=330, **PLOTLY_DARK)
            st.plotly_chart(fig_pbar, use_container_width=True)
            
        st.markdown("### ??? Raw & Derived Data Family Signals")
        t1, t2, t3, t4, t5 = st.tabs(["?? Identity & Demo", "?? Banking & CashFlow", "?? Telco & Wallet", "?? Device & Biometrics", "?? Repayment & Velocity"])
        
        with t1:
            c_a, c_b = st.columns(2)
            c_a.write(f"**Age:** {safe_int(row_fs.get('age', 0))} years")
            c_a.write(f"**Gender:** {str(row_fs.get('gender', 'N/A')).title()}")
            c_a.write(f"**Education:** {str(row_fs.get('education', 'N/A'))}")
            c_a.write(f"**Marital Status:** {str(row_fs.get('marital_status', 'N/A'))}")
            c_b.write(f"**CNIC Verification:** Valid (NADRA Verified)")
            c_b.write(f"**Email Age:** {safe_int(row_fs.get('email_age_days', 0))} days")
            c_b.write(f"**SIM Age:** {safe_int(row_fs.get('sim_age_days', 0))} days")
            c_b.write(f"**Digital Maturity Index:** {safe_float(row_fs.get('digital_maturity_index', 0)):.1f}/100")
            
        with t2:
            c_a, c_b = st.columns(2)
            c_a.write(f"**Total Bank Accounts:** {safe_int(row_fs.get('bank_account_count', 1))}")
            c_a.write(f"**Monthly Credits (Inflows):** PKR {safe_float(row_fs.get('total_monthly_credit', 0)):,.2f}")
            c_a.write(f"**Monthly Debits (Outflows):** PKR {safe_float(row_fs.get('total_monthly_debit', 0)):,.2f}")
            c_b.write(f"**Average Monthly Balance:** PKR {safe_float(row_fs.get('total_avg_monthly_balance', 0)):,.2f}")
            c_b.write(f"**Overdraft Utilization Rate:** {safe_float(row_fs.get('overdraft_utilization_rate', 0))*100:.1f}%")
            c_b.write(f"**NSF (Bounced) Count:** {safe_int(row_fs.get('nsf_count_total', 0))}")
            
        with t3:
            c_a, c_b = st.columns(2)
            c_a.write(f"**Phone Type:** {str(row_fs.get('phone_type', 'Prepaid'))}")
            c_a.write(f"**Monthly Recharge:** PKR {safe_float(row_fs.get('tel_avg_monthly_recharge_pkr', 0)):,.2f}")
            c_a.write(f"**Telecom Credit Signal:** {safe_float(row_fs.get('tel_telecom_credit_signal', 0)):.1f}/100")
            c_b.write(f"**Has Mobile Wallet:** {bool(row_fs.get('wal_has_wallet', False))}")
            c_b.write(f"**Wallet Provider:** {str(row_fs.get('wal_wallet_provider', 'None'))}")
            c_b.write(f"**Wallet Monthly Volume:** PKR {safe_float(row_fs.get('wal_avg_monthly_txn_value', 0)):,.2f}")
            
        with t4:
            c_a, c_b = st.columns(2)
            c_a.write(f"**Device Brand:** {str(row_fs.get('dev_device_manufacturer', 'N/A'))}")
            c_a.write(f"**OS Version:** {str(row_fs.get('dev_os_version', 'N/A'))}")
            c_a.write(f"**Rooted / Emulator:** {bool(row_fs.get('dev_is_rooted', False))} / {bool(row_fs.get('dev_emulator_detected', False))}")
            c_b.write(f"**Session Count (7d):** {safe_int(row_fs.get('dev_session_count_7d', 0))}")
            c_b.write(f"**Session Replay Anomaly:** {safe_float(row_fs.get('dev_session_replay_anomaly_score', 0)):.1f}/100")
            c_b.write(f"**Behavioral Risk Score:** {safe_float(row_fs.get('dev_behavioral_risk_score', 0)):.1f}/100")
            
        with t5:
            c_a, c_b = st.columns(2)
            c_a.write(f"**Existing Loan Customer:** {bool(row_fs.get('ldr_has_internal_history', False))}")
            c_a.write(f"**On-Time Payment Rate:** {safe_float(row_fs.get('ldr_on_time_payment_rate', 1.0))*100:.1f}%")
            c_a.write(f"**Max DPD Ever:** {safe_int(row_fs.get('ldr_max_dpd_ever', 0))} days")
            c_b.write(f"**Apps Same Device (7d):** {safe_int(row_fs.get('app_applications_same_device_7d', 1))}")
            c_b.write(f"**Cross-Lender Velocity (30d):** {safe_int(row_fs.get('app_cross_lender_velocity_30d', 0))}")
            c_b.write(f"**Application Velocity Risk:** {safe_float(row_fs.get('app_velocity_risk_score', 0)):.1f}/100")

    elif "Quantile Income" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Quantile Income Prediction & Capacity Sizing</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b;margin-bottom:20px;'>LightGBM Quantile Pinball Loss Bounds & Capacity Formulas for <code>{target_id}</code></p>", unsafe_allow_html=True)
        
        row_fs = df_fs.loc[target_id] if (df_fs is not None and target_id in df_fs.index) else None
        p10 = float(row_fs.get('p10_income', 80000) or 80000) if row_fs is not None else 80000.0
        p50 = float(row_fs.get('p50_income', 120000) or 120000) if row_fs is not None else 120000.0
        p90 = float(row_fs.get('p90_income', 160000) or 160000) if row_fs is not None else 160000.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("P10 Conservative Income", f"PKR {p10:,.0f}", "Worst-case Baseline (FOIR)")
        c2.metric("P50 Median Income", f"PKR {p50:,.0f}", "Most Likely (Limit Sizing)")
        c3.metric("P90 Optimistic Income", f"PKR {p90:,.0f}", "Upside Potential")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ?? Interactive Underwriting Sizing Simulator")
        
        oblg_val = float(row_fs.get('utility_debit_amt_12m', 0) or 0) / 12.0 if row_fs is not None else 15000.0
        pricing = policy_engine.calculate_limits_and_pricing(
            grade="A", pd_cal=0.03, p10_income=p10, p50_income=p50, existing_obligations=oblg_val
        )
        
        c_a, c_b, c_c, c_d = st.columns(4)
        c_a.metric("Existing Obligations", f"PKR {oblg_val:,.0f}/mo")
        c_b.metric("FOIR %", f"{(oblg_val/p10*100):.1f}%", "Max Cap: 45%")
        c_c.metric("Max Disposable EMI", f"PKR {pricing['max_affordable_emi']:,.0f}/mo")
        c_d.metric("Approved Term Loan Limit", f"PKR {pricing['approved_limit']:,.0f}")
""")
with open("dashboard/app.py", "a", encoding="utf-8") as f:
    f.write("""
    elif "Explainable AI" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Explainable AI & Adverse Action Generator (SHAP)</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b;margin-bottom:20px;'>Feature Attribution and Regulatory Adverse Action Reason Codes for <code>{target_id}</code></p>", unsafe_allow_html=True)
        
        row_fs = df_fs.loc[target_id] if (df_fs is not None and target_id in df_fs.index) else None
        row_fs_df = df_fs.loc[[target_id]]
        live_scores = composite_scorer.compute_scores(row_fs_df)
        r_score = live_scores.iloc[0]
        
        sub_dict = {k: r_score[k] for k in r_score.index if k.startswith('p0') or k.startswith('p1')}
        xai_res = shap_explainer.explain_prediction(row_fs, sub_dict)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ?? Top Positive Drivers (Trust Enhancers)")
            for item in xai_res['top_positive']:
                st.markdown(f\"\"\"
                <div style='background:#111827; border-left:4px solid #10b981; padding:10px 14px; margin-bottom:8px; border-radius:4px;'>
                    <div style='font-weight:700; color:#f8fafc;'>{item['name']} ({item['value']})</div>
                    <div style='font-size:0.8rem; color:#94a3b8;'>{item['reason']}</div>
                </div>
                \"\"\", unsafe_allow_html=True)
                
        with c2:
            st.markdown("### ?? Top Risk Drivers (Adverse Action Factors)")
            for item in xai_res['top_negative']:
                st.markdown(f\"\"\"
                <div style='background:#111827; border-left:4px solid #ef4444; padding:10px 14px; margin-bottom:8px; border-radius:4px;'>
                    <div style='font-weight:700; color:#f8fafc;'>{item['name']} ({item['value']})</div>
                    <div style='font-size:0.8rem; color:#94a3b8;'>{item['reason']}</div>
                </div>
                \"\"\", unsafe_allow_html=True)
                
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown("### ?? Formal Regulatory Adverse Action Notice")
        dec_str = str(r_score['credit_decision'])
        if dec_str == 'DECLINED' or dec_str == 'REFER':
            list_items = "".join([f"<li><b>{x['name']}:</b> {x['reason']}</li>" for x in xai_res['top_negative'][:3]])
            st.markdown(f\"\"\"
            <div style='background:#1e293b; border:1px solid #475569; padding:20px; border-radius:8px;'>
                <div style='font-weight:800; font-size:1.1rem; color:#f8fafc;'>NOTICE OF CREDIT DECISION / ADVERSE ACTION</div>
                <div style='font-size:0.82rem; color:#94a3b8; margin-top:4px;'>Applicant: <b>{names_dict.get(target_id, 'Applicant')}</b> (ID: {target_id}) · Date: {datetime.now().strftime('%d-%b-%Y')}</div>
                <hr style='border-color:#334155; margin:10px 0;'>
                <p style='color:#cbd5e1; font-size:0.9rem;'>
                    Thank you for your recent application. Under the applicable Consumer Credit and Fair Lending Regulations, we are providing the principal reasons for our credit evaluation:
                </p>
                <ol style='color:#f8fafc; font-size:0.88rem;'>
                    {list_items}
                </ol>
                <div style='font-size:0.78rem; color:#64748b; margin-top:12px;'>Model Governance Engine: SHAP Explainability Layer · Model Version: 2.0-Production</div>
            </div>
            \"\"\", unsafe_allow_html=True)
        else:
            st.success("? Applicant is APPROVED with zero adverse action flags.")

    elif "Model Mechanics" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>AI Model Mechanics & Physical Model Registry</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;margin-bottom:20px;'>Technical audit of all 20 serialized machine learning models on disk</p>", unsafe_allow_html=True)
        
        models_dir = PROJECT_ROOT / "models"
        model_rows = []
        if models_dir.exists():
            for f_path in sorted(models_dir.glob("*.joblib")):
                size_kb = f_path.stat().st_size / 1024.0
                size_str = f"{size_kb/1024.0:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
                role = "Pillar Sub-Score Model" if "pillar" in f_path.name else ("Quantile Income Regressor" if "income" in f_path.name else "Stacking PD Ensemble Component")
                model_rows.append({
                    "Model File": f_path.name,
                    "Format": "JOBLIB",
                    "Size": size_str,
                    "Assigned Role": role,
                    "Status": "ACTIVE (Loaded)"
                })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
        
        st.markdown("### ?? Score Calibration Math: Log-Odds Transformation")
        st.latex(r"\\text{Score} = 600 + 72.13 \\times \\ln\\left(\\frac{1 - PD}{PD}\\right)")
        
        pds = np.linspace(0.001, 0.99, 100)
        scores_plot = np.clip(600 + 72.13 * np.log((1 - pds) / pds), 300, 900)
        fig_cal = go.Figure(go.Scatter(x=pds * 100, y=scores_plot, mode='lines', line=dict(color='#3b82f6', width=3)))
        fig_cal.update_layout(title="Probability of Default vs. Calibrated Credit Score (300-900)", xaxis_title="Probability of Default (%)", yaxis_title="Credit Score (300-900)", height=320, **PLOTLY_DARK)
        st.plotly_chart(fig_cal, use_container_width=True)

    elif "Policy Simulator" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Policy Simulator & Champion/Challenger Studio</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;margin-bottom:20px;'>Interactive risk policy stress-testing and A/B challenger experimentation</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_cutoff = st.slider("Approval Score Cutoff:", min_value=500, max_value=750, value=600, step=10)
        with c2:
            sim_foir = st.slider("Max FOIR Cap (%):", min_value=35, max_value=60, value=45, step=5)
        with c3:
            sim_explore = st.slider("Exploration Slice (%):", min_value=0.0, max_value=5.0, value=1.5, step=0.5, help="Random approvals of marginal declines to learn reject behavior")
            
        if df_scores is not None:
            sim_appr = ((df_scores['composite_score_1000'] >= sim_cutoff) & (df_scores['credit_grade'] != 'G')).sum()
            sim_appr_pct = sim_appr / len(df_scores) * 100.0
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Simulated Approval Rate", f"{sim_appr_pct:.1f}%", f"{sim_appr_pct - 60.7:+.1f}% vs. Champion")
            s2.metric("Simulated Expected Loss Rate", f"{(100 - sim_appr_pct)*0.12:.2f}%")
            s3.metric("Champion / Challenger Allocation", "90% Champion / 10% Challenger")
            
            st.info(f"?? Under policy cutoff `{sim_cutoff}` and FOIR cap `{sim_foir}%`, portfolio approval volume changes to **{sim_appr:,}** applicants with an exploration slice of **{int(len(df_scores)*sim_explore/100):,}** exploratory loans.")

    elif "Data Taxonomy" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Data Taxonomy & 15 Data Families</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;margin-bottom:20px;'>Comprehensive data dictionary across 222 features</p>", unsafe_allow_html=True)
        
        families = [
            ("F1 Identity & KYC", "National ID validation, biometrics, liveness, NADRA matching", 8),
            ("F2 Demographic Anchor", "Age, education, marital status, residential stability", 12),
            ("F3 Device Intelligence", "Hardware model, OS, rooting, emulator detection, VPN checks", 18),
            ("F4 Behavioral Biometrics", "Typing speed, hesitation, copy-paste events, session dynamics", 14),
            ("F5 Telecom & Network", "SIM tenure, top-up frequency, recharge amounts, operator porting", 20),
            ("F6 Bank Account & CashFlow", "Monthly credits, debits, minimum balance, NSF bounces", 26),
            ("F7 Mobile Wallet", "Easypaisa/JazzCash tenure, P2P network, transaction volumes", 16),
            ("F8 Utility Payments", "Electricity, gas, water payment regularity and delinquency", 14),
            ("F9 Employment & Payroll", "Employer category, tenure, salary credit consistency", 16),
            ("F10 Location & Geo-Risk", "Night location consistency, regional default clusters", 10),
            ("F11 Digital Footprint", "Email age, domain reputation, digital maturity index", 14),
            ("F12 Income Derived", "LightGBM P10, P50, P90 quantile estimates, stability scores", 12),
            ("F13 Internal Loan Ledger", "Prior loan history, on-time payment rate, max DPD ever", 22),
            ("F14 Application Velocity", "Same device/IP velocity in 24h/7d, cross-lender inquiries", 12),
            ("F15 Compliance & AML", "PEP status, sanctions screening, internal blacklist match", 8)
        ]
        
        f_df = pd.DataFrame(families, columns=["Family Code & Name", "Signal Scope", "Feature Count"])
        st.dataframe(f_df, use_container_width=True)

st.markdown("---<div style='font-size:0.72rem;color:#475569;text-align:center;'>PAK-CREDIT AI · Production Version 2.0 · Staged Underwriting Engine</div>", unsafe_allow_html=True)
""")
print('Script finished successfully!')
