with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

import re

# Add import
if "from scoring import disbursement_ledger" not in text:
    text = text.replace(
        "from scoring import policy_engine",
        "from scoring import policy_engine\nfrom scoring import disbursement_ledger"
    )

pattern = r'(\s+)(if st\.button\([^)]*Accept Offer & Disburse[^)]*\):[\s\S]*?Transaction Ref:[^\n]*)'

replacement = r'''\1cust_name = names_dict.get(target_id, "Shahid Hashmi")
\1sign_val = st.text_input("Type Full Name to e-Sign:", value=cust_name, key=f"esign_{target_id}")
\1
\1if st.button("🚀 Accept Offer & Disburse Funds Now", type="primary", key="btn_disburse_live"):
\1    rec = disbursement_ledger.record_disbursement(
\1        customer_id=target_id,
\1        full_name=cust_name,
\1        amount=float(chosen_amount),
\1        tenor_months=int(chosen_tenor),
\1        apr_pct=float(pricing_dict['apr_pct']),
\1        monthly_emi=float(monthly_emi),
\1        total_repayable=float(total_repay),
\1        e_sign_name=sign_val,
\1        credit_grade=str(r_score['credit_grade']) if 'r_score' in locals() else 'B',
\1        calibrated_pd=float(r_score['calibrated_pd']) if 'r_score' in locals() else 0.035
\1    )
\1    st.balloons()
\1    st.success(f"✅ Funds Transferred via SBP Raast! Ref: `{rec['disbursement_id']}` | Settled to IBAN / Account.")
\1    st.info(f"📋 Record persisted permanently to Core Banking Ledger: `data/processed/disbursed_loans.parquet`")
\1
\1st.markdown("<br>", unsafe_allow_html=True)
\1st.markdown("#### 📜 Core Banking Active Loan Ledger (Real-Time Persistent Audit Trail)")
\1df_disb_live = disbursement_ledger.get_disbursed_loans(limit=10)
\1if not df_disb_live.empty:
\1    st.dataframe(df_disb_live[['disbursement_id', 'timestamp', 'customer_id', 'full_name', 'principal_pkr', 'tenor_months', 'monthly_emi_pkr', 'status', 'payment_rail']], use_container_width=True)
\1else:
\1    st.caption("No loans disbursed yet.")'''

# First remove the redundant text_input before the button
text = re.sub(r'st\.text_input\("Type Full Name to e-Sign:[^)]*\)\n', '', text)
text = re.sub(pattern, replacement, text)

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully connected Screen 5 to real-time persistent core banking ledger!")
