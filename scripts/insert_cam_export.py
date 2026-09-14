with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add import
if "from scoring import credit_memo_generator" not in text:
    text = text.replace(
        "from scoring import disbursement_ledger",
        "from scoring import disbursement_ledger\nfrom scoring import credit_memo_generator"
    )

target_pattern = 'st.markdown(f"<p style=\'color:#64748b;margin-bottom:20px;\'>Live 12-Pillar Assessment and Data Family Signals for <code style=\'color:#93c5fd;\'>{target_id} ({cust_name})</code></p>", unsafe_allow_html=True)'

export_snippet = """
        # Institutional Credit Assessment Memorandum (CAM) Export
        try:
            p10_memo = float(r_score.get('p10_income', 50000))
            p50_memo = float(r_score.get('p50_income', 65000))
            oblg_memo = float(row_fs.get('utility_debit_amt_12m', 0) or 0) / 12.0
            pricing_memo = policy_engine.calculate_limits_and_pricing(
                grade=grade_val, pd_cal=float(r_score['calibrated_pd']),
                p10_income=p10_memo, p50_income=p50_memo, existing_obligations=oblg_memo
            )
            sub_dict_memo = {k: r_score[k] for k in r_score.index if k.startswith('p0') or k.startswith('p1')}
            xai_memo = shap_explainer.explain_prediction(row_fs, sub_dict_memo)
            cam_text = credit_memo_generator.generate_institutional_credit_memo(
                customer_id=target_id,
                name=cust_name,
                res_dict=r_score.to_dict(),
                pricing_dict=pricing_memo,
                xai_dict=xai_memo
            )
            c_head1, c_head2 = st.columns([4, 1])
            with c_head2:
                st.download_button(
                    label="📄 Export Risk Memo (CAM)",
                    data=cam_text,
                    file_name=f"CAM_{target_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    help="Official Credit Assessment Memorandum for Risk Committee Approval"
                )
        except Exception:
            pass"""

if target_pattern in text and "Export Risk Memo (CAM)" not in text:
    text = text.replace(target_pattern, target_pattern + export_snippet)
    print("Export Risk Memo button successfully added to Real-Time Underwriting tab!")

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("dashboard/app.py updated cleanly!")
