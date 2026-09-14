with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add import
if "from scoring import credit_memo_generator" not in text:
    text = text.replace(
        "from scoring import disbursement_ledger",
        "from scoring import disbursement_ledger\nfrom scoring import credit_memo_generator"
    )

# Search for the customer deep dive header where we can place the download memo button
target_str = 'st.markdown(f"<p style=\'color:#64748b;margin-bottom:20px;\'>Deep-dive single borrower assessment for <code>{target_id}</code></p>", unsafe_allow_html=True)'

replacement_str = target_str + """
        # Institutional Credit Assessment Memorandum (CAM) Export
        try:
            sub_dict_memo = {k: r_score[k] for k in r_score.index if k.startswith('p0') or k.startswith('p1')}
            xai_memo = shap_explainer.explain_prediction(row_fs, sub_dict_memo)
            cam_text = credit_memo_generator.generate_institutional_credit_memo(
                customer_id=target_id,
                name=cust_name_disp,
                res_dict=r_score.to_dict(),
                pricing_dict=pricing_dict,
                xai_dict=xai_memo
            )
            col_m1, col_m2 = st.columns([4, 1])
            with col_m2:
                st.download_button(
                    label="📄 Export Risk Memo (CAM)",
                    data=cam_text,
                    file_name=f"CAM_{target_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    help="Official Credit Assessment Memorandum for Risk Committee Approval"
                )
        except Exception:
            pass"""

if target_str in text and "Export Risk Memo (CAM)" not in text:
    text = text.replace(target_str, replacement_str)
    print("Added CAM export button to deep dive tab!")
else:
    print("CAM target string not found or already added")

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully updated dashboard/app.py with Institutional Credit Memo export!")
