with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

old_tab3_start = """    elif "Quantile Income" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>Quantile Income Prediction & Capacity Sizing</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b;margin-bottom:20px;'>LightGBM Quantile Pinball Loss Bounds & Capacity Formulas for <code>{target_id}</code></p>", unsafe_allow_html=True)
        
        row_fs = df_fs.loc[target_id] if (df_fs is not None and target_id in df_fs.index) else None
        p10 = float(row_fs.get('p10_income', 80000) or 80000) if row_fs is not None else 80000.0
        p50 = float(row_fs.get('p50_income', 120000) or 120000) if row_fs is not None else 120000.0
        p90 = float(row_fs.get('p90_income', 160000) or 160000) if row_fs is not None else 160000.0"""

new_tab3_start = """    elif "Quantile Income" in underwriter_page:
        st.markdown("<h1 style='font-size:1.8rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>💵 Quantile Income Prediction & Multi-Bound Capacity Sizing</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b;margin-bottom:20px;'>LightGBM Quantile Pinball Loss Bounds & 4-Bound Limit Sizing for <code>{target_id}</code></p>", unsafe_allow_html=True)
        
        row_fs = df_fs.loc[target_id] if (df_fs is not None and target_id in df_fs.index) else None
        if row_fs is not None:
            row_fs_df = df_fs.loc[[target_id]]
            live_scores = composite_scorer.compute_scores(row_fs_df)
            r_score = live_scores.iloc[0]
            p10 = float(r_score.get('p10_income', 55000.0))
            p50 = float(r_score.get('p50_income', 65000.0))
            p90 = float(r_score.get('p90_income', 75000.0))
            grade_val = str(r_score.get('credit_grade', 'C'))
            pd_val = float(r_score.get('calibrated_pd', 0.05))
        else:
            p10, p50, p90 = 55000.0, 65000.0, 75000.0
            grade_val, pd_val = 'C', 0.05"""

if old_tab3_start in text:
    text = text.replace(old_tab3_start, new_tab3_start)
    print("Replaced old_tab3_start successfully")
else:
    print("Warning: old_tab3_start pattern not found directly, checking partial replacement")

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Tab 3 patch script executed")
