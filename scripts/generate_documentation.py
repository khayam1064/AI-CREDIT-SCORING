import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import os

def set_cell_background(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup - Margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Style definitions
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x41, 0x55) # Slate 700

    # Header / Title Block
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Enterprise Credit Scoring & Staged Underwriting Engine")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B) # Slate 800
    title.paragraph_format.space_after = Pt(4)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run("System Architecture, ML Models, and Dashboard Specification")
    sub_run.font.name = 'Arial'
    sub_run.font.size = Pt(14)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B) # Slate 500
    subtitle.paragraph_format.space_after = Pt(24)

    # Add a horizontal separator line
    p_sep = doc.add_paragraph()
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sep_run = p_sep.add_run("__________________________________________________________________")
    p_sep_run.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1) # Slate 300
    p_sep.paragraph_format.space_after = Pt(24)

    # --- Section 1 ---
    h1 = doc.add_heading(level=1)
    h1_run = h1.add_run("1. Executive Summary & Scoring Topology")
    h1_run.font.name = 'Arial'
    h1_run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph()
    p1.add_run(
        "This document describes the design and implementation of a state-of-the-art enterprise-grade Credit Scoring "
        "and Staged Underwriting Engine. The engine is engineered to process massive portfolios of credit applicants "
        "(up to 250,000+ customers) using a multi-staged risk evaluation topology. The model uses 15 distinct alternative "
        "and traditional data families (including device telemetry, biometrics, bank/wallet ledgers, telecom signals, and "
        "repayment behavior) to evaluate creditworthiness."
    )
    p1.paragraph_format.space_after = Pt(12)

    p2 = doc.add_paragraph()
    p2.add_run(
        "The underwriting system enforces a strict 4-stage credit decision topology to screen, assess, score, and scale "
        "each loan application:"
    )
    
    # 4 stages list
    stages = [
        ("Stage 0: Hard Stop Refusal Gates", "Filters applicants based on core risk signals: verified identity/trust, emulator presence, application velocity (>3 in 7 days), Debt-Service Ratio / FOIR exceeds 45%, and non-positive predicted income (P10 <= 0). Non-compliance results in an immediate Decline (Grade G, Score 300)."),
        ("Stage 1: Twelve Sub-Score Pillars", "Calculates 12 sub-scores (0-100 scale) for each applicant. Each pillar has its own machine learning or deterministic rules-based model using specialized feature sets from the database."),
        ("Stage 2: PD Stacking Ensemble", "Predicts a calibrated Probability of Default (PD) by combining the 11 numeric sub-scores and 6 raw credit variables through an ensembled model structure (LightGBM, XGBoost, and Logistic Regression) unified under a Logistic Stacking Meta-Learner."),
        ("Stage 3: Log-Odds Score Calibration & Grading", "Transforms the final stacked PD to an industry-standard credit score on a 300–900 scale and assigns a Risk Grade (A to G) and final decision (Approved, Refer, or Declined) using a risk-adjusted framework.")
    ]
    for stg_title, stg_desc in stages:
        p_stg = doc.add_paragraph(style='List Bullet')
        p_stg.paragraph_format.space_after = Pt(4)
        run_title = p_stg.add_run(stg_title + ": ")
        run_title.font.bold = True
        run_title.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        p_stg.add_run(stg_desc)

    # --- Section 2 ---
    h2 = doc.add_heading(level=1)
    h2_run = h2.add_run("2. The Twelve Sub-Score Pillars")
    h2_run.font.name = 'Arial'
    h2_run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)

    p3 = doc.add_paragraph()
    p3.add_run(
        "The Stage 1 sub-score assessment evaluates the applicant across twelve dimensions, each calibrated as a 0-100 "
        "score (where 100 represents the lowest risk and 0 represents the highest risk). Below is the detailed breakdown:"
    )
    p3.paragraph_format.space_after = Pt(12)

    # Table for 12 pillars
    table_pillars = doc.add_table(rows=1, cols=5)
    table_pillars.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Pillar / Sub-score", "Question Answered", "Primary Inputs", "Model Type", "Weight"]
    hdr_cells = table_pillars.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1E293B")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)

    pillars_data = [
        ("P01: Identity / Trust", "Is this a real, verified person?", "CNIC status, email age, SIM age", "Heuristic + 2% Variance", "8%"),
        ("P02: Fraud Risk", "Is this application deceptive?", "Device, behavioural biometrics, IP velocity", "LightGBM Classifier", "7%"),
        ("P03: Income", "How much do they earn, really?", "Bank ledger credit/debit signals, wallet signals", "LightGBM Regressor (Quantile)", "12%"),
        ("P04: Cash Flow", "Is money in > money out reliably?", "Balance trends, monthly credit, debit volumes", "LightGBM Classifier", "15%"),
        ("P05: Affordability", "Can they carry the proposed EMI?", "FOIR, DSR, disposable income buffers", "LightGBM Classifier", "15%"),
        ("P06: Stability", "How anchored is their life?", "SIM tenure, age, marital/employment status", "LightGBM Classifier", "8%"),
        ("P07: Behavioral", "Do they act like a good payer?", "App telemetry, session replays, copy-paste", "LightGBM Classifier", "6%"),
        ("P08: Digital Footprint", "How mature is their online presence?", "Email domain, browser type, device specs", "LightGBM Classifier", "5%"),
        ("P09: Bureau Proxy", "What is their formal credit history?", "Overdraft usage, utility bill payment counts", "LightGBM Classifier", "10%"),
        ("P10: Relationship", "Are they a loyal internal customer?", "Prior loans, active loans, repayment record", "LightGBM Classifier", "6%"),
        ("P11: Collection Risk", "How hard will it be to recover?", "DPD history, age, location risk category", "LightGBM Classifier", "8%"),
        ("P12: Compliance", "Do they pass regulatory gates?", "blacklist checks, PEP status, AML matches", "Deterministic Gate", "Pass/Fail")
    ]

    for item in pillars_data:
        row_cells = table_pillars.add_row().cells
        for col_idx, text in enumerate(item):
            row_cells[col_idx].text = text
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=150, right=150)
            set_cell_background(row_cells[col_idx], "F8FAFC" if col_idx % 2 == 0 else "FFFFFF")
            for paragraph in row_cells[col_idx].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- Section 3 ---
    h3 = doc.add_heading(level=1)
    h3_run = h3.add_run("3. Machine Learning Models & Stacking Architecture")
    h3_run.font.name = 'Arial'
    h3_run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    h3.paragraph_format.space_before = Pt(18)
    h3.paragraph_format.space_after = Pt(6)

    # Subsection: Income Model
    h3_1 = doc.add_heading(level=2)
    h3_1_run = h3_1.add_run("3.1 Quantile LightGBM Income Model")
    h3_1_run.font.name = 'Arial'
    h3_1_run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    h3_1.paragraph_format.space_before = Pt(12)
    h3_1.paragraph_format.space_after = Pt(4)

    p_inc = doc.add_paragraph()
    p_inc.add_run(
        "Instead of predicting a single mean income (which can misrepresent risk for low-earners), the system trains "
        "three separate LightGBM Regressors with pinball loss functions to predict three distinct quantiles: "
    )
    p_inc_q10 = doc.add_paragraph(style='List Bullet')
    p_inc_q10.add_run("P10 (10th Percentile / conservative): ").font.bold = True
    p_inc_q10.add_run("Used to compute the Debt Service Ratio (DSR) and FOIR. Serves as a worst-case baseline to prevent over-lending.")
    p_inc_q50 = doc.add_paragraph(style='List Bullet')
    p_inc_q50.add_run("P50 (50th Percentile / median): ").font.bold = True
    p_inc_q50.add_run("Represents the most probable expected income. Used to size credit limits.")
    p_inc_q90 = doc.add_paragraph(style='List Bullet')
    p_inc_q90.add_run("P90 (90th Percentile / optimistic): ").font.bold = True
    p_inc_q90.add_run("Provides insight into the applicant's maximum upside potential.")

    # Subsection: Stacking PD Model
    h3_2 = doc.add_heading(level=2)
    h3_2_run = h3_2.add_run("3.2 Stage 2 Probability of Default (PD) Stacking")
    h3_2_run.font.name = 'Arial'
    h3_2_run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    h3_2.paragraph_format.space_before = Pt(12)
    h3_2.paragraph_format.space_after = Pt(4)

    p_pd = doc.add_paragraph()
    p_pd.add_run(
        "To predict the default probability, we built a multi-model ensembling system that combines base learners "
        "using a meta-classifier. The model takes 17 inputs (the 11 numeric sub-scores + 6 raw numeric risk variables: "
        "age, savings_ratio, overdraft_utilization_rate, ldr_avg_dpd_last_12m, ldr_on_time_payment_rate, nsf_count_total)."
    )
    p_pd.paragraph_format.space_after = Pt(6)

    p_pd_layers = doc.add_paragraph()
    p_pd_layers.add_run("Base Learners: ").font.bold = True
    p_pd_layers.add_run("We train three independent classifiers on the input dataset: (1) LightGBM Classifier, (2) XGBoost Classifier, and (3) a Logistic Regression model (pre-scaled using a StandardScaler). This ensures representation from gradient boosted trees and linear classifiers.")
    
    p_meta = doc.add_paragraph()
    p_meta.add_run("Logistic Stacking Meta-Learner: ").font.bold = True
    p_meta.add_run("The probability outputs from the base learners are concatenated and fed into a Logistic Regression meta-learner (with L2 regularization). The meta-learner assigns optimal weights to each base learner to output the final calibrated Probability of Default (PD), achieving an ROC-AUC of ~0.796.")

    # --- Section 4 ---
    h4 = doc.add_heading(level=1)
    h4_run = h4.add_run("4. Calibration & Risk-Grading")
    h4_run.font.name = 'Arial'
    h4_run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    h4.paragraph_format.space_before = Pt(18)
    h4.paragraph_format.space_after = Pt(6)

    p_cal = doc.add_paragraph()
    p_cal.add_run(
        "To make raw default probabilities intuitive for credit underwriters, the PD output from the stacking meta-learner "
        "is calibrated to a credit score range of 300 to 900 using a log-odds transformation:"
    )
    p_cal.paragraph_format.space_after = Pt(8)

    # Formula
    p_formula = doc.add_paragraph()
    p_formula.alignment = WD_ALIGN_PARAGRAPH.CENTER
    formula_run = p_formula.add_run("Score = 600 + 72.13 * ln((1 - PD) / PD)")
    formula_run.font.name = 'Courier New'
    formula_run.font.bold = True
    formula_run.font.size = Pt(13)
    p_formula.paragraph_format.space_after = Pt(12)

    p_grade = doc.add_paragraph()
    p_grade.add_run("Scores are binned into 7 risk grades with corresponding credit decisions:")
    
    # Grades table
    table_grades = doc.add_table(rows=1, cols=4)
    table_grades.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers_grd = ["Grade", "Score Range", "Credit Decision", "Description"]
    hdr_cells_grd = table_grades.rows[0].cells
    for i, h in enumerate(headers_grd):
        hdr_cells_grd[i].text = h
        set_cell_background(hdr_cells_grd[i], "0F172A")
        set_cell_margins(hdr_cells_grd[i], top=100, bottom=100, left=120, right=120)
        for paragraph in hdr_cells_grd[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)

    grades_data = [
        ("A", "800 - 900", "APPROVED", "Premium profile with near-zero default risk."),
        ("B", "700 - 799", "APPROVED", "Good profile with low, stable credit risk."),
        ("C", "600 - 699", "REFER", "Moderate risk profile. Referred for manual review."),
        ("D", "500 - 599", "REFER", "Elevated risk profile. Requires senior underwriting sign-off."),
        ("E", "400 - 499", "DECLINED", "High credit risk. Auto-declined."),
        ("F", "300 - 399", "DECLINED", "Severe credit risk. Auto-declined."),
        ("G", "300 (Fixed)", "DECLINED", "Failed Stage 0 Hard Stop / Regulatory Gate.")
    ]

    for item in grades_data:
        row_cells = table_grades.add_row().cells
        for col_idx, text in enumerate(item):
            row_cells[col_idx].text = text
            set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
            set_cell_background(row_cells[col_idx], "F8FAFC" if col_idx % 2 == 0 else "FFFFFF")
            for paragraph in row_cells[col_idx].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- Section 5 ---
    h5 = doc.add_heading(level=1)
    h5_run = h5.add_run("5. Interactive Dashboard Architecture")
    h5_run.font.name = 'Arial'
    h5_run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    h5.paragraph_format.space_before = Pt(18)
    h5.paragraph_format.space_after = Pt(6)

    p_db = doc.add_paragraph()
    p_db.add_run(
        "A premium Streamlit dashboard is implemented in dashboard/app.py. It has been modified to run the "
        "entire staged scoring pipeline live on the fly whenever a customer is selected. The application consists of:"
    )
    
    db_features = [
        ("Unified Search Bar", "Allows instant lookups of any of the 250,000 customers by typing their ID (e.g. CUST0000022) or by searching for their full name."),
        ("Portfolio Overview", "Displays overall KPI cards, Pie charts for decision distribution, and Bar charts for risk grade distribution across the entire portfolio."),
        ("Customer Deep Dive", "Shows a radar profile of the 12 sub-scores, a horizontal bar chart of pillar scores, and a tabbed grid containing the raw & derived data family signals."),
        ("Income & Underwriting", "Sizes credit limits (Term Loan, Credit Card) based on predicted quantile incomes and enforces the maximum 45% FOIR cap."),
        ("Model Mechanics", "Features a physical model registry that tracks and describes all trained machine learning assets on disk.")
    ]

    for feat_title, feat_desc in db_features:
        p_feat = doc.add_paragraph(style='List Bullet')
        p_feat.paragraph_format.space_after = Pt(4)
        run_title = p_feat.add_run(feat_title + ": ")
        run_title.font.bold = True
        run_title.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        p_feat.add_run(feat_desc)

    # Save document
    filename = "staged_credit_scoring_documentation.docx"
    doc.save(filename)
    print(f"Documentation saved successfully to: {os.path.abspath(filename)}")

if __name__ == '__main__':
    create_document()
