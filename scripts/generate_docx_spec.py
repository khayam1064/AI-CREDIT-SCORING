import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
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

def create_docx():
    doc = docx.Document()
    
    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("Enterprise Credit AI Engine & Staged Decision Platform")
    tr.font.name = 'Arial'
    tr.font.size = Pt(22)
    tr.font.bold = True
    tr.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    title.paragraph_format.space_after = Pt(2)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Complete End-to-End Mathematical Architecture, IFRS 9 ECL, Multi-Model Stacking & SHAP Specification")
    sr.font.name = 'Arial'
    sr.font.size = Pt(12)
    sr.font.italic = True
    sr.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    sub.paragraph_format.space_after = Pt(20)

    # Section 1
    h1 = doc.add_heading(level=1)
    h1.add_run("1. Machine Learning Model Inventory & Objectives")
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)

    # Models Table
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Task / Domain", "Algorithm", "Loss / Objective", "Key Metric & Rationale"]
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1E293B")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        for p in hdr_cells[i].paragraphs:
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)

    model_rows = [
        ("PD (Main Credit Risk)", "LightGBM + XGBoost + Logistic Stacking", "Binary Log-Loss with Monotonic Constraints", "AUC-ROC, Gini, KS, Brier. SOTA tabular performance; fast; native SHAP."),
        ("Regulatory Benchmark", "Logistic Regression on WoE Features", "Log-Loss + L2 Ridge Regularization", "Gini, Hosmer-Lemeshow. Fully auditable monotonic coefficients for examiners."),
        ("Income Quantiles", "LightGBM Quantile Regressor (P10/P50/P90)", "Pinball (Asymmetric Quantile) Loss", "MAPE, Quantile Coverage. Uncertainty-aware income bounds, not point guesses."),
        ("Cash-Flow Forecast", "LSTM / Temporal CNN on Sequences", "MSE / Quantile Loss", "MAPE by horizon. Captures pay-cycle seasonality for EMI collection optimization."),
        ("Point Fraud", "LightGBM + Isolation Forest + Autoencoder", "Log-loss / Reconstruction Error", "Precision@k, Alert Rate. Catches known fraud signatures & novel anomalies."),
        ("Fraud Rings", "Graph Neural Network (GNN) / Louvain Graph", "Link-Prediction Loss", "Ring Recall, FP Rate. Detects relational multi-identity syndicate attacks."),
        ("Deepfake / Liveness", "Vendor Vision CNN + Challenge-Response", "ISO/IEC 30107-3 Standards", "APCER / BPCER. Specialized biometric verification."),
        ("Behavioral Telemetry", "GRU over In-App Telemetry Streams", "Log-Loss", "AUC Uplift. Captures keystroke cadence and form hesitation signals."),
        ("Survival / Lifetime PD", "Cox Proportional Hazards / Gradient Survival", "Partial Likelihood", "C-Index. Models default timing for IFRS 9 multi-year staging."),
        ("Collections Roll-Rate", "Markov Chain (Transitions) + GBM", "Transition Likelihood", "Roll-rate accuracy. Standard delinquency bucket migration mechanics."),
        ("Limit Strategy (Mature)", "Contextual Bandits / Reinforcement Learning", "Profit Reward with Risk & Fairness", "Profit per limit unit, Loss rate. Learns optimal credit-line growth ladder."),
        ("Segmentation", "K-Means / HDBSCAN on Embeddings", "Inertia / Density (Silhouette)", "Silhouette Score. Risk-tier communication & challenger underwriting."),
        ("Document Intelligence", "OCR + LayoutLMv3 Transformer", "Sequence-to-Sequence Cross-Entropy", "Field F1. Automated parsing of bank statements and salary slips.")
    ]

    for item in model_rows:
        row_cells = table.add_row().cells
        for col_idx, text in enumerate(item):
            row_cells[col_idx].text = text
            set_cell_margins(row_cells[col_idx], top=70, bottom=70, left=100, right=100)
            set_cell_background(row_cells[col_idx], "F8FAFC" if col_idx % 2 == 0 else "FFFFFF")
            for p in row_cells[col_idx].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 2: Mathematical Formulations
    h2 = doc.add_heading(level=1)
    h2.add_run("2. End-to-End Mathematical Formulations")
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    p_math1 = doc.add_paragraph()
    p_math1.add_run("2.1 Base Learners and Stacking:\n").font.bold = True
    p_math1.add_run("F_M(x) = sum(v * h_m(x)) where v is learning rate and h_m(x) is a regression tree fit to negative pseudo-gradients. Monotonic constraints are imposed (PD non-increasing in income, non-decreasing in FOIR and DPD). The stacking layer combines OOF predictions: z(x) = [PD_lgbm(x), PD_xgb(x), PD_logit(x), sub-scores...], and PD_ens(x) = sigma(w * z(x) + b).\n\n")

    p_math1.add_run("2.2 Probability Calibration:\n").font.bold = True
    p_math1.add_run("PD_cal = IsotonicFit(PD_ens -> observed default frequency by score decile). Evaluated via Brier Score = (1/N) * sum((PD_i - y_i)^2) and Hosmer-Lemeshow goodness-of-fit test.\n\n")

    p_math1.add_run("2.3 Points to Double the Odds (PDO) Score Formulation:\n").font.bold = True
    p_math1.add_run("Odds = (1 - PD_cal) / PD_cal\nScore = Offset + Factor * ln(Odds)\nFactor = PDO / ln(2) = 40 / ln(2) = 57.7078\nOffset = BaseScore - Factor * ln(BaseOdds) = 660 - 57.7078 * ln(15) = 503.727\nScore = clip(503.73 + 57.71 * ln(Odds), 300, 900)\n\n")

    p_math1.add_run("2.4 Expected Credit Loss (ECL) & IFRS 9:\n").font.bold = True
    p_math1.add_run("ECL = PD * LGD * EAD, where LGD = 1 - Recovery Rate (65-85% for unsecured nano-loans) and EAD = Outstanding Balance + CCF * Undrawn Limit. Lifetime ECL (Stage 2/3) = sum_t (PD_t * LGD_t * EAD_t) / (1 + EIR)^t.\n\n")

    p_math1.add_run("2.5 Risk-Based Pricing (APR):\n").font.bold = True
    p_math1.add_run("APR = CoF + OpEx + (PD * LGD) + (K * Hurdle) + Margin, subject to APR <= Regulatory Cap (36%) and EMI <= Affordability Ceiling.\n\n")

    p_math1.add_run("2.6 Multi-Bound Limit Assignment Engine:\n").font.bold = True
    p_math1.add_run("Limit = min(AffordabilityLimit, RiskLimit, PolicyCap, ProgressionCap)\nAffordabilityLimit = MaxEMI * tenor_adj\nRiskLimit = k(grade) * P50_Monthly_Income * 12 * 0.50 (k: A=1.5, B=1.2, C=0.8, D=0.5, E=0.25)\nProgressionCap = BaseFirstLoan * 1.5^min(Clean_Repaid_Loans, 6)\nPolicyCap = PKR 2,000,000.\n")

    # Section 3: Decision Matrix
    h3 = doc.add_heading(level=1)
    h3.add_run("3. Enterprise Decision Matrix & Grade Policy")
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    table_dm = doc.add_table(rows=1, cols=6)
    table_dm.alignment = WD_TABLE_ALIGNMENT.CENTER
    dm_headers = ["Score Band", "Grade", "12M PD Band", "Decision", "First-Loan Limit", "Pricing Tier"]
    dm_cells = table_dm.rows[0].cells
    for i, h in enumerate(dm_cells):
        dm_cells[i].text = dm_headers[i]
        set_cell_background(dm_cells[i], "0F172A")
        set_cell_margins(dm_cells[i], top=90, bottom=90, left=100, right=100)
        for p in dm_cells[i].paragraphs:
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(8.5)

    dm_rows = [
        ("780 – 900", "A", "< 1.5%", "Auto-Approve", "Up to 100% of Cap", "Prime Rate (14–18% APR)"),
        ("720 – 779", "B", "1.5 – 3.0%", "Auto-Approve", "70% of Cap", "Standard Rate (18–22% APR)"),
        ("660 – 719", "C", "3.0 – 6.0%", "Auto-Approve", "45% of Cap", "Standard Rate (22–26% APR)"),
        ("600 – 659", "D", "6.0 – 10.0%", "Approve (Reduced)", "25% of Cap", "Standard Plus (26–30% APR)"),
        ("560 – 599", "E", "10.0 – 15.0%", "Approve Small / Review", "10% of Cap (Starter Loan)", "Highest Tier (30–34% APR)"),
        ("520 – 559", "F", "15.0 – 22.0%", "Manual Review / Decline", "Starter Only with Mitigants", "Special Tier (34–36% APR)"),
        ("< 520 / Stop", "G", "> 22.0%", "Decline with Reasons", "No Facility (PKR 0)", "N/A (Decline)")
    ]

    for item in dm_rows:
        row_cells = table_dm.add_row().cells
        for col_idx, text in enumerate(item):
            row_cells[col_idx].text = text
            set_cell_margins(row_cells[col_idx], top=70, bottom=70, left=90, right=90)
            set_cell_background(row_cells[col_idx], "F8FAFC" if col_idx % 2 == 0 else "FFFFFF")
            for p in row_cells[col_idx].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Save
    out_file = "Enterprise_Credit_AI_Engine_Production_Specification.docx"
    doc.save(out_file)
    print(f"Successfully generated {out_file}")

create_docx()
