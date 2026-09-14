with open("scoring/composite_scorer.py", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """def _grade_risk(score: int) -> tuple:
    \"\"\"Stage 3: Maps 300-900 score to Grades A-G.\"\"\"
    if score >= 780:
        return "A", "APPROVED"
    elif score >= 720:
        return "B", "APPROVED"
    elif score >= 600:
        return "C", "APPROVED"
    elif score >= 560:
        return "D", "REFER"
    elif score >= 520:
        return "E", "REFER"
    else:
        return "G", "DECLINED\""""

new_block = """def _grade_risk(score: int) -> tuple:
    \"\"\"
    Stage 3: Maps 300-900 score to Grades A-G strictly according to Table 2:
    - 780 - 900: Grade A | 12M PD < 1.5%   | Auto-Approve
    - 720 - 779: Grade B | 12M PD 1.5-3.0% | Auto-Approve
    - 660 - 719: Grade C | 12M PD 3.0-6.0% | Auto-Approve
    - 600 - 659: Grade D | 12M PD 6.0-10.0%| Approve (Reduced)
    - 560 - 599: Grade E | 12M PD 10.0-15% | Approve Small / Review (REFER)
    - 520 - 559: Grade F | 12M PD 15.0-22% | Manual Review / Decline (REFER)
    - < 520:     Grade G | 12M PD > 22.0%  | Decline with Reasons (DECLINED)
    \"\"\"
    if score >= 780:
        return "A", "APPROVED"
    elif score >= 720:
        return "B", "APPROVED"
    elif score >= 660:
        return "C", "APPROVED"
    elif score >= 600:
        return "D", "APPROVED"
    elif score >= 560:
        return "E", "REFER"
    elif score >= 520:
        return "F", "REFER"
    else:
        return "G", "DECLINED\""""

text = text.replace(old_block, new_block)

with open("scoring/composite_scorer.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated _grade_risk in scoring/composite_scorer.py to exact Table 2 specification!")
