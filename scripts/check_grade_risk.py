with open("scoring/composite_scorer.py", "r", encoding="utf-8") as f:
    text = f.read()

old_grade_func = """def _grade_risk(score: int) -> tuple:
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
        return "G", "DECLINED\"\"\""""

new_grade_func = """def _grade_risk(score: int) -> tuple:
    \"\"\"
    Stage 3: Maps 300-900 score to Grades A-G as specified in Table 2:
    - 780 - 900: Grade A | Auto-Approve
    - 720 - 779: Grade B | Auto-Approve
    - 660 - 719: Grade C | Auto-Approve
    - 600 - 659: Grade D | Approve (Reduced)
    - 560 - 599: Grade E | Approve Small / Review (REFER)
    - 520 - 559: Grade F | Manual Review / Decline (REFER)
    - < 520:     Grade G | Decline with Reasons (DECLINED)
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
        return "G", "DECLINED\"\"\""""

# Let's check if old_grade_func matches
with open("scoring/composite_scorer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines[:60]):
    if "def _grade_risk" in l:
        print("Found line:", i+1)
        for j in range(i, i+16):
            print(f"{j+1}: {repr(lines[j])}")
        break
