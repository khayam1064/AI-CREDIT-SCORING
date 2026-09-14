with open('scoring/composite_scorer.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace grade bands
code = code.replace("if score >= 800:", "if score >= 780:")
code = code.replace("elif score >= 700:", "elif score >= 720:")
code = code.replace("elif score >= 500:", "elif score >= 560:")
code = code.replace("elif score >= 400:", "elif score >= 520:")
code = code.replace('return "F", "DECLINED"', 'return "G", "DECLINED"')

# Replace PDO formulation
old_pdo = "credit_score = 600.0 + 72.13 * np.log(odds)"
new_pdo = "# PDO Formulation: BaseScore=660 at BaseOdds=15:1, PDO=40\n    # Factor = 40 / ln(2) = 57.7078, Offset = 660 - 57.7078 * ln(15) = 503.727\n    credit_score = 503.727 + 57.7078 * np.log(odds)"
code = code.replace(old_pdo, new_pdo)

with open('scoring/composite_scorer.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Successfully updated composite_scorer.py with exact PDO & Decision Matrix bands!")
