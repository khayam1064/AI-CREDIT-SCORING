import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring import composite_scorer

test_scores = [
    (850, 'A', 'APPROVED'),
    (740, 'B', 'APPROVED'),
    (680, 'C', 'APPROVED'),
    (620, 'D', 'APPROVED'),
    (580, 'E', 'REFER'),
    (540, 'F', 'REFER'),
    (490, 'G', 'DECLINED')
]

print("Testing exact Table 2 Grade boundaries:")
all_passed = True
for sc, exp_g, exp_d in test_scores:
    g, d = composite_scorer._grade_risk(sc)
    status = "PASS" if (g == exp_g and d == exp_d) else "FAIL"
    print(f"Score {sc:3d} -> Grade {g} ({d}) | Expected: {exp_g} ({exp_d}) | {status}")
    if status == "FAIL": all_passed = False

print("\nOverall Validation:", "ALL PASSED 100%" if all_passed else "FAILURES DETECTED")
