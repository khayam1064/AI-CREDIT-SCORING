import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scoring import composite_scorer, policy_engine

test_cases = [
    ('Grade A', 0.008),
    ('Grade B', 0.022),
    ('Grade C', 0.045),
    ('Grade D', 0.080),
    ('Grade E', 0.125),
    ('Grade F', 0.180),
    ('Grade G', 0.250)
]

print("Target    | PD    | Odds   | Calibrated Score | Grade | Grade in docx")
print("-" * 65)
for target, pd_val in test_cases:
    odds = (1.0 - pd_val) / pd_val
    score = int(round(min(900, max(300, 503.727 + 57.7078 * np.log(odds)))))
    grade = composite_scorer._grade_risk(score)
    print(f"{target:9s} | {pd_val*100:4.1f}% | {odds:6.1f} | {score:16d} | {grade:5s} | {target.split()[-1]}")
