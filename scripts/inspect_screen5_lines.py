with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Add import if not present
if "from scoring import disbursement_ledger" not in text:
    text = text.replace(
        "from scoring import policy_engine",
        "from scoring import policy_engine\nfrom scoring import disbursement_ledger"
    )

# Replace Screen 5 button logic with persistent ledger writing and live status table
old_screen5_btn = """            if st.button("dYZ% Accept Offer & Disburse Funds Now", type="primary"):
                st.balloons()
                st.success(f"dYZ% Loan of PKR {chosen_amount:,.0f} approved & disbursed! Transaction Ref: `PK-DISB-{np.random.randint(100000, 999999)}`")"""

# Let's check the exact string in app.py around lines 420-430
with open("dashboard/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "Accept Offer & Disburse" in l:
        print(f"Line {i+1}: {repr(l)}")
        print(f"Next line: {repr(lines[i+1])}")
        print(f"Next 2 line: {repr(lines[i+2])}")
