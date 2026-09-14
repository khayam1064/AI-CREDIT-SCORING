with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
dY_matches = re.findall(r'dY[a-zA-Z0-9\?\!\"\']', text)
ufffd_matches = [c for c in text if ord(c) == 0xFFFD]
print("dY artifacts remaining:", len(dY_matches))
print("U+FFFD replacement chars remaining:", len(ufffd_matches))
