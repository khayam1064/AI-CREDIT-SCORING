with open('dashboard/app.py', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Replace any  characters with clean symbols
text = text.replace('b_icon = {"APPROVED": "o.", "REFER": "s,?", "DECLINED": "?O"}.get(dec_val, "?")', 'b_icon = {"APPROVED": "✅", "REFER": "⚠️", "DECLINED": "❌"}.get(dec_val, "❌")')
text = text.replace('', '')

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Cleaned all  character artifacts!")
