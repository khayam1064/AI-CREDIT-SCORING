import sys, base64

mode = sys.argv[1] # 'w' or 'a'
b64_data = sys.argv[2]
raw = base64.b64decode(b64_data).decode('utf-8')

with open('dashboard/app.py', mode, encoding='utf-8') as f:
    f.write(raw)

print(f'Wrote {len(raw)} chars in mode {mode}')
