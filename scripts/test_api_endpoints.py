import urllib.request
import json

endpoints = [
    'http://localhost:8000/health',
    'http://localhost:8000/api/v1/customers/CUST0000001',
    'http://localhost:8000/api/v1/portfolio/ifrs9',
    'http://localhost:8000/api/v1/portfolio/drift-monitoring'
]

for url in endpoints:
    try:
        res = urllib.request.urlopen(url, timeout=5)
        data = json.loads(res.read().decode())
        status_key = data.get("status", "OK")
        print(f"SUCCESS: {url} -> HTTP {res.getcode()} | Status: {status_key}")
    except Exception as e:
        print(f"FAIL: {url} -> {e}")
