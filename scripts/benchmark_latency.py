import urllib.request
import json
import time

for cid in ["CUST0000001", "CUST0000002", "CUST0000022", "CUST0000090"]:
    t0 = time.time()
    res = urllib.request.urlopen(f"http://localhost:8080/api/v1/customers/{cid}", timeout=10)
    data = json.loads(res.read().decode())
    elapsed = (time.time() - t0) * 1000
    print(f"[{cid}] Latency: {elapsed:5.1f}ms | Decision: {data['credit_decision']:8s} | Score: {data['composite_score']:3d} | Grade: {data['credit_grade']} | Limit: PKR {data['credit_limits']['approved_facility_limit']:7,.0f}")
