import urllib.request
import json
import time

t0 = time.time()
try:
    res = urllib.request.urlopen("http://localhost:8080/api/v1/customers/CUST0000001", timeout=30)
    data = json.loads(res.read().decode())
    print(f"Customer API response received in {time.time()-t0:.2f}s:")
    print("Decision:", data["credit_decision"])
    print("Score:   ", data["composite_score"], "(Grade", data["credit_grade"] + ")")
    print("PD%:     ", data["calibrated_pd_pct"])
    print("Limits:  ", data["credit_limits"])
    print("Pricing: ", data["pricing"])
except Exception as e:
    print("Request failed:", e)
