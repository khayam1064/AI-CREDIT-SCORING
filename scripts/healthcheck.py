import urllib.request
try:
    res = urllib.request.urlopen("http://localhost:8502", timeout=5)
    print("Healthcheck:", res.getcode(), "OK")
except Exception as e:
    print("Healthcheck error:", e)
