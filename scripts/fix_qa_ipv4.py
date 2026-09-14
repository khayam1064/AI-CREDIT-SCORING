with open("scripts/qa_stress_test_suite.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace localhost with 127.0.0.1 in urllib calls to prevent Windows IPv6 dual-stack fallback
code = code.replace("http://localhost:8080/health", "http://127.0.0.1:8080/health")
code = code.replace("http://localhost:8502", "http://127.0.0.1:8502")

with open("scripts/qa_stress_test_suite.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated QA test script with direct IPv127 loopback!")
