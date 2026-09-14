import socket
import time

for target in ["127.0.0.1", "localhost"]:
    t0 = time.time()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((target, 8080))
    s.send(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    resp = s.recv(1024)
    s.close()
    elapsed = (time.time() - t0) * 1000
    print(f"Target '{target}' raw TCP request-response latency: {elapsed:.2f} ms")
