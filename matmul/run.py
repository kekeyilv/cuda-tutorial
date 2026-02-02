import subprocess

Dims = [
    (37, 73, 41),
    (100, 100, 100),
    (200, 400, 600),
    (1000, 1000, 1000),
    (2000, 1500, 4000),
]
for N, K, M in Dims:
    print(f"[N = {N}, K = {K}, M = {M}]")
    subprocess.run(["./build/matmul", str(N), str(K), str(M)])
