import subprocess

Ns = [100, 10000, 1000000, 100000000]
block_sizes = [64, 128, 256, 512]
for N in Ns:
    for block_size in block_sizes:
        print(f"[N = {N}, blockSize = {block_size}]")
        subprocess.run(["./build/vecadd", str(N), str(block_size)])
