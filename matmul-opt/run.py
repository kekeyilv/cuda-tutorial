import subprocess

dims = [
    (37, 73, 41),
    (100, 100, 100),
    (1000, 1000, 1000),
    (2000, 1500, 4000),
    (10000, 10000, 10000),
]

tile_widths = [2, 8, 16, 32]

for N, K, M in dims:
    print(f"[N = {N}, K = {K}, M = {M}]")
    for tile_width in tile_widths:
        print(f"[tile_width = {tile_width}]")
        subprocess.run(
            ["./build/matmul-opt", str(N), str(K), str(M), str(tile_width), "1"]
        )
