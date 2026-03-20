import subprocess

Ns = [50, 100, 800]
tile_widths = [4, 8]

for N in Ns:
    print(f"[N = {N}]")
    for tile_width in tile_widths:
        print(f"[tile_width = {tile_width}]")
        subprocess.run(
            [
                "./build/stencil",
                str(N),
                str(tile_width),
            ]
        )
