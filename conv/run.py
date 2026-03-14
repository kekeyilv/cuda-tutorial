import subprocess

dims = [(8192, 16384), (32768, 16384)]
radii = [2, 4, 8]
tile_widths = [16, 32]

for W, H in dims:
    print(f"[W = {W}, H = {H}]")
    for radius in radii:
        print(f"[radius = {radius}]")
        for tile_width in tile_widths:
            print(f"[tile_width = {tile_width}]")
            subprocess.run(
                [
                    "./build/conv",
                    str(W),
                    str(H),
                    str(radius),
                    str(tile_width),
                ]
            )
