from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/conv")
        .add_arg_group(("W", "H"), [(8192, 16384), (32768, 16384)])
        .add_arg_group("radius", [2, 4, 8])
        .add_arg_group("tile_width", [16, 32])
    )
    benchmark.run()
