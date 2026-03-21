from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/stencil")
        .add_arg_group("N", [50, 100, 800])
        .add_arg_group("tile_width", [4, 8])
    )
    benchmark.run()
