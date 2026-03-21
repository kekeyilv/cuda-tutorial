from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/matmul-opt")
        .add_arg_group(
            ("N", "K", "M"),
            [
                (37, 73, 41),
                (100, 100, 100),
                (1000, 1000, 1000),
                (2000, 1500, 4000),
                (10000, 10000, 10000),
            ],
        )
        .add_arg_group("tile_width", [2, 8, 16, 32])
        .add_arg_group("padding", [1])
        .add_arg_group("coarse_factor", [1, 4, 16, 64])
    )
    benchmark.run()
