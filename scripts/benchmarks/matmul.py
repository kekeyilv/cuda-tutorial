from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = Benchmark("../build/matmul").add_arg_group(
        ("N", "K", "M"),
        [
            (37, 73, 41),
            (100, 100, 100),
            (200, 400, 600),
            (1000, 1000, 1000),
            (2000, 1500, 4000),
        ],
    )
    benchmark.run()
