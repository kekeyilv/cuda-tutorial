from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/vecadd")
        .add_arg_group("N", [100, 10000, 1000000, 100000000])
        .add_arg_group("block_size", [64, 128, 256, 512])
    )
    benchmark.run()
