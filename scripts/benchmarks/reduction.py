from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/reduction")
        .add_arg_group("N", [1000000, 500000000])
        .add_arg_group("block_size", [256, 1024])
        .add_arg_group("coarsen_factor", [1, 4, 16])
    )
    benchmark.run()
