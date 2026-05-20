from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/sort")
        .add_arg_group("N", [1000, 1000000, 100000000])
        .add_arg_group("block_size", [256, 1024])
        .add_arg_group("merge_tile_size", [32])
    )
    benchmark.run()
