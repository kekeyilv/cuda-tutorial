from benchmark import Benchmark

if __name__ == "__main__":
    benchmark = (
        Benchmark("../build/stencil")
        .add_arg_group("N", [50, 100, 800])
        .add_arg_group(("tw3d", "tw2d"), [(4, 8), (8, 16), (8, 32)])
        .add_arg_group("coarsen_height", [256, 1024])
    )
    benchmark.run()
