import matplotlib.pyplot as plt
import subprocess
import json
import sys
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.table import Table


class Benchmark:

    def __init__(self, executable: str):
        self.executable = executable
        self.args = []
        self.argnames = []
        self.console = Console()

    def add_arg_group(self, names: tuple[str] | str, group: list[tuple] | list):
        self.argnames.append(str(names))
        if not isinstance(group[0], tuple):
            # wrap args into tuples if they haven't been in
            group = list(map(lambda x: (x,), group))
        self.args.append(group)
        return self

    def run(self):
        table = Table(box=box.ROUNDED)
        colomns: list[Table] = []
        prev_args = [(None, None)] * len(self.argnames)

        def gen_args(s: list[list[tuple]]):
            """Generate the argument lists to be passed to the excutable."""
            if len(s) == 0:
                yield []
            else:
                for i in s[0]:
                    for j in gen_args(s[1::]):  # recursively get argument list
                        yield [i] + j  # merge arguments

        for header in self.argnames + ["name", "time", "result"]:
            subtable = Table(
                show_header=False,
                min_width=self.console.measure(header).minimum,
                box=box.MINIMAL,
            )
            subtable.add_column(header)
            colomns.append(subtable)
            table.add_column(header)
        colomns[-2].columns[0].justify = "right"

        table.add_row(*colomns)

        with Live(
            table,
            console=self.console,
            refresh_per_second=1,
            vertical_overflow="visible",
        ):
            for args in gen_args(self.args):
                process = subprocess.Popen(
                    [self.executable]
                    + [
                        str(arg) for arg_tuple in args for arg in arg_tuple
                    ],  # flatten the arg list
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                for index, arg in enumerate(args):
                    if len(arg) == 1:
                        arg_str = str(arg[0])
                    else:
                        arg_str = str(arg)
                    if arg_str != prev_args[index][0]:
                        if prev_args[index][1] is not None:
                            # to offset the height of the section line
                            prev_args[index][1].height -= 1
                        arg_align = Align(arg_str, vertical="middle", height=0)
                        prev_args[index] = (arg_str, arg_align)
                        colomns[index].add_row(
                            arg_align,
                            end_section=True,
                        )

                for line in process.stdout:
                    if line:
                        try:
                            result = json.loads(line)
                            colomns[-3].add_row(result["name"])
                            colomns[-2].add_row(f"{result["time"]:.2f}ms")
                            colomns[-1].add_row(result["result"])
                            for _, arg_align in prev_args:
                                arg_align.height += 1
                        except:
                            pass
                colomns[-3].add_section()
                colomns[-2].add_section()
                colomns[-1].add_section()
                for _, arg_align in prev_args:
                    arg_align.height += 1
                process.wait()


# predefined benchmark behaviors
benchmarks = {
    "vecadd": Benchmark("vecadd/build/vecadd")
    .add_arg_group("N", [100, 10000, 1000000, 100000000])
    .add_arg_group("block_sizes", [64, 128, 256, 512]),
    "matmul": Benchmark("matmul/build/matmul").add_arg_group(
        ("N", "K", "M"),
        [
            (37, 73, 41),
            (100, 100, 100),
            (200, 400, 600),
            (1000, 1000, 1000),
            (2000, 1500, 4000),
        ],
    ),
    "matmul-opt": Benchmark("matmul-opt/build/matmul-opt")
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
    .add_arg_group("coarse_factor", [1, 4, 16, 64]),
    "wmma": Benchmark("wmma/build/wmma").add_arg_group(
        ("N", "K", "M"),
        [
            (37, 73, 41),
            (100, 100, 100),
            (200, 400, 600),
            (1000, 1000, 1000),
            (2000, 1500, 4000),
            (10000, 15000, 10000),
        ],
    ),
    "conv": Benchmark("conv/build/conv")
    .add_arg_group(("W", "H"), [(8192, 16384), (32768, 16384)])
    .add_arg_group("radius", [2, 4, 8])
    .add_arg_group("tile_widths", [16, 32]),
    "stencil": Benchmark("stencil/build/stencil")
    .add_arg_group("N", [50, 100, 800])
    .add_arg_group("tile_width", [4, 8]),
}


def main():
    if len(sys.argv) < 2:
        print("usage: python3 benchmark.py target_name")
        exit(1)
    benchmarks[sys.argv[1]].run()


if __name__ == "__main__":
    main()
