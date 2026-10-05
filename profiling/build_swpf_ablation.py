#!/usr/bin/env python3
"""Build isolated hint/scheduling controls using the original CMake flags."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("no-hints", "original-loop-hints", "original-loop-lookahead"),
                        default="no-hints")
    args = parser.parse_args()
    repo, build, output = Path.cwd(), args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    include = output / "include"
    shutil.copytree(repo / "include", include, dirs_exist_ok=True)
    header = include / "gunrock/framework/operators/advance/swpf/prefetch_ptx.hxx"
    original = header.read_text()
    modified = original.replace('asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));', '(void)ptr;')
    modified = modified.replace('asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));', '(void)ptr;')
    assert modified != original and "asm volatile" not in modified
    if args.mode == "no-hints":
        header.write_text(modified)
    else:
        # A diagnostic control: preserve the original serial merge loop and
        # add hints without buffering edges or moving actual demand loads.
        merge = include / "gunrock/framework/operators/advance/merge_path.hxx"
        text = merge.read_text()
        needle = "  if constexpr (algorithm == swpf_algorithm_t::none) {"
        assert text.count(needle) == 1
        text = text.replace(needle, "  if constexpr (true) {")
        needle = "              auto e = starting_edge + rank;\n"
        assert text.count(needle) == 1
        text = text.replace(needle, needle + """
              if constexpr (algorithm != swpf_algorithm_t::none) {
                // Stay inside this CSR row. State addresses become known only
                // after the original destination load below.
                if (global_atom + distance < tile_row_end_offsets[thread_start.x]) {
                  swpf::prefetch_read<target>(G.get_column_indices() + e + distance);
                  if constexpr (state_prefetch_t::uses_edge_weights)
                    swpf::prefetch_read<target>(G.get_nonzero_values() + e + distance);
                }
              }
""")
        needle = "              bool cond = op(v, n, e, w);"
        assert text.count(needle) == 1
        text = text.replace(needle, """              if constexpr (algorithm != swpf_algorithm_t::none)
                state_prefetch.template operator()<target>(v, n);

""" + needle)
        if args.mode == "original-loop-lookahead":
            needle = "                    swpf::prefetch_read<target>(G.get_nonzero_values() + e + distance);"
            text = text.replace(needle, needle + """
                  // Resolve only the future address, without buffering or
                  // changing the order of current edge loads/operations.
                  const auto future_dst = G.get_destination_vertex(e + distance);
                  state_prefetch.template operator()<target>(v, future_dst);
""")
            text = text.replace("""              if constexpr (algorithm != swpf_algorithm_t::none)
                state_prefetch.template operator()<target>(v, n);

""", "")
        merge.write_text(text)
    (output / "prefetch_ptx-original.hxx").write_text(original)
    (output / "bin").mkdir(exist_ok=True)

    def compile_app(app):
        directory = build / "examples/algorithms" / app / "CMakeFiles" / (app + ".dir")
        variables = {}
        for line in (directory / "flags.make").read_text().splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                variables[key] = value
        includes = shlex.split((directory / "includes_CUDA.rsp").read_text())
        includes = ["-I" + str(include) if token == "-I" + str(repo / "include") else token
                    for token in includes]
        obj = output / (app + ".o")
        compile_command = ["/usr/local/cuda/bin/nvcc", "-forward-unknown-to-host-compiler",
                           *shlex.split(variables["CUDA_DEFINES"]), *includes,
                           *shlex.split(variables["CUDA_FLAGS"]), "-x", "cu", "-c",
                           str(repo / "examples/algorithms" / app / (app + ".cu")),
                           "-o", str(obj)]
        link_command = ["/usr/bin/g++", str(obj), "-o", str(output / "bin" / app),
                        *shlex.split((directory / "linkLibs.rsp").read_text()),
                        "-L/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs",
                        "-L/usr/local/cuda-12.6/targets/x86_64-linux/lib"]
        (output / (app + "-commands.json")).write_text(json.dumps([compile_command, link_command], indent=2) + "\n")
        with (output / (app + "-build.log")).open("w") as log:
            subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(link_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Built isolated {args.mode} control: {app}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(compile_app, ("bfs", "sssp", "bc")))


if __name__ == "__main__":
    main()
