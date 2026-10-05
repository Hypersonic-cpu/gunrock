#!/usr/bin/env python3
"""Inspect compiled prefetch kernels and CUDA theoretical occupancy (no profiler)."""

import argparse
import csv
import ctypes as ct
from pathlib import Path
import re
import subprocess


def checked(code):
    if code:
        raise RuntimeError(f"CUDA driver call failed with error {code}")


def driver():
    cuda = ct.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ct.c_uint],
        "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
        "cuDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
        "cuDevicePrimaryCtxRetain": [ct.POINTER(ct.c_void_p), ct.c_int],
        "cuCtxSetCurrent": [ct.c_void_p],
        "cuModuleLoad": [ct.POINTER(ct.c_void_p), ct.c_char_p],
        "cuModuleUnload": [ct.c_void_p],
        "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
        "cuFuncGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
        "cuOccupancyMaxActiveBlocksPerMultiprocessor": [
            ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
    }
    for name, signature in signatures.items():
        getattr(cuda, name).argtypes = signature
        getattr(cuda, name).restype = ct.c_int
    checked(cuda.cuInit(0))
    device = ct.c_int()
    checked(cuda.cuDeviceGet(ct.byref(device), 0))
    context = ct.c_void_p()
    checked(cuda.cuDevicePrimaryCtxRetain(ct.byref(context), device))
    checked(cuda.cuCtxSetCurrent(context))
    max_threads = ct.c_int()
    checked(cuda.cuDeviceGetAttribute(ct.byref(max_threads), 39, device))
    return cuda, max_threads.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cuda, max_threads = driver()
    rows = []
    for app in ("bfs", "sssp", "bc"):
        executable = (args.build_bin / app).resolve()
        app_folder = folder / app
        app_folder.mkdir(exist_ok=True)
        subprocess.run([args.cuobjdump, "--extract-elf", "all", str(executable)],
                       cwd=app_folder, check=True, capture_output=True, text=True)
        cubin = next(app_folder.glob("*.sm_86.cubin"))
        module = ct.c_void_p()
        checked(cuda.cuModuleLoad(ct.byref(module), str(cubin).encode()))
        usage = subprocess.check_output([args.cuobjdump, "--dump-resource-usage",
                                         str(executable)], text=True)
        ptx = subprocess.check_output([args.cuobjdump, "--dump-ptx",
                                       str(executable)], text=True)
        (app_folder / "resource-usage.txt").write_text(usage)
        (app_folder / "kernels.ptx").write_text(ptx)
        entries = list(re.finditer(r"\.visible \.entry ([^\s(]+)", ptx))
        ptx_bodies = {match[1]: ptx[match.start():entries[i + 1].start()
                                  if i + 1 < len(entries) else len(ptx)]
                      for i, match in enumerate(entries)}
        functions = re.findall(r" Function (\S+):\n\s+([^\n]+)", usage)
        for name, resources in functions:
            if "merge_path_kernel" not in name:
                continue
            demangled = subprocess.check_output(["c++filt", name], text=True).strip()
            variant = re.search(r"swpf_algorithm_t\)(\d+), .*?swpf_target_t\)(\d+), (\d+),", demangled)
            if not variant:
                raise RuntimeError(f"Cannot identify kernel variant: {demangled}")
            algorithm, target, distance = map(int, variant.groups())
            phase = app
            if app == "bc":
                output_type = re.findall(r"advance_io_type_t\)(\d+)", demangled)[1]
                phase = "bc_backward" if output_type == "3" else "bc_forward"
            function = ct.c_void_p()
            checked(cuda.cuModuleGetFunction(ct.byref(function), module, name.encode()))
            attributes = {}
            for field, attr in (("registers_per_thread", 4), ("shared_bytes", 1),
                                ("local_bytes_per_thread", 3)):
                value = ct.c_int()
                checked(cuda.cuFuncGetAttribute(ct.byref(value), attr, function))
                attributes[field] = value.value
            blocks = ct.c_int()
            checked(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ct.byref(blocks), function, 256, 0))
            body = ptx_bodies[name]
            l1 = body.count("prefetch.global.L1")
            l2 = body.count("prefetch.global.L2")
            if algorithm == 0 and (l1 or l2):
                raise RuntimeError(f"none kernel contains prefetch: {name}")
            if algorithm != 0 and l2 == 0:
                raise RuntimeError(f"missing L2 atomic prefetch: {name}")
            if target == 1 and l1:
                raise RuntimeError(f"l2 variant contains L1 prefetch: {name}")
            rows.append(dict(app=app, phase=phase, swpf=("none", "gp", "spp")[algorithm],
                             target=("l1", "l2")[target], distance=distance,
                             **attributes, active_blocks_per_sm=blocks.value,
                             theoretical_occupancy=blocks.value * 256 / max_threads,
                             ptx_l1_instructions=l1, ptx_l2_instructions=l2,
                             resource_usage=resources, symbol=name))
        checked(cuda.cuModuleUnload(module))
    if len(rows) != 68:
        raise RuntimeError(f"Expected 68 kernel variants, found {len(rows)}")
    with (folder / "kernel-resources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} kernels: none contains no explicit prefetch; "
          "atomic targets use L2; occupancy is a CUDA resource estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
