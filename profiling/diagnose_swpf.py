#!/usr/bin/env python3
"""Capture single-run SWPF timelines and exact-invocation application replays."""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from analyze_nsys_invocations import load_kernels
from run_native_suite import Suite


SECTIONS = ("LaunchStats", "Occupancy", "SpeedOfLight", "MemoryWorkloadAnalysis",
            "SchedulerStats", "WarpStateStats", "InstructionStats")
EXTRA_METRICS = (
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__inst_executed.sum",
)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(command, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (stem + "-command.txt")).write_text(shlex.join(command) + "\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "8"
    with (folder / (stem + ".log")).open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {folder}/{stem}.log")


def cases(native, binaries, inputs):
    rows = list(csv.DictReader((native / "best-variants.csv").open()))
    result = []
    for row in rows:
        for algorithm in ("none", "gp", "spp"):
            target, distance = "l2", 2
            if algorithm != "none":
                match = re.fullmatch(r"(l[12]), d=(\d+)", row["best_" + algorithm])
                target, distance = match[1], int(match[2])
            label = "none" if algorithm == "none" else f"{algorithm}-{target}-d{distance}"
            result.append(dict(app=row["app"], graph=row["graph"], variant=label,
                               swpf=algorithm, target=target, distance=distance,
                               command=[str((binaries / row["app"]).resolve()),
                                        "--binary-in", str(inputs / (row["graph"] + ".bin")),
                                        "--src", "13331", "--num_runs", "1",
                                        "--advance", "merge_path", "--swpf", algorithm,
                                        "--swpf-target", target, "--swpf-distance", str(distance)]))
    return result


def phase_groups(kernels, app):
    groups = defaultdict(list)
    for kernel in kernels:
        if "merge_path_kernel" in kernel["kernel_function"]:
            groups[kernel["mangled_name"]].append(kernel)
    # BC's forward phase precedes backward and has a different exact function.
    ordered = sorted(groups.values(), key=lambda group: group[0]["start_ns"])
    expected = 2 if app == "bc" else 1
    if len(ordered) != expected:
        raise RuntimeError(f"Expected {expected} merge-path functions, got {len(ordered)}")
    phases = ("forward", "backward") if app == "bc" else ("advance",)
    return dict(zip(phases, ordered))


def trace(args, plan):
    summaries, invocation_rows, ranked_rows = [], [], []
    for index, case in enumerate(plan, 1):
        folder = args.output / "nsys" / case["app"] / case["graph"] / case["variant"]
        report, database = folder / "report.nsys-rep", folder / "report.sqlite"
        print(f"NSYS [{index}/{len(plan)}] {case['app']} {case['graph']} {case['variant']}", flush=True)
        if not report.exists():
            run(["sudo", "-n", shutil.which("nsys"), "profile", "--trace=cuda,nvtx",
                 "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
                 "--output=" + str(folder / "report"), *case["command"]], folder, "profile")
        if not database.exists():
            run([shutil.which("nsys"), "export", "--type=sqlite", "--force-overwrite=true",
                 "--output=" + str(database), str(report)], folder, "export")
        kernels = load_kernels(database)
        all_groups = defaultdict(list)
        for kernel in kernels:
            all_groups[kernel["mangled_name"]].append(kernel)
        for group in sorted(all_groups.values(), key=lambda g: sum(k["duration_ms"] for k in g), reverse=True):
            ranked_rows.append(dict(app=case["app"], graph=case["graph"], variant=case["variant"],
                                    function=group[0]["kernel_function"], count=len(group),
                                    total_ms=sum(k["duration_ms"] for k in group),
                                    max_ms=max(k["duration_ms"] for k in group),
                                    demangled=group[0]["demangled_name"]))
        for phase, group in phase_groups(kernels, case["app"]).items():
            total = sum(k["duration_ms"] for k in group)
            top = sorted(enumerate(group, 1), key=lambda pair: pair[1]["duration_ms"], reverse=True)[:3]
            summaries.append(dict(app=case["app"], graph=case["graph"], variant=case["variant"],
                                  phase=phase, launches=len(group), merge_path_ms=total,
                                  top_iterations=";".join(str(i) for i, _ in top),
                                  top3_share=sum(k["duration_ms"] for _, k in top) / total))
            for ordinal, kernel in enumerate(group, 1):
                invocation_rows.append(dict(app=case["app"], graph=case["graph"], variant=case["variant"],
                                            phase=phase, invocation=ordinal, **kernel))
        write_csv(args.output / "nsys-summary.csv", summaries)
        write_csv(args.output / "kernel-ranking.csv", ranked_rows)
        write_csv(args.output / "merge-path-invocations.csv", invocation_rows)


def ncu(args, plan):
    rows = list(csv.DictReader((args.output / "merge-path-invocations.csv").open()))
    jobs = []
    for case in plan:
        for phase in (("forward", "backward") if case["app"] == "bc" else ("advance",)):
            selected = [row for row in rows if row["app"] == case["app"] and row["graph"] == case["graph"]
                        and row["variant"] == case["variant"] and row["phase"] == phase]
            baseline = [row for row in rows if row["app"] == case["app"] and row["graph"] == case["graph"]
                        and row["variant"] == "none" and row["phase"] == phase]
            # Compare the same two baseline hotspots; also collect the variant's
            # own peak if its hotspot moved. Ordinals are per exact function.
            wanted = {int(row["invocation"]) for row in sorted(baseline, key=lambda r: float(r["duration_ms"]), reverse=True)[:2]}
            wanted.add(int(max(selected, key=lambda r: float(r["duration_ms"]))["invocation"]))
            for row in selected:
                if int(row["invocation"]) in wanted:
                    jobs.append(dict(case=case, phase=phase, expected=row))
    (args.output / "ncu-plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
    verified = []
    for index, job in enumerate(jobs, 1):
        case, expected = job["case"], job["expected"]
        ordinal = int(expected["invocation"])
        folder = args.output / "ncu" / case["app"] / case["graph"] / case["variant"] / f"{job['phase']}-{ordinal}"
        report = folder / "report.ncu-rep"
        print(f"NCU [{index}/{len(jobs)}] {case['app']} {case['graph']} {case['variant']} {job['phase']} #{ordinal}", flush=True)
        if not report.exists():
            command = ["sudo", "-n", shutil.which("ncu"), "--replay-mode", "application",
                       "--app-replay-match", "grid", "--app-replay-mode", "strict",
                       "--cache-control", "none", "--clock-control", "none",
                       "--kernel-name-base", "mangled", "--kernel-name", expected["mangled_name"],
                       "--launch-skip", str(ordinal - 1), "--launch-count", "1",
                       "--force-overwrite", "--export", str(folder / "report")]
            for section in SECTIONS:
                command += ["--section", section]
            command += ["--metrics", ",".join(EXTRA_METRICS), *case["command"]]
            run(command, folder, "profile")
        csv_path = folder / "raw.csv"
        if not csv_path.exists():
            export = [shutil.which("ncu"), "--import", str(report), "--page", "raw", "--csv", "--print-units", "base", "--print-kernel-base", "mangled"]
            with csv_path.open("w") as stream:
                subprocess.run(export, stdout=stream, check=True)
        data = list(csv.DictReader(csv_path.open()))
        data = [row for row in data if row.get("ID", "").isdigit()]
        ids = {row["ID"] for row in data}
        if len(ids) != 1 or any(row["Kernel Name"] != expected["mangled_name"] for row in data):
            raise RuntimeError(f"Wrong kernel or result count in {csv_path}")
        # NCU's raw CSV is one wide row per profiled launch, preceded by a
        # units row. Also accept the long metric table from alternate exports.
        metrics = ({row["Metric Name"]: row["Metric Value"] for row in data}
                   if "Metric Name" in data[0] else data[0])
        grid = metrics.get("launch__grid_size")
        if grid is not None and int(float(grid.replace(",", ""))) != int(expected["grid_blocks"]):
            raise RuntimeError(f"Grid mismatch in {csv_path}")
        verified.append(dict(app=case["app"], graph=case["graph"], variant=case["variant"],
                             phase=job["phase"], invocation=ordinal, nsys_ms=expected["duration_ms"],
                             grid_blocks=expected["grid_blocks"], raw_csv=str(csv_path), report=str(report)))
        write_csv(args.output / "ncu-verified.csv", verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-bin", type=Path, default=Path("build-v2/bin"))
    parser.add_argument("--binary-root", type=Path,
                        help="Override binary_root from the captured native suite YAML")
    parser.add_argument("--stage", choices=("nsys", "ncu", "all"), default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    binary_root = args.binary_root or Suite.load(args.native_root / "suite-config.yaml").binary_root
    plan = cases(args.native_root, args.build_bin, binary_root)
    (args.output / "case-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.stage in ("nsys", "all"):
        trace(args, plan)
    if args.stage in ("ncu", "all"):
        ncu(args, plan)


if __name__ == "__main__":
    main()
