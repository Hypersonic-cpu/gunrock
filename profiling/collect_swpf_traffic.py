#!/usr/bin/env python3
"""Collect direct DRAM/sector counters for none, hints and no-hint controls."""

import argparse
import csv
from pathlib import Path
import shlex
import shutil
import subprocess

from diagnose_swpf import run, write_csv


METRICS = ("gpu__time_duration.sum", "dram__bytes.sum", "dram__bytes_read.sum",
           "dram__bytes_write.sum", "lts__t_sectors.sum",
           "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
           "launch__grid_size", "launch__registers_per_thread")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    primary = list(csv.DictReader((root / "ncu-verified.csv").open()))
    controls = list(csv.DictReader((root / "no-hint-timeline/ncu-verified.csv").open()))
    lookup = {(r["app"], r["graph"], r["variant"], r["phase"], r["invocation"]): r for r in primary}
    rows = []
    for index, control in enumerate(controls, 1):
        key = tuple(control[field] for field in ("app", "graph", "variant", "phase", "invocation"))
        baseline_key = key[:2] + ("none",) + key[3:]
        for mode, reference in (("none", lookup[baseline_key]), ("hints", lookup[key]), ("no-hints", control)):
            print(f"Traffic [{index}/{len(controls)}] {key} {mode}", flush=True)
            folder = root / "traffic" / control["app"] / control["graph"] / control["variant"] / f"{control['phase']}-{control['invocation']}" / mode
            source = Path(reference["report"]).parent
            original = shlex.split((source / "profile-command.txt").read_text())
            command = []
            i = 0
            while i < len(original):
                if original[i] in ("--section", "--metrics"):
                    i += 2
                else:
                    command.append(original[i])
                    i += 1
            command[command.index("--export") + 1] = str(folder / "report")
            # Profiler arguments must precede the target executable.
            executable_index = next(i for i, value in enumerate(command) if
                                    value.endswith("/bin/" + control["app"]))
            command[executable_index:executable_index] = ["--metrics", ",".join(METRICS)]
            if not (folder / "report.ncu-rep").exists():
                run(command, folder, "profile")
            raw = folder / "raw.csv"
            with raw.open("w") as stream:
                subprocess.run([shutil.which("ncu"), "--import", str(folder / "report.ncu-rep"),
                                "--page", "raw", "--csv", "--print-units", "base",
                                "--print-kernel-base", "mangled"], stdout=stream, check=True)
            data = [row for row in csv.DictReader(raw.open()) if row["ID"].isdigit()]
            assert len(data) == 1 and int(float(data[0]["launch__grid_size"].replace(",", ""))) == int(reference["grid_blocks"])
            rows.append(dict(app=control["app"], graph=control["graph"], variant=control["variant"],
                             phase=control["phase"], invocation=control["invocation"], mode=mode,
                             **{metric: float(data[0][metric].replace(",", "")) for metric in METRICS},
                             report=str(folder / "report.ncu-rep")))
            write_csv(root / "traffic-counters.csv", rows)


if __name__ == "__main__":
    main()
