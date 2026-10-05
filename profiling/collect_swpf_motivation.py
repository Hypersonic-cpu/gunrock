#!/usr/bin/env python3
"""Collect four software controls and active warp-cycle breakdowns.

The prefetch-only executable is an isolated original-loop-hints or
original-loop-lookahead build. The former has short state-hint lead time;
the latter adds future index loads to discover indirect addresses. Neither
is equivalent to GP/SPP with scheduling effects magically removed.
"""

import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

from diagnose_swpf import run, trace, write_csv
from validate_swpf import fixtures


KEY = ("app", "graph", "variant", "phase", "invocation")
STATES = ("barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss",
          "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
          "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
          "short_scoreboard", "sleeping", "tex_throttle", "wait")
COUNTERS = {
    "duration_ns": "gpu__time_duration.sum",
    "active_warps_per_scheduler": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "issued_warps_per_scheduler": "smsp__issue_active.avg.per_cycle_active",
    "issue_slots_used_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
    "grid_blocks": "launch__grid_size",
    "dram_bytes": "dram__bytes.sum",
    "dram_active_pct": "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_sectors": "lts__t_sectors.sum",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "warp_instructions": "smsp__inst_executed.sum",
    "sm_clock_hz": "gpc__cycles_elapsed.avg.per_second",
}
COUNTERS.update({state + "_pct":
                 f"smsp__warp_issue_stalled_{state}_per_warp_active.pct"
                 for state in STATES})


def raw_row(path):
    rows = [r for r in csv.DictReader(path.open()) if r.get("ID", "").isdigit()]
    assert len(rows) == 1, path
    return rows[0]


def pure_plan(root, binaries, controls):
    existing = json.loads((root / "case-plan.json").read_text())
    wanted = {tuple(r[k] for k in KEY[:3]) for r in controls}
    result = []
    for case in existing:
        if tuple(case[k] for k in KEY[:3]) in wanted:
            case["command"][0] = str(binaries / case["app"])
            result.append(case)
    return result


def validate(plan, output):
    rows = []
    small = fixtures(output / "fixtures")
    for case in plan:
        commands = []
        for graph, path, sources in small:
            for source in sources:
                command = list(case["command"])
                index = command.index("--binary-in")
                command[index:index + 2] = ["--market", str(path)]
                command[command.index("--src") + 1] = str(source)
                commands.append((graph, source, command))
        commands.append((case["graph"], 13331, case["command"]))
        for graph, source, command in commands:
            folder = output / "validation" / case["app"] / graph / str(source) / case["variant"]
            if not (folder / "validate.log").exists():
                run([*command, "--validate"], folder, "validate")
            assert "Number of errors : 0" in (folder / "validate.log").read_text()
            rows.append(dict(app=case["app"], graph=graph, source=source,
                             variant=case["variant"], status="OK", log=str(folder / "validate.log")))
            write_csv(output / "validation-summary.csv", rows)
        print(f"Validated prefetch-only {case['app']} {case['graph']} {case['variant']}", flush=True)


def collect(root, output, controls, plan, only_prefetch=False):
    primary = list(csv.DictReader((root / "ncu-verified.csv").open()))
    lookup = {tuple(r[k] for k in KEY): r for r in primary}
    invocations = list(csv.DictReader((output / "merge-path-invocations.csv").open()))
    pure_lookup = {tuple(r[k] for k in KEY): r for r in invocations}
    case_lookup = {tuple(r[k] for k in KEY[:3]): r for r in plan}
    rows = []
    for index, control in enumerate(controls, 1):
        key = tuple(control[k] for k in KEY)
        original = lookup[key[:2] + ("none",) + key[3:]]
        for mode, reference in (("original", original), ("prefetch_only", lookup[key]),
                                ("schedule_prefetch", lookup[key]), ("schedule_only", control)):
            if only_prefetch and mode != "prefetch_only":
                continue
            folder = output / "ncu" / control["app"] / control["graph"] / control["variant"] / f"{control['phase']}-{control['invocation']}" / mode
            source = Path(reference["report"]).parent
            old = shlex.split((source / "profile-command.txt").read_text())
            command = []
            i = 0
            while i < len(old):
                if old[i] in ("--section", "--metrics"):
                    i += 2
                else:
                    command.append(old[i])
                    i += 1
            command[command.index("--export") + 1] = str(folder / "report")
            executable_index = next(i for i, value in enumerate(command)
                                    if value.endswith("/bin/" + control["app"]))
            expected = reference["grid_blocks"]
            if mode == "prefetch_only":
                case = case_lookup[key[:3]]
                command[executable_index:] = case["command"]
                kernel = pure_lookup[key]
                command[command.index("--kernel-name") + 1] = kernel["mangled_name"]
                expected = kernel["grid_blocks"]
            command[executable_index:executable_index] = ["--metrics", ",".join(COUNTERS.values())]
            print(f"NCU breakdown [{index}/{len(controls)}] {key} {mode}", flush=True)
            if not (folder / "report.ncu-rep").exists():
                run(command, folder, "profile")
            raw = folder / "raw.csv"
            with raw.open("w") as stream:
                subprocess.run([shutil.which("ncu"), "--import", str(folder / "report.ncu-rep"),
                                "--page", "raw", "--csv", "--print-units", "base",
                                "--print-kernel-base", "mangled"], stdout=stream, check=True)
            data = raw_row(raw)
            assert data["Kernel Name"] == command[command.index("--kernel-name") + 1]
            row = {k: control[k] for k in KEY}
            row.update(mode=mode, **{name: float(data[metric].replace(",", ""))
                                   for name, metric in COUNTERS.items()})
            assert int(row["grid_blocks"]) == int(expected)
            row["sync_pct"] = row["barrier_pct"] + row["membar_pct"]
            row["eligible_active_warp_pct"] = 100 * row["eligible_warps_per_scheduler"] / row["active_warps_per_scheduler"]
            row["no_issue_slot_pct"] = 100 - row["issue_slots_used_pct"]
            row["state_sum_pct"] = sum(row[s + "_pct"] for s in STATES)
            # Application replay can change frontier order and memory timing;
            # state ratios may be sampled in separate passes. Preserve their
            # measured sum instead of silently renormalizing the CSV.
            assert abs(row["state_sum_pct"] - 100) < 10, row
            row.update(raw_csv=str(raw), report=str(folder / "report.ncu-rep"))
            rows.append(row)
            write_csv(output / "warp-cycle-breakdown.csv", rows)


def summarize(root, output):
    rows = list(csv.DictReader((output / "warp-cycle-breakdown.csv").open()))
    baseline = {tuple(r[k] for k in KEY): r for r in rows if r["mode"] == "original"}
    performance, positive = [], []
    for row in rows:
        original = baseline[tuple(row[k] for k in KEY)]
        speedup = float(original["duration_ns"]) / float(row["duration_ns"])
        perf = {k: row[k] for k in (*KEY, "mode")}
        perf.update(duration_ms=float(row["duration_ns"]) / 1e6, speedup=speedup,
                    same_grid=row["grid_blocks"] == original["grid_blocks"],
                    dram_bytes_ratio=float(row["dram_bytes"]) / float(original["dram_bytes"]),
                    instruction_ratio=float(row["warp_instructions"]) / float(original["warp_instructions"]))
        performance.append(perf)
        if row["mode"] != "original" and speedup > 1:
            positive.append({**row, "speedup": speedup})
    write_csv(output / "kernel-performance.csv", performance)
    write_csv(output / "positive-combinations.csv", positive)
    primary = list(csv.DictReader((root / "nsys-summary.csv").open()))
    schedule = list(csv.DictReader((root / "no-hint-timeline/nsys-summary.csv").open()))
    lookahead = root / "motivation-lookahead/nsys-summary.csv"
    pure = list(csv.DictReader((lookahead if lookahead.exists() else output / "nsys-summary.csv").open()))
    timeline = []
    base = {(r["app"], r["graph"], r["phase"]): float(r["merge_path_ms"])
            for r in primary if r["variant"] == "none"}
    for mode, source in (("original", primary), ("schedule_prefetch", primary),
                         ("schedule_only", schedule), ("prefetch_only", pure)):
        for row in source:
            if (row["variant"] == "none") != (mode == "original"):
                continue
            key = row["app"], row["graph"], row["phase"]
            timeline.append({**row, "mode": mode, "phase_speedup": base[key] / float(row["merge_path_ms"])})
    write_csv(output / "phase-performance.csv", timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing diagnostics directory")
    parser.add_argument("--output-name", default="motivation-controls")
    parser.add_argument("--prefetch-build", type=Path)
    parser.add_argument("--only-prefetch", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    controls = list(csv.DictReader((root / "no-hint-timeline/ncu-verified.csv").open()))
    binaries = args.prefetch_build.resolve() if args.prefetch_build else output / "prefetch-only-build/bin"
    plan = pure_plan(root, binaries, controls)
    (output / "prefetch-only-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    validate(plan, output)
    trace(SimpleNamespace(output=output), plan)
    collect(root, output, controls, plan, args.only_prefetch)
    if not args.only_prefetch:
        summarize(root, output)
    print("Finished four-way controls and warp-cycle CSVs", flush=True)


if __name__ == "__main__":
    main()
