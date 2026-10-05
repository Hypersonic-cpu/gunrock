#!/usr/bin/env python3
"""Extract comparable Nsight counters from the SWPF exact-launch reports."""

import argparse
import csv
from pathlib import Path

from diagnose_swpf import write_csv


METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "sm_clock_hz": "gpc__cycles_elapsed.avg.per_second",
    "registers_per_thread": "launch__registers_per_thread",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
    "issued_warps_per_scheduler": "smsp__issue_active.avg.per_cycle_active",
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "dram_bytes_per_second": "dram__bytes.sum.per_second",
    "dram_active_pct": "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm_util_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_util_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "warp_instructions": "smsp__inst_executed.sum",
    "long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "short_scoreboard_pct": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "lg_throttle_pct": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "long_scoreboard_cycles_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "short_scoreboard_cycles_per_issue": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "lg_throttle_cycles_per_issue": "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "active_threads_per_warp": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "unpredicated_threads_per_warp": "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verified = list(csv.DictReader((args.root / "ncu-verified.csv").open()))
    rows = []
    for row in verified:
        raw = next(r for r in csv.DictReader(Path(row["raw_csv"]).open()) if r["ID"].isdigit())
        for label, name in METRICS.items():
            value = raw.get(name, "")
            row[label] = float(value.replace(",", "")) if value and value != "n/a" else ""
        row["warp_instructions_per_block"] = row["warp_instructions"] / float(row["grid_blocks"])
        # This is explicitly an estimate from two exported metrics. Direct
        # DRAM byte counters, when present, take priority over the rate product.
        direct_bytes = raw.get("dram__bytes.sum", "")
        row["dram_bytes_estimated"] = (float(direct_bytes.replace(",", "")) if direct_bytes
                                       else row["dram_bytes_per_second"] * row["duration_ns"] / 1e9)
        row["under_20us"] = row["duration_ns"] < 20000
        rows.append(row)
    write_csv(args.root / "ncu-counters.csv", rows)
    pairs = []
    baseline = {(r["app"], r["graph"], r["phase"], r["invocation"]): r
                for r in rows if r["variant"] == "none"}
    for row in rows:
        key = row["app"], row["graph"], row["phase"], row["invocation"]
        if row["variant"] == "none" or key not in baseline:
            continue
        original = baseline[key]
        pair = dict(app=row["app"], graph=row["graph"], phase=row["phase"],
                    invocation=row["invocation"], variant=row["variant"],
                    same_grid=row["grid_blocks"] == original["grid_blocks"],
                    baseline_grid=original["grid_blocks"], variant_grid=row["grid_blocks"],
                    baseline_ns=original["duration_ns"], variant_ns=row["duration_ns"],
                    ncu_speedup=original["duration_ns"] / row["duration_ns"],
                    baseline_nsys_ms=original["nsys_ms"], variant_nsys_ms=row["nsys_ms"],
                    nsys_speedup=float(original["nsys_ms"]) / float(row["nsys_ms"]))
        for label in METRICS:
            pair["none_" + label], pair["swpf_" + label] = original[label], row[label]
        pair["instruction_ratio"] = row["warp_instructions"] / original["warp_instructions"]
        pair["dram_bytes_estimate_ratio"] = row["dram_bytes_estimated"] / original["dram_bytes_estimated"]
        pairs.append(pair)
    write_csv(args.root / "ncu-paired.csv", pairs)
    print(f"Summarized {len(rows)} exact-launch reports; {len(pairs)} matched pairs.")


if __name__ == "__main__":
    main()
