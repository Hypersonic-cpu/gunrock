#!/usr/bin/env python3
"""Compare all native prefetch variants with none and recheck timing anomalies."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import yaml

from run_native_suite import NativeRunner, Suite


def load_rows(root):
    with (root / "native-summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["status"] != "OK" or int(row["samples"]) != 10 for row in rows):
        raise RuntimeError("All cases must succeed and retain 10 native samples")
    return rows


def key(row):
    return row["benchmark"], row["graph"], int(row["source"])


def variant(row):
    return row["swpf"], row["swpf_target"], int(row["swpf_distance"])


def summarize(root, resource_file):
    rows = load_rows(root)
    baselines = {key(row): float(row["gpu_median_ms"])
                 for row in rows if row["swpf"] == "none"}
    anomalies = []
    for row in rows:
        median = float(row["gpu_median_ms"])
        mean = float(row["gpu_mean_ms"])
        if not (median > 0 and mean > 0):
            raise RuntimeError(f"Invalid native timing: {row}")
        cv = float(row["gpu_stddev_ms"]) / mean
        row["speedup_over_none"] = baselines[key(row)] / median
        row["coefficient_of_variation"] = cv
        row["timing_flag"] = ""
        if cv > 0.10 or float(row["gpu_max_ms"]) > 1.5 * median:
            row["timing_flag"] = "high timing variation; retain all samples and recheck"
            anomalies.append(row)
    with (root / "swpf-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Software prefetch native results", "",
             "Each case retains all 10 program-reported CUDA-event algorithm timings; "
             "speedup is baseline median / variant median. Source is 13331. "
             "No Nsight/hardware counter collection was used.", "",
             "| App | Graph | None median (ms) | Best GP configuration | GP speedup | Best SPP configuration | SPP speedup |",
             "|---|---|---:|---|---:|---|---:|"]
    best_rows = []
    for case_key, baseline in baselines.items():
        choices = [row for row in rows if key(row) == case_key]
        gp = min((row for row in choices if row["swpf"] == "gp"),
                 key=lambda row: float(row["gpu_median_ms"]))
        spp = min((row for row in choices if row["swpf"] == "spp"),
                  key=lambda row: float(row["gpu_median_ms"]))
        label = lambda row: f"{row['swpf_target']}, d={row['swpf_distance']}"
        lines.append(f"| {case_key[0]} | {case_key[1]} | {baseline:.4f} | "
                     f"{label(gp)} | {gp['speedup_over_none']:.3f}x | "
                     f"{label(spp)} | {spp['speedup_over_none']:.3f}x |")
        best_rows.append(dict(app=case_key[0], graph=case_key[1],
                              baseline_median_ms=baseline, best_gp=label(gp),
                              gp_speedup=gp["speedup_over_none"], best_spp=label(spp),
                              spp_speedup=spp["speedup_over_none"]))
    with (root / "best-variants.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)
    lines += ["", f"Timing anomalies flagged: {len(anomalies)} / {len(rows)}. "
              "Raw samples are retained in each native.json; original results are never replaced by a filtered statistic.",
              "", "Algorithm timings include merge-path, scan/filter kernels, and host scheduling inside enact. "
              "They do not isolate a main kernel's device execution time.",
              "", "Register/shared/local memory and theoretical occupancy are recorded in "
              f"`{resource_file}`. Theoretical occupancy is a CUDA driver resource estimate, "
              "not achieved occupancy. Cache hit rates, DRAM bytes, long-scoreboard stalls, "
              "and eligible warps/cycle were not measured."]
    (root / "motivation-summary.md").write_text("\n".join(lines) + "\n")
    print(f"Compared {len(rows)} cases; {len(anomalies)} timing anomalies flagged.", flush=True)
    return anomalies


def recheck(root, anomalies):
    if not anomalies:
        return
    metadata = yaml.safe_load((root / "environment.yaml").read_text())
    suite = Suite.load(root / "suite-config.yaml")
    wanted = {(key(row), variant(row)) for row in anomalies}
    # Include none for each affected app/graph to identify baseline noise or drift.
    affected = {key(row) for row in anomalies}
    selected = []
    for app in suite.applications:
        cases = tuple(case for case in app.cases if
                      ((app.name, case.graph, case.source),
                       (case.swpf, case.swpf_target, case.swpf_distance)) in wanted or
                      (case.swpf == "none" and
                       (app.name, case.graph, case.source) in affected))
        if cases:
            selected.append(replace(app, cases=cases))
    suite = replace(suite, applications=tuple(selected))
    runner = NativeRunner(suite, root / "anomaly-recheck",
                          Path(metadata["data_root"]), Path(metadata["build_bin"]),
                          Path(metadata["cuda_home"]), metadata["cuda_visible_devices"])
    if runner.run() != 0:
        raise RuntimeError("Anomaly recheck failed")
    comparisons = []
    second = load_rows(root / "anomaly-recheck")
    originals = {(key(row), variant(row)): row for row in load_rows(root)}
    for row in second:
        old = originals[(key(row), variant(row))]
        comparisons.append(dict(app=row["benchmark"], graph=row["graph"],
                                swpf=row["swpf"], target=row["swpf_target"],
                                distance=row["swpf_distance"],
                                original_median_ms=old["gpu_median_ms"],
                                recheck_median_ms=row["gpu_median_ms"],
                                median_ratio=float(row["gpu_median_ms"]) / float(old["gpu_median_ms"]),
                                recheck_cv=float(row["gpu_stddev_ms"]) / float(row["gpu_mean_ms"])))
    with (root / "anomaly-investigation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (root / "motivation-summary.md").open("a") as stream:
        stream.write(f"\nRechecked {len(second)} cases, including their none baselines, "
                     "with 10 fresh program-reported samples each. Compare "
                     "`anomaly-investigation.csv` and `anomaly-recheck/native-summary.csv`.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    anomalies = summarize(args.native_root.resolve(), args.resources.resolve())
    if args.recheck:
        recheck(args.native_root.resolve(), anomalies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
