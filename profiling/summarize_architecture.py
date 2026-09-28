#!/usr/bin/env python3
"""Build v2 architecture and v1.2 comparison summaries from recorded artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


MODES = ("natural", "cold")
ROLES = ("p10", "p50", "p90", "peak", "tail")
STEADY_STATE_GUIDANCE_NS = 20_000


def number(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value, digits=3):
    parsed = number(value)
    return "n/a" if parsed is None else f"{parsed:.{digits}f}"


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def native_stats(case_dir: Path) -> dict:
    path = case_dir / "native" / "native.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [float(x) for x in data.get("process_times", [])]
    if not samples:
        return {}
    return {
        "native_sample_count": len(samples),
        "native_mean_ms": statistics.mean(samples),
        "native_median_ms": statistics.median(samples),
        "native_stddev_ms": statistics.pstdev(samples),
        "native_min_ms": min(samples),
        "native_max_ms": max(samples),
        "native_reported_avg_ms": number(data.get("avg_process_time")),
        "native_gpu": data.get("gpuinfo", {}).get("name", ""),
        "native_cuda_runtime": data.get("gpuinfo", {}).get("runtime_version", ""),
    }


def algorithm_range_ms(case_dir: Path):
    """Read the v2 algorithm NVTX range from Nsys stats output."""
    path = case_dir / "nsys" / "run.log"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":algorithm" not in line:
            continue
        numbers = re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", line)
        if len(numbers) >= 2:
            return float(numbers[1]) / 1e6
    return None


def analysis_data(case_dir: Path) -> dict:
    path = case_dir / "invocations" / "analysis.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_reference(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_csv(path)
    return {(row.get("algorithm", ""), row.get("graph", "")): row for row in rows}


def correlation(pairs: list[tuple[float, float]]):
    if len(pairs) < 3:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in pairs) / math.sqrt(var_x * var_y)


def percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def case_record(case_dir: Path, reference: dict) -> tuple[dict, list[dict], dict]:
    algorithm = case_dir.parent.parent.name
    graph = case_dir.parent.name
    analysis = analysis_data(case_dir)
    ncu_rows = read_csv(case_dir / "summary" / "ncu-five-point.csv")
    ncu_by_role = {(row.get("cache_mode"), row.get("regime")): row for row in ncu_rows}
    native = native_stats(case_dir)
    dominant_ms = number(analysis.get("dominant_kernel_aggregate_gpu_ms"))
    all_kernel_ms = number(analysis.get("all_kernel_aggregate_gpu_ms"))
    algorithm_ms = algorithm_range_ms(case_dir)
    native_median = native.get("native_median_ms")
    ref_avg = number(reference.get("whole_app_avg_ms_from_native_run"))
    speedup = ref_avg / native_median if ref_avg is not None and native_median else None
    gain = 100.0 * (ref_avg - native_median) / ref_avg if ref_avg else None
    ref_dom = number(reference.get("dominant_kernel_aggregate_gpu_ms"))
    ref_share = number(reference.get("dominant_kernel_share_of_all_kernel_time_pct"))
    ref_all = ref_dom * 100.0 / ref_share if ref_dom is not None and ref_share else None

    record = {
        "algorithm": algorithm,
        "graph": graph,
        "case_dir": str(case_dir),
        "dominant_kernel_short_name": analysis.get("dominant_kernel_short_name", ""),
        "dominant_kernel_demangled_name": analysis.get("dominant_kernel_demangled_name", ""),
        "dominant_kernel_mangled_name": analysis.get("dominant_kernel_mangled_filter", ""),
        "kernel_identity_basis": analysis.get("kernel_filter_basis", "exact mangled name"),
        "invocation_count": analysis.get("invocation_count", ""),
        "meaningful_invocation_count": analysis.get("meaningful_invocation_count", ""),
        "dominant_kernel_sum_ms": dominant_ms,
        "all_kernel_sum_ms": all_kernel_ms,
        "dominant_kernel_share_of_all_kernel_time_pct": analysis.get("dominant_function_share_of_all_kernel_time_pct", ""),
        "algorithm_range_ms": algorithm_ms,
        "algorithm_kernel_gap_ms": algorithm_ms - all_kernel_ms if algorithm_ms is not None and all_kernel_ms is not None else None,
        "heavy_regime_gpu_time_ms": analysis.get("heavy_regime_gpu_time_ms", ""),
        "heavy_regime_gpu_time_fraction_pct": analysis.get("heavy_regime_gpu_time_fraction_pct", ""),
        **native,
        "ncu_verified_points": len(ncu_rows),
        "v1_2_native_avg_ms": ref_avg,
        "v1_2_dominant_kernel_sum_ms": ref_dom,
        "v1_2_dominant_kernel_share_of_all_kernel_time_pct": ref_share,
        "v1_2_all_kernel_sum_ms_derived": ref_all,
        "v1_2_algorithm_range_ms": "",
        "v2_vs_v1_2_speedup_median": speedup,
        "v2_vs_v1_2_gain_pct_median": gain,
        "comparison_primary_statistic": "v2 native median / v1.2 reported native average",
        "v1_2_reference_case_dir": reference.get("case_dir", ""),
    }
    for mode in MODES:
        for role in ROLES:
            row = ncu_by_role.get((mode, role), {})
            for metric in (
                "invocation_index", "nsys_duration_ms", "ncu_duration_ns", "nsys_kernel_time_contribution_pct",
                "kernel_duration_quality", "l1_hit_rate_pct", "l1_hit_rate_reported_pct", "l1_hit_rate_quality",
                "l2_hit_rate_pct", "l2_hit_rate_reported_pct", "l2_hit_rate_quality", "l2_miss_rate_pct",
                "l2_effective_bw_gbytes_per_s_est", "device_memory_bw_gbytes_per_s_est",
                "device_memory_throughput_pct_peak", "occupancy_pct_peak", "active_warps_per_cycle",
                "eligible_warps_per_cycle", "sm_throughput_pct_peak", "long_scoreboard_ratio",
            ):
                record[f"{mode}_{role}_{metric}"] = row.get(metric, "")

    peak = ncu_by_role.get(("natural", "peak"), {})
    tail = ncu_by_role.get(("natural", "tail"), {})
    peak_ns = number(peak.get("ncu_duration_ns"))
    tail_ns = number(tail.get("ncu_duration_ns"))
    record["peak_counter_quality"] = peak.get("kernel_duration_quality", "unavailable")
    record["peak_interpretation"] = "usable_at_least_20us" if peak_ns is not None and peak_ns >= STEADY_STATE_GUIDANCE_NS else "short_kernel_rates_inconclusive"
    record["peak_vs_tail_l2_hit_delta_pp"] = (
        number(peak.get("l2_hit_rate_pct")) - number(tail.get("l2_hit_rate_pct"))
        if number(peak.get("l2_hit_rate_pct")) is not None and number(tail.get("l2_hit_rate_pct")) is not None else ""
    )
    record["peak_vs_tail_dram_peak_delta_pp"] = (
        number(peak.get("device_memory_throughput_pct_peak")) - number(tail.get("device_memory_throughput_pct_peak"))
        if number(peak.get("device_memory_throughput_pct_peak")) is not None and number(tail.get("device_memory_throughput_pct_peak")) is not None else ""
    )
    record["natural_cold_l2_delta_peak_pp"] = (
        number(peak.get("l2_hit_rate_pct")) - number(ncu_by_role.get(("cold", "peak"), {}).get("l2_hit_rate_pct"))
        if number(peak.get("l2_hit_rate_pct")) is not None and number(ncu_by_role.get(("cold", "peak"), {}).get("l2_hit_rate_pct")) is not None else ""
    )
    record["natural_cold_l2_delta_p50_pp"] = (
        number(ncu_by_role.get(("natural", "p50"), {}).get("l2_hit_rate_pct")) - number(ncu_by_role.get(("cold", "p50"), {}).get("l2_hit_rate_pct"))
        if number(ncu_by_role.get(("natural", "p50"), {}).get("l2_hit_rate_pct")) is not None and number(ncu_by_role.get(("cold", "p50"), {}).get("l2_hit_rate_pct")) is not None else ""
    )
    return record, ncu_rows, analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", default="full-suite")
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="read-only reference result root configured by AGENTS.local.md",
    )
    args = parser.parse_args()

    group_root = args.root / args.group
    case_summary_paths = sorted(group_root.rglob("summary/ncu-five-point.csv"))
    if not case_summary_paths:
        raise SystemExit(f"no per-case NCU summaries under {group_root}")
    reference_path = args.reference_root / args.group / "summary" / "architecture-case-summary.csv"
    references = load_reference(reference_path)
    cases: list[dict] = []
    per_invocation: list[dict] = []
    case_aux: list[tuple[dict, dict, dict[str, dict]]] = []
    for summary_path in case_summary_paths:
        case_dir = summary_path.parent.parent
        algorithm = case_dir.parent.parent.name
        graph = case_dir.parent.name
        record, ncu_rows, analysis = case_record(case_dir, references.get((algorithm, graph), {}))
        cases.append(record)
        for row in ncu_rows:
            per_invocation.append({
                "algorithm": algorithm,
                "graph": graph,
                "dominant_kernel_phase": "algorithm-kernel",
                "algorithm_range_ms": record.get("algorithm_range_ms", ""),
                "all_kernel_sum_ms": record.get("all_kernel_sum_ms", ""),
                **row,
            })
        case_aux.append((record, analysis, {(row.get("cache_mode"), row.get("regime")): row for row in ncu_rows}))

    usable_natural = [
        row for record, _analysis, rows in case_aux
        for role in ROLES
        for row in [rows.get(("natural", role), {})]
        if number(row.get("ncu_duration_ns")) is not None
        and number(row.get("ncu_duration_ns")) >= STEADY_STATE_GUIDANCE_NS
    ]
    scoreboard_values = [number(row.get("long_scoreboard_ratio")) for row in usable_natural if number(row.get("long_scoreboard_ratio")) is not None]
    eligible_values = [number(row.get("eligible_warps_per_cycle")) for row in usable_natural if number(row.get("eligible_warps_per_cycle")) is not None]
    high_scoreboard = percentile(scoreboard_values, 0.75)
    low_eligible = percentile(eligible_values, 0.25)
    pooled_pairs = [
        (number(row.get("long_scoreboard_ratio")), number(row.get("eligible_warps_per_cycle")))
        for row in usable_natural
        if number(row.get("long_scoreboard_ratio")) is not None and number(row.get("eligible_warps_per_cycle")) is not None
    ]
    pooled_corr = correlation(pooled_pairs)

    for record, _analysis, rows in case_aux:
        peak = rows.get(("natural", "peak"), {})
        peak_ns = number(peak.get("ncu_duration_ns"))
        dram = number(peak.get("device_memory_throughput_pct_peak"))
        scoreboard = number(peak.get("long_scoreboard_ratio"))
        eligible = number(peak.get("eligible_warps_per_cycle"))
        if peak_ns is None or peak_ns < STEADY_STATE_GUIDANCE_NS:
            classification = "short-kernel / rates inconclusive"
        elif dram is not None and dram >= 70:
            classification = "DRAM-bandwidth-saturated"
        elif scoreboard is not None and eligible is not None and high_scoreboard is not None and low_eligible is not None and scoreboard >= high_scoreboard and eligible <= low_eligible:
            classification = "latency-leaning"
        elif dram is not None and dram < 50:
            classification = "substantial-bandwidth-headroom"
        else:
            classification = "mixed/other"
        record["architecture_classification"] = classification
        if classification == "DRAM-bandwidth-saturated":
            prefetch = "poor candidate"
        elif classification == "latency-leaning" and dram is not None and dram < 70:
            prefetch = "promising/early-producer candidate"
        elif classification == "substantial-bandwidth-headroom":
            prefetch = "mixed/insufficient evidence"
        else:
            prefetch = "mixed/insufficient evidence"
        record["prefetch_screening_label"] = prefetch
        case_pairs = [
            (number(rows.get(("natural", role), {}).get("long_scoreboard_ratio")), number(rows.get(("natural", role), {}).get("eligible_warps_per_cycle")))
            for role in ROLES
            if number(rows.get(("natural", role), {}).get("ncu_duration_ns")) is not None
            and number(rows.get(("natural", role), {}).get("ncu_duration_ns")) >= STEADY_STATE_GUIDANCE_NS
            and number(rows.get(("natural", role), {}).get("long_scoreboard_ratio")) is not None
            and number(rows.get(("natural", role), {}).get("eligible_warps_per_cycle")) is not None
        ]
        record["natural_scoreboard_eligible_correlation"] = correlation(case_pairs)

    summary_dir = group_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_rows(summary_dir / "architecture-case-summary.csv", cases)
    write_rows(summary_dir / "architecture-per-invocation.csv", per_invocation)

    comparison_rows = []
    for record in cases:
        comparison_rows.append({
            key: record.get(key, "")
            for key in (
                "algorithm", "graph", "v1_2_native_avg_ms", "native_mean_ms", "native_median_ms", "native_stddev_ms",
                "native_min_ms", "native_max_ms", "v2_vs_v1_2_speedup_median", "v2_vs_v1_2_gain_pct_median",
                "algorithm_range_ms", "all_kernel_sum_ms", "algorithm_kernel_gap_ms", "dominant_kernel_sum_ms",
                "dominant_kernel_share_of_all_kernel_time_pct", "v1_2_dominant_kernel_sum_ms", "v1_2_all_kernel_sum_ms_derived",
                "v1_2_algorithm_range_ms", "architecture_classification",
            )
        })
    write_rows(summary_dir / "comparison-summary.csv", comparison_rows)
    write_rows(group_root / "comparison-summary.csv", comparison_rows)

    lines = [
        "# Gunrock v2 architecture and comparison findings",
        "",
        f"Cases summarized: {len(cases)}. Per-case NCU tables are under each case `summary/`; complete tables are `{summary_dir / 'architecture-case-summary.csv'}` and `{summary_dir / 'architecture-per-invocation.csv'}`.",
        "",
        "Native timing is the end-to-end metric. It uses ten unprofiled example runs and reports the median as the primary v2 statistic, with mean, population standard deviation, minimum, and maximum retained. The v1.2 column is its recorded native average from the read-only reference result.",
        "",
        "The Nsys algorithm range is the v2 `:algorithm` NVTX range and includes device synchronization, kernel launches, and framework gaps inside the range. The Nsys all-kernel sum is the sum of CUDA kernel activity durations from the exported SQLite trace; it is a diagnostic, not an end-to-end replacement. The v1.2 reference traces do not contain the v2 `algorithm` NVTX marker, so v1.2 algorithm-range values are intentionally reported as unavailable.",
        "",
        "NCU rows are one invocation at a time in natural-cache and cold-kernel replay. No unweighted average across p10/p50/p90/peak/tail or across cache modes is used. Rate-based interpretation excludes launches under 20 us or out-of-range rates; raw counters remain in the CSV files.",
        "",
        "## End-to-end comparison and Nsys structure",
        "",
        "| Case | v1.2 avg (ms) | v2 median (ms) | v2 mean ± stddev (ms) | v2 min–max (ms) | v1.2/v2 median | gain | v2 algorithm range (ms) | v2 all-kernel sum (ms) | range−kernel gap (ms) | dominant kernel share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in cases:
        lines.append(
            f"| {record['algorithm']}/{record['graph']} | {fmt(record.get('v1_2_native_avg_ms'))} | {fmt(record.get('native_median_ms'))} | {fmt(record.get('native_mean_ms'))} ± {fmt(record.get('native_stddev_ms'))} | {fmt(record.get('native_min_ms'))}–{fmt(record.get('native_max_ms'))} | {fmt(record.get('v2_vs_v1_2_speedup_median'), 4)}x | {fmt(record.get('v2_vs_v1_2_gain_pct_median'), 2)}% | {fmt(record.get('algorithm_range_ms'))} | {fmt(record.get('all_kernel_sum_ms'))} | {fmt(record.get('algorithm_kernel_gap_ms'))} | {fmt(record.get('dominant_kernel_share_of_all_kernel_time_pct'), 2)}% |"
        )

    lines.extend([
        "",
        "## Selected-kernel architecture screening",
        "",
        "| Case | dominant exact-kernel launches | meaningful launches | NCU peak duration | peak DRAM (% peak) | peak long-scoreboard | peak eligible warps | classification | prefetch screening |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for record in cases:
        lines.append(
            f"| {record['algorithm']}/{record['graph']} | {record.get('invocation_count','')} | {record.get('meaningful_invocation_count','')} | {fmt(record.get('natural_peak_ncu_duration_ns'), 0)} ns | {fmt(record.get('natural_peak_device_memory_throughput_pct_peak'), 2)} | {fmt(record.get('natural_peak_long_scoreboard_ratio'), 2)} | {fmt(record.get('natural_peak_eligible_warps_per_cycle'), 2)} | {record.get('architecture_classification','')} | {record.get('prefetch_screening_label','')} |"
        )

    lines.extend([
        "",
        "## Natural versus cold cache observations",
        "",
        "The following differences are natural-cache L2 hit rate minus cold-replay L2 hit rate for the selected p50 and peak launches. Blank values mean the corresponding rate was not interpretable under the duration/range policy.",
        "",
        "| Case | p50 L2 delta (percentage points) | peak L2 delta (percentage points) |",
        "|---|---:|---:|",
    ])
    for record in cases:
        lines.append(f"| {record['algorithm']}/{record['graph']} | {fmt(record.get('natural_cold_l2_delta_p50_pp'), 2)} | {fmt(record.get('natural_cold_l2_delta_peak_pp'), 2)} |")

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        f"The descriptive Pearson correlation between long-scoreboard ratio and eligible warps over {len(pooled_pairs)} usable natural-cache points is {fmt(pooled_corr, 3)}; it is observational and mixes cases, not a causal claim.",
        "",
        "RoadNet-CA selected launches are short in the NCU reports, so their multi-pass hit-rate and stall-rate values are retained but should not be used as strong architecture conclusions. The longer soc-orkut points provide more interpretable per-invocation counters. NCU replay overhead and cache/replay policy can change absolute timing; use Nsys and native timing for performance claims.",
        "",
        "The version comparison also crosses the v1.2 CUDA/toolchain/runtime environment and the v2 CUDA 12.6 environment. The table reports observed results under the recorded environments; it does not attribute every difference to the framework change alone.",
        "",
    ])
    (group_root / "architecture-findings.md").write_text("\n".join(lines), encoding="utf-8")
    (group_root / "comparison-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {summary_dir / 'architecture-case-summary.csv'} and {group_root / 'architecture-findings.md'}")


if __name__ == "__main__":
    main()
