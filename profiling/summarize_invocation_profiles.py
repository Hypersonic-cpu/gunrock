#!/usr/bin/env python3
"""Summarize verified per-invocation NCU reports without cross-launch averaging."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MODES = ("natural", "cold")
ROLES = ("p10", "p50", "p90", "peak", "tail")
STEADY_STATE_GUIDANCE_NS = 20_000


def read_ncu_table(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"empty NCU CSV: {path}")
        units_row = next(reader, None)
        units = dict(zip(headers, units_row or []))
        for values in reader:
            if len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            row = dict(zip(headers, values))
            if row.get("ID", "").strip() and row.get("Kernel Name", "").strip():
                return row, units
    raise ValueError(f"no NCU data row in {path}")


def number(value):
    if value is None or value == "":
        return None
    try:
        result = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def value(row: dict, *keys: str):
    for key in keys:
        parsed = number(row.get(key, ""))
        if parsed is not None:
            return parsed
    return ""


def rate_quality(rate, duration_ns):
    parsed = number(rate)
    if parsed is None:
        return "unavailable"
    out_of_range = parsed < 0.0 or parsed > 100.0
    short = number(duration_ns) is None or number(duration_ns) < STEADY_STATE_GUIDANCE_NS
    if out_of_range and short:
        return "out_of_range_and_under_20us"
    if out_of_range:
        return "out_of_range"
    if short:
        return "under_20us_multipass_sensitive"
    return "in_range_at_least_20us"


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(case_dir: Path) -> dict[tuple[str, str], dict]:
    path = case_dir / "invocations" / "selection.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {(row["role"], row["kernel_function"]): row for row in csv.DictReader(stream)}


def invocation_rows(case_dir: Path) -> dict[int, dict]:
    path = case_dir / "invocations" / "dominant-invocations.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {int(row["invocation_index"]): row for row in csv.DictReader(stream)}


def context(metadata: dict, selected: dict, invocation: dict, raw: dict, units: dict,
            mode: str, role: str, point_dir: Path, case_dir: Path) -> dict:
    duration_ns = value(raw, "gpu__time_duration.sum")
    nsys_duration = number(metadata.get("nsys_duration_ms", selected.get("duration_ms", "")))
    l1_hit_reported = value(raw, "l1tex__t_sector_hit_rate.pct")
    l2_hit_reported = value(raw, "lts__t_sector_hit_rate.pct")
    l1_quality = rate_quality(l1_hit_reported, duration_ns)
    l2_quality = rate_quality(l2_hit_reported, duration_ns)
    l1_usable = l1_quality == "in_range_at_least_20us"
    l2_usable = l2_quality == "in_range_at_least_20us"

    l1_total = value(raw, "l1tex__t_sectors.sum")
    l1_hit = value(raw, "l1tex__t_sectors_lookup_hit.sum")
    l1_miss = value(raw, "l1tex__t_sectors_lookup_miss.sum")
    l1_miss_rate = value(raw, "l1tex__t_sector_miss_rate.pct")
    if l1_miss_rate == "" and isinstance(l1_total, (int, float)) and l1_total > 0 and isinstance(l1_miss, (int, float)):
        l1_miss_rate = 100.0 * l1_miss / l1_total

    l2_miss_rate = value(raw, "lts__t_sector_miss_rate.pct")
    if l2_miss_rate == "" and l2_usable and isinstance(l2_hit_reported, (int, float)):
        l2_miss_rate = 100.0 - l2_hit_reported
    if not l1_usable:
        l1_miss_rate = ""
    if not l2_usable:
        l2_miss_rate = ""

    l2_sectors = value(raw, "lts__d_sectors.sum", "lts__t_sectors.sum")
    dram_bytes = value(raw, "dram__bytes.sum")
    dram_bps = value(raw, "dram__bytes.sum.per_second")
    l2_bps = (
        l2_sectors * 32.0 / (duration_ns * 1e-9)
        if isinstance(l2_sectors, (int, float)) and isinstance(duration_ns, (int, float)) and duration_ns > 0
        else ""
    )
    if dram_bps == "" and isinstance(dram_bytes, (int, float)) and isinstance(duration_ns, (int, float)) and duration_ns > 0:
        dram_bps = dram_bytes / (duration_ns * 1e-9)

    grid = [number(invocation.get(key, "")) for key in ("grid_x", "grid_y", "grid_z")]
    block = [number(invocation.get(key, "")) for key in ("block_x", "block_y", "block_z")]
    grid_text = "x".join(str(int(x)) if x is not None and x.is_integer() else str(x) for x in grid if x is not None)
    block_text = "x".join(str(int(x)) if x is not None and x.is_integer() else str(x) for x in block if x is not None)
    metadata_grid = metadata.get("grid", {}).get("nsys", [])
    metadata_block = metadata.get("block", {}).get("nsys", [])
    if not grid_text and metadata_grid:
        grid_text = "x".join(str(x) for x in metadata_grid)
    if not block_text and metadata_block:
        block_text = "x".join(str(x) for x in metadata_block)

    return {
        "cache_mode": mode,
        "regime": role,
        "target_percentile_or_rule": metadata.get("target_percentile_or_rule", selected.get("target_percentile_or_rule", "")),
        "invocation_index": metadata.get("invocation_index", selected.get("invocation_index", "")),
        "kernel_function": metadata.get("kernel_function", selected.get("kernel_function", "")),
        "kernel_name": raw.get("Kernel Name", metadata.get("ncu_kernel_name", "")),
        "nsys_duration_ms": nsys_duration,
        "ncu_duration_ns": duration_ns,
        "nsys_kernel_time_contribution_pct": number(metadata.get("kernel_time_contribution_pct", selected.get("kernel_time_contribution_pct", ""))),
        "heavy_regime_gpu_time_fraction_pct": number(metadata.get("heavy_regime_gpu_time_fraction_pct", selected.get("heavy_regime_gpu_time_fraction_pct", ""))),
        "work_metric_kind": metadata.get("work_metric_kind", selected.get("work_metric_kind", "")),
        "work_metric_value": number(metadata.get("work_metric_value", selected.get("work_metric_value", ""))),
        "grid": grid_text,
        "block": block_text,
        "grid_blocks": number(invocation.get("grid_blocks", "")),
        "l1_sector_count": l1_total,
        "l1_hit_sectors": l1_hit,
        "l1_miss_sectors": l1_miss,
        "l1_hit_rate_pct": l1_hit_reported if l1_usable else "",
        "l1_hit_rate_reported_pct": l1_hit_reported,
        "l1_hit_rate_quality": l1_quality,
        "l1_miss_rate_pct": l1_miss_rate,
        "l2_request_count": value(raw, "lts__t_requests.sum"),
        "l2_hit_requests": value(raw, "lts__t_requests_aperture_device_lookup_hit.sum"),
        "l2_miss_requests": value(raw, "lts__t_requests_aperture_device_lookup_miss.sum"),
        "l2_data_sectors": l2_sectors,
        "l2_hit_rate_pct": l2_hit_reported if l2_usable else "",
        "l2_hit_rate_reported_pct": l2_hit_reported,
        "l2_hit_rate_quality": l2_quality,
        "l2_miss_rate_pct": l2_miss_rate,
        "l2_effective_bw_bytes_per_s_est": l2_bps,
        "l2_effective_bw_gbytes_per_s_est": l2_bps / 1e9 if isinstance(l2_bps, (int, float)) else "",
        "l2_throughput_pct_peak": value(raw, "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        "device_memory_bytes": dram_bytes,
        "device_memory_read_bytes": value(raw, "dram__bytes_read.sum"),
        "device_memory_write_bytes": value(raw, "dram__bytes_write.sum"),
        "device_memory_bw_bytes_per_s_est": dram_bps,
        "device_memory_bw_gbytes_per_s_est": dram_bps / 1e9 if isinstance(dram_bps, (int, float)) else "",
        "device_memory_throughput_pct_peak": value(raw, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
        "active_warps_per_cycle": value(raw, "smsp__warps_active.avg.per_cycle_active"),
        "eligible_warps_per_cycle": value(raw, "smsp__warps_eligible.avg.per_cycle_active"),
        "occupancy_pct_peak": value(raw, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "eligible_warps_pct_peak": value(raw, "smsp__warps_eligible.avg.pct_of_peak_sustained_elapsed", "smsp__warps_eligible.avg.pct_of_peak_sustained_active"),
        "issue_active_pct_peak": value(raw, "sm__issue_active.avg.pct_of_peak_sustained_elapsed"),
        "sm_throughput_pct_peak": value(raw, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_instructions_pct_peak": value(raw, "sm__inst_executed.avg.pct_of_peak_sustained_elapsed"),
        "long_scoreboard_ratio": value(raw, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
        "long_scoreboard_unit": units.get("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", ""),
        "launch_block_size": value(raw, "launch__block_size"),
        "launch_grid_size": value(raw, "launch__grid_size"),
        "registers_per_thread": value(raw, "launch__registers_per_thread"),
        "shared_memory_per_block": value(raw, "launch__shared_mem_per_block"),
        "occupancy_per_block_size": value(raw, "launch__occupancy_per_block_size"),
        "occupancy_per_register_count": value(raw, "launch__occupancy_per_register_count"),
        "occupancy_per_shared_memory": value(raw, "launch__occupancy_per_shared_mem_size"),
        "occupancy_limit_warps": value(raw, "launch__occupancy_limit_warps"),
        "waves_per_sm": value(raw, "launch__waves_per_multiprocessor"),
        "ncu_report": metadata.get("ncu_report", str(point_dir / "report.ncu-rep")),
        "nsys_report": metadata.get("nsys_report", str(case_dir / "nsys" / "report.nsys-rep")),
        "rate_metric_guidance": "Raw rates are retained. Interpreted hit/miss rates and rate-based classifications exclude out-of-range values and launches under 20 us because multipass measurements can be sensitive.",
        "kernel_duration_quality": "under_20us_multipass_sensitive" if duration_ns is None or duration_ns < STEADY_STATE_GUIDANCE_NS else "at_least_20us_guidance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    case_dir = args.case_dir
    summary_dir = case_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    selections = selected_rows(case_dir)
    invocations = invocation_rows(case_dir)
    point_rows: list[dict] = []
    stall_rows: list[dict] = []

    for mode in MODES:
        mode_dir = case_dir / "ncu" / mode
        for role in ROLES:
            point_dir = mode_dir / role
            metadata_path = point_dir / "verification.json"
            raw_path = point_dir / "metrics-raw.csv"
            if not metadata_path.is_file() or not raw_path.is_file():
                raise SystemExit(f"missing NCU artifact for {mode}/{role} in {case_dir}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("passed", metadata.get("verified", False)):
                raise SystemExit(f"unverified NCU point: {metadata_path}")
            raw, units = read_ncu_table(raw_path)
            kernel_name = metadata.get("ncu_kernel_name", raw.get("Kernel Name", ""))
            if raw.get("Kernel Name", "") != kernel_name:
                raise SystemExit(f"NCU row changed after verification: {raw_path}")
            selected = selections.get((role, metadata.get("kernel_function", "")), {})
            index = int(metadata.get("invocation_index", selected.get("invocation_index", -1)))
            invocation = invocations.get(index, {})
            row = context(metadata, selected, invocation, raw, units, mode, role, point_dir, case_dir)
            point_rows.append(row)
            stall = {
                "cache_mode": mode,
                "regime": role,
                "invocation_index": row["invocation_index"],
                "kernel_function": row["kernel_function"],
            }
            for name in raw:
                if name.startswith("smsp__average_warps_issue_stalled_"):
                    stall[name] = value(raw, name)
            stall_rows.append(stall)

    point_rows.sort(key=lambda row: (MODES.index(row["cache_mode"]), ROLES.index(row["regime"])))
    stall_rows.sort(key=lambda row: (MODES.index(row["cache_mode"]), ROLES.index(row["regime"])))
    context_fields = [
        "cache_mode", "regime", "target_percentile_or_rule", "invocation_index",
        "kernel_function", "kernel_name", "nsys_duration_ms", "ncu_duration_ns",
        "nsys_kernel_time_contribution_pct", "heavy_regime_gpu_time_fraction_pct",
        "work_metric_kind", "work_metric_value", "grid", "block", "ncu_report", "nsys_report",
    ]
    write_rows(summary_dir / "ncu-five-point.csv", point_rows)
    memory_fields = context_fields + [key for key in point_rows[0] if key.startswith(("l1_", "l2_", "device_memory_"))]
    scheduler_fields = context_fields + [
        "active_warps_per_cycle", "eligible_warps_per_cycle", "occupancy_pct_peak",
        "eligible_warps_pct_peak", "issue_active_pct_peak", "sm_throughput_pct_peak",
        "sm_instructions_pct_peak", "long_scoreboard_ratio", "long_scoreboard_unit",
        "kernel_duration_quality", "rate_metric_guidance",
    ]
    launch_fields = context_fields + [
        "launch_block_size", "launch_grid_size", "registers_per_thread", "shared_memory_per_block",
        "occupancy_per_block_size", "occupancy_per_register_count", "occupancy_per_shared_memory",
        "occupancy_limit_warps", "waves_per_sm", "kernel_duration_quality",
    ]
    write_rows(summary_dir / "ncu-memory.csv", [{key: row.get(key, "") for key in memory_fields} for row in point_rows])
    write_rows(summary_dir / "ncu-scheduler.csv", [{key: row.get(key, "") for key in scheduler_fields} for row in point_rows])
    write_rows(summary_dir / "ncu-launch.csv", [{key: row.get(key, "") for key in launch_fields} for row in point_rows])
    write_rows(summary_dir / "ncu-stalls.csv", stall_rows)
    write_rows(summary_dir / "ncu-stall-breakdown.csv", stall_rows)

    lines = [
        f"Case: {case_dir}",
        "Per-invocation observations; no unweighted average across regimes or cache modes.",
        "The heavy-regime percentage is the Nsys GPU-time fraction for the top decile of meaningful invocations.",
        "Raw hit rates are retained; interpreted rates are blank for out-of-range or sub-20-us launches.",
        "On the A40, device-memory throughput is DRAM/GDDR6, not HBM.",
        "mode\tregime\tindex\tNsys_ms\tNCU_ns\tL1_hit_pct\tL1_quality\tL2_hit_pct\tL2_quality\tDRAM_GBps\tDRAM_peak_pct\toccupancy_pct\teligible_warps\tlong_scoreboard\theavy_time_pct",
    ]
    for row in point_rows:
        lines.append("\t".join(str(row.get(key, "")) for key in (
            "cache_mode", "regime", "invocation_index", "nsys_duration_ms", "ncu_duration_ns",
            "l1_hit_rate_pct", "l1_hit_rate_quality", "l2_hit_rate_pct", "l2_hit_rate_quality",
            "device_memory_bw_gbytes_per_s_est", "device_memory_throughput_pct_peak", "occupancy_pct_peak",
            "eligible_warps_per_cycle", "long_scoreboard_ratio", "heavy_regime_gpu_time_fraction_pct",
        )))
    (summary_dir / "ncu-five-point.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized {len(point_rows)} verified NCU points in {summary_dir}")


if __name__ == "__main__":
    main()
