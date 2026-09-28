#!/usr/bin/env python3
"""Verify one Ncu report against its Nsys-selected exact invocation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED_SECTIONS = {
    "GPU Speed Of Light Throughput": "SpeedOfLight",
    "Memory Workload Analysis": "MemoryWorkloadAnalysis",
    "Warp State Statistics": "WarpStateStats",
    "Scheduler Statistics": "SchedulerStats",
    "Launch Statistics": "LaunchStats",
}


def numeric(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selection_row(path: Path, role: str) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("role") == role:
                return row
    raise SystemExit(f"selection role {role!r} not found in {path}")


def invocation_row(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["invocation_index"]) == index:
                return row
    raise SystemExit(f"invocation index {index} not found in {path}")


def ncu_data_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = []
        for row in reader:
            if numeric(row.get("ID", "")) is not None and row.get("Kernel Name"):
                rows.append(row)
    return headers, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--ncu-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selected = selection_row(args.selection, args.role)
    expected_index = int(selected["invocation_index"])
    expected = invocation_row(args.invocations, expected_index)
    command = args.command.read_text(encoding="utf-8")
    details = args.details.read_text(encoding="utf-8", errors="replace")
    headers, rows = ncu_data_rows(args.raw)
    errors: list[str] = []
    if not args.ncu_report.is_file() or args.ncu_report.stat().st_size == 0:
        errors.append("missing or empty NCU report")
    if selected["launch_skip"] != str(expected_index):
        errors.append("selection launch_skip does not equal invocation index")
    if "--kernel-name-base mangled" not in command:
        errors.append("command does not use --kernel-name-base mangled")
    if selected["ncu_kernel_filter"] not in command:
        errors.append("command does not contain the exact mangled kernel filter")
    if f"--launch-skip={selected['launch_skip']}" not in command and f"--launch-skip {selected['launch_skip']}" not in command:
        errors.append("command launch skip does not match selection")
    if "--launch-count=1" not in command and "--launch-count 1" not in command:
        errors.append("command does not set --launch-count=1")
    if "--profile" not in command:
        errors.append("profile command does not contain --profile")
    if len(rows) != 1:
        errors.append(f"expected one NCU data row, found {len(rows)}")

    raw_row = rows[0] if rows else {}
    if raw_row:
        if raw_row.get("Kernel Name") != selected["ncu_kernel_filter"]:
            errors.append("NCU kernel name is not the exact selected mangled name")
        expected_grid = f"({expected['grid_x']}, {expected['grid_y']}, {expected['grid_z']})"
        expected_block = f"({expected['block_x']}, {expected['block_y']}, {expected['block_z']})"
        if raw_row.get("Grid Size") != expected_grid:
            errors.append(f"grid mismatch: expected {expected_grid}, got {raw_row.get('Grid Size')}")
        if raw_row.get("Block Size") != expected_block:
            errors.append(f"block mismatch: expected {expected_block}, got {raw_row.get('Block Size')}")

    missing_sections = [name for name in REQUIRED_SECTIONS if f"Section: {name}" not in details]
    if missing_sections:
        errors.append("missing required NCU sections: " + ", ".join(missing_sections))

    duration_ns = numeric(raw_row.get("gpu__time_duration.sum", ""))
    quality_flags: list[str] = []
    if duration_ns is not None and duration_ns < 20_000:
        quality_flags.append("under_20us_multipass_sensitive")
    out_of_range_fields = []
    for field, value in raw_row.items():
        if not field.endswith(".pct"):
            continue
        parsed = numeric(value)
        if parsed is not None and (parsed < 0.0 or parsed > 100.0):
            out_of_range_fields.append(field)
    if out_of_range_fields:
        quality_flags.append("out_of_range_rate")

    result = {
        "passed": not errors,
        "role": args.role,
        "target_percentile_or_rule": selected["target_percentile_or_rule"],
        "invocation_index": expected_index,
        "launch_skip": int(selected["launch_skip"]),
        "kernel_function": selected["kernel_function"],
        "exact_mangled_kernel": selected["ncu_kernel_filter"],
        "ncu_kernel_name": raw_row.get("Kernel Name", ""),
        "nsys_duration_ms": numeric(selected.get("duration_ms", "")),
        "kernel_time_contribution_pct": numeric(selected.get("kernel_time_contribution_pct", "")),
        "heavy_regime_gpu_time_fraction_pct": numeric(
            selected.get("heavy_regime_gpu_time_fraction_pct", "")
        ),
        "work_metric_kind": selected.get("work_metric_kind", ""),
        "work_metric_value": numeric(selected.get("work_metric_value", "")),
        "grid": {
            "nsys": [int(expected["grid_x"]), int(expected["grid_y"]), int(expected["grid_z"])]
        },
        "block": {
            "nsys": [int(expected["block_x"]), int(expected["block_y"]), int(expected["block_z"])]
        },
        "ncu_report": str(args.ncu_report),
        "nsys_report": str(args.ncu_report.parent.parent.parent.parent / "nsys" / "report.nsys-rep"),
        "ncu_data_rows": len(rows),
        "raw_headers": headers,
        "raw_values": raw_row,
        "duration_ns": duration_ns,
        "quality_flags": quality_flags,
        "out_of_range_rate_fields": out_of_range_fields,
        "required_sections": list(REQUIRED_SECTIONS.values()),
        "missing_sections": missing_sections,
        "errors": errors,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("NCU verification failed: " + "; ".join(errors))
    print(
        f"PASS {args.role}: launch_skip={expected_index} duration_ns={duration_ns} "
        f"quality={','.join(quality_flags) or 'in_range_at_least_20us'}"
    )


if __name__ == "__main__":
    main()
