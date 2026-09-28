#!/usr/bin/env python3
"""Select five exact-mangled, invocation-aware NCU points from an Nsys DB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


ROLES = ("p10", "p50", "p90", "peak", "tail")


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty set")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_kernels(database: Path) -> list[dict]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT k.start, k.end, k.contextId, k.streamId, k.deviceId,
                   k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                   k.gridId, mangled.value, demangled.value, short.value
              FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
              JOIN StringIds AS mangled ON mangled.id = k.mangledName
              JOIN StringIds AS demangled ON demangled.id = k.demangledName
              JOIN StringIds AS short ON short.id = k.shortName
             ORDER BY k.start
            """
        ).fetchall()
    finally:
        connection.close()

    kernels = []
    for row in rows:
        (
            start,
            end,
            context,
            stream,
            device,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            grid_id,
            mangled,
            demangled,
            short,
        ) = row
        grid_x = int(grid_x or 0)
        grid_y = int(grid_y or 0)
        grid_z = int(grid_z or 0)
        duration_ns = max(0, int(end) - int(start))
        kernels.append(
            {
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ns": duration_ns,
                "duration_ms": duration_ns / 1e6,
                "context_id": context,
                "stream_id": stream,
                "device_id": device,
                "grid_id": grid_id,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "grid_blocks": grid_x * grid_y * grid_z,
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "mangled_name": mangled,
                "demangled_name": demangled,
                "kernel_function": short,
            }
        )
    return kernels


def write_csv(path: Path, fields: list[str], rows: list[dict], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter=delimiter, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--near-empty-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 <= args.near_empty_fraction < 1.0:
        raise SystemExit("--near-empty-fraction must be in [0, 1)")

    kernels = load_kernels(args.database)
    if not kernels:
        raise SystemExit(f"no CUDA kernel activities found in {args.database}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for kernel in kernels:
        groups[kernel["mangled_name"]].append(kernel)
    ranked = []
    for mangled, items in groups.items():
        ranked.append(
            {
                "kernel_function": items[0]["kernel_function"],
                "demangled_name": items[0]["demangled_name"],
                "mangled_name": mangled,
                "invocation_count": len(items),
                "aggregate_gpu_ms": sum(item["duration_ms"] for item in items),
                "max_invocation_ms": max(item["duration_ms"] for item in items),
            }
        )
    ranked.sort(key=lambda item: item["aggregate_gpu_ms"], reverse=True)
    repeated = [item for item in ranked if item["invocation_count"] >= 2]
    if not repeated:
        raise SystemExit("no repeated exact kernel function found in the Nsys trace")

    dominant = repeated[0]
    if dominant["invocation_count"] < 5:
        raise SystemExit(
            f"dominant exact kernel has only {dominant['invocation_count']} invocations; "
            "recapture with enough repeated launches"
        )
    invocations = sorted(
        groups[dominant["mangled_name"]],
        key=lambda item: (item["grid_id"], item["start_ns"]),
    )
    for index, item in enumerate(invocations):
        item["invocation_index"] = index

    total_function_ms = sum(item["duration_ms"] for item in invocations)
    all_kernel_ms = sum(item["duration_ms"] for item in kernels)
    for item in invocations:
        item["kernel_total_gpu_ms"] = total_function_ms
        item["kernel_time_contribution_pct"] = (
            100.0 * item["duration_ms"] / total_function_ms if total_function_ms else 0.0
        )
        item["dominant_function_share_of_all_kernel_time_pct"] = (
            100.0 * total_function_ms / all_kernel_ms if all_kernel_ms else 0.0
        )
        item["trace_position_pct"] = 100.0 * (item["invocation_index"] + 1) / len(invocations)

    grid_values = [float(item["grid_blocks"]) for item in invocations]
    grid_p10 = quantile(grid_values, 0.10)
    grid_p90 = quantile(grid_values, 0.90)
    grid_varies = grid_p90 > max(1.0, grid_p10) * 1.20
    metric_key = "grid_blocks" if grid_varies else "duration_ms"
    metric_kind = "grid_blocks_proxy" if grid_varies else "duration_ms_proxy"
    metric_values = [float(item[metric_key]) for item in invocations]
    metric_p90 = quantile(metric_values, 0.90)
    if metric_key == "grid_blocks":
        near_empty_floor = max(1.0, math.ceil(args.near_empty_fraction * metric_p90))
    else:
        near_empty_floor = args.near_empty_fraction * metric_p90
    meaningful = [item for item in invocations if float(item[metric_key]) >= near_empty_floor]
    if len(meaningful) < 5:
        raise SystemExit(
            f"only {len(meaningful)} meaningful invocations remain after the "
            f"{args.near_empty_fraction:.1%} p90 cutoff"
        )
    meaningful_indices = {item["invocation_index"] for item in meaningful}
    for item in invocations:
        item["work_metric_kind"] = metric_kind
        item["work_metric_value"] = item[metric_key]
        item["near_empty_threshold"] = near_empty_floor
        item["meaningful_for_percentiles"] = int(item["invocation_index"] in meaningful_indices)
        item["iteration_index"] = ""
        item["frontier_or_edge_work"] = ""

    heavy_count = max(1, math.ceil(0.10 * len(meaningful)))
    heavy = sorted(
        meaningful,
        key=lambda item: (float(item[metric_key]), item["duration_ms"]),
        reverse=True,
    )[:heavy_count]
    heavy_time_ms = sum(item["duration_ms"] for item in heavy)
    heavy_fraction_pct = 100.0 * heavy_time_ms / total_function_ms if total_function_ms else 0.0

    used: set[int] = set()
    selected: list[dict] = []

    def add(role: str, item: dict, target: str) -> None:
        row = dict(item)
        row["role"] = role
        row["target_percentile_or_rule"] = target
        row["heavy_regime_gpu_time_fraction_pct"] = heavy_fraction_pct
        row["launch_skip"] = item["invocation_index"]
        row["ncu_kernel_filter"] = item["mangled_name"]
        row["nsys_database"] = str(args.database)
        selected.append(row)
        used.add(item["invocation_index"])

    peak = max(meaningful, key=lambda item: (float(item[metric_key]), item["duration_ms"]))
    add("peak", peak, "maximum work proxy; duration breaks ties")
    late_start = max(0, math.floor(0.80 * len(invocations)))
    late = [item for item in invocations[late_start:] if item["grid_blocks"] > 0]
    tail_pool = [
        item for item in late if item["invocation_index"] in meaningful_indices and item["invocation_index"] not in used
    ]
    if not tail_pool:
        tail_pool = [item for item in late if item["invocation_index"] not in used]
    if not tail_pool:
        tail_pool = [item for item in meaningful if item["invocation_index"] not in used]
    if not tail_pool:
        raise SystemExit("could not select a distinct tail invocation")
    tail = min(
        tail_pool,
        key=lambda item: (float(item[metric_key]), -item["duration_ms"], -item["invocation_index"]),
    )
    add("tail", tail, "smallest work proxy in final 20% of matching launches")

    meaningful_values = [float(item[metric_key]) for item in meaningful]
    meaningful_durations = [item["duration_ms"] for item in meaningful]
    for role, fraction in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        target = quantile(meaningful_values, fraction)
        duration_target = quantile(meaningful_durations, fraction)
        available = [item for item in meaningful if item["invocation_index"] not in used]
        if not available:
            raise SystemExit("could not select five distinct invocation indices")
        chosen = min(
            available,
            key=lambda item: (
                abs(float(item[metric_key]) - target),
                abs(item["duration_ms"] - duration_target),
                item["invocation_index"],
            ),
        )
        add(role, chosen, f"p{int(fraction * 100)}={target:.6g} {metric_kind}")
    selected.sort(key=lambda item: ROLES.index(item["role"]))
    if len({item["invocation_index"] for item in selected}) != 5:
        raise SystemExit("internal selection error: selected points are not distinct")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "kernel-rankings.csv",
        [
            "kernel_function",
            "demangled_name",
            "mangled_name",
            "invocation_count",
            "aggregate_gpu_ms",
            "max_invocation_ms",
        ],
        ranked,
    )
    invocation_fields = [
        "invocation_index",
        "start_ns",
        "end_ns",
        "duration_ns",
        "duration_ms",
        "context_id",
        "stream_id",
        "device_id",
        "grid_id",
        "grid_x",
        "grid_y",
        "grid_z",
        "grid_blocks",
        "block_x",
        "block_y",
        "block_z",
        "mangled_name",
        "demangled_name",
        "kernel_function",
        "kernel_total_gpu_ms",
        "kernel_time_contribution_pct",
        "dominant_function_share_of_all_kernel_time_pct",
        "trace_position_pct",
        "work_metric_kind",
        "work_metric_value",
        "near_empty_threshold",
        "meaningful_for_percentiles",
        "iteration_index",
        "frontier_or_edge_work",
    ]
    write_csv(args.out_dir / "dominant-invocations.csv", invocation_fields, invocations)
    selection_fields = [
        "role",
        "target_percentile_or_rule",
        "invocation_index",
        "launch_skip",
        "kernel_function",
        "demangled_name",
        "mangled_name",
        "ncu_kernel_filter",
        "duration_ms",
        "kernel_time_contribution_pct",
        "grid_x",
        "grid_y",
        "grid_z",
        "grid_blocks",
        "block_x",
        "block_y",
        "block_z",
        "work_metric_kind",
        "work_metric_value",
        "trace_position_pct",
        "heavy_regime_gpu_time_fraction_pct",
        "nsys_database",
    ]
    write_csv(args.out_dir / "selection.csv", selection_fields, selected)
    write_csv(args.out_dir / "selection.tsv", selection_fields, selected, delimiter="\t")

    summary = {
        "selector": "gunrock-v2/profiling/analyze_nsys_invocations.py",
        "algorithm": args.algorithm,
        "graph": args.graph,
        "nsys_database": str(args.database),
        "dominant_kernel_short_name": dominant["kernel_function"],
        "dominant_kernel_demangled_name": dominant["demangled_name"],
        "dominant_kernel_mangled_filter": dominant["mangled_name"],
        "kernel_filter_basis": "exact mangled name",
        "invocation_count": len(invocations),
        "dominant_kernel_aggregate_gpu_ms": total_function_ms,
        "all_kernel_aggregate_gpu_ms": all_kernel_ms,
        "dominant_function_share_of_all_kernel_time_pct": 100.0 * total_function_ms / all_kernel_ms if all_kernel_ms else 0.0,
        "work_metric_kind": metric_kind,
        "near_empty_fraction": args.near_empty_fraction,
        "near_empty_filter_threshold": near_empty_floor,
        "meaningful_invocation_count": len(meaningful),
        "excluded_near_empty_count": len(invocations) - len(meaningful),
        "heavy_regime_definition": "top decile of meaningful launches by work proxy (or duration proxy)",
        "heavy_regime_gpu_time_ms": heavy_time_ms,
        "heavy_regime_gpu_time_fraction_pct": heavy_fraction_pct,
        "selected_points": [
            {
                key: item[key]
                for key in (
                    "role",
                    "invocation_index",
                    "duration_ms",
                    "grid_x",
                    "grid_y",
                    "grid_z",
                    "grid_blocks",
                    "work_metric_value",
                    "kernel_time_contribution_pct",
                    "heavy_regime_gpu_time_fraction_pct",
                    "trace_position_pct",
                )
            }
            for item in selected
        ],
    }
    (args.out_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"{args.algorithm}/{args.graph}: dominant kernel={dominant['demangled_name']}",
        f"invocations={len(invocations)} meaningful={len(meaningful)} excluded_near_empty={len(invocations)-len(meaningful)}",
        f"aggregate_gpu_ms={total_function_ms:.6f} heavy_top_decile_ms={heavy_time_ms:.6f} heavy_fraction={heavy_fraction_pct:.2f}%",
        f"work_metric={metric_kind} near_empty_fraction={args.near_empty_fraction} threshold={near_empty_floor}",
        "role\tindex\tduration_ms\tgrid_blocks\tcontribution_pct\ttrace_position_pct",
    ]
    for item in selected:
        lines.append(
            f"{item['role']}\t{item['invocation_index']}\t{item['duration_ms']:.6f}\t"
            f"{item['grid_blocks']}\t{item['kernel_time_contribution_pct']:.4f}\t{item['trace_position_pct']:.2f}"
        )
    (args.out_dir / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
