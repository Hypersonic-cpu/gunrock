#!/usr/bin/env python3
"""Map requested BFS iteration positions to global Nsight kernel launch IDs."""

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml


def resolve_position(value, total):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"unsupported iteration position: {value!r}")
    if value == "N-15":
        return total - 15
    match = re.fullmatch(r"middle(?:([+-]\d+))?", value)
    if match:
        return total // 2 + int(match.group(1) or 0)
    raise ValueError(f"unsupported iteration position: {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--kernel-launch-table", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("table", "ncu-args"), default="table",
        help="print the selected mapping or repeatable --kernel-launch-id flags",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.suite_config.read_text(encoding="utf-8"))
    selection = config.get("ncu_iteration_selection")
    if not isinstance(selection, dict):
        raise SystemExit("suite YAML must define ncu_iteration_selection")
    kernel_substring = selection.get("kernel_name_substring")
    positions_spec = selection.get("positions")
    if not isinstance(kernel_substring, str) or not kernel_substring:
        raise SystemExit("ncu_iteration_selection.kernel_name_substring is required")
    if not isinstance(positions_spec, list) or not positions_spec:
        raise SystemExit("ncu_iteration_selection.positions must be a non-empty list")

    with args.kernel_launch_table.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if kernel_substring in (row.get("demangled_name", "") or row.get("Name", ""))
        ]
    rows.sort(key=lambda row: int(row["kernel_launch_id"]))
    total = len(rows)
    if not total:
        raise SystemExit(f"no kernel matching {kernel_substring!r} in {args.kernel_launch_table}")

    if total < 5:
        selected_positions = list(range(1, total + 1))
    else:
        selected_positions = []
        for value in positions_spec:
            position = resolve_position(value, total)
            if 1 <= position <= total and position not in selected_positions:
                selected_positions.append(position)

        # If anchor and middle positions coincide for a short traversal, fill
        # to five distinct iterations with the nearest remaining middle ones.
        middle = total // 2
        for distance in range(total + 1):
            candidates = [middle] if distance == 0 else [middle - distance, middle + distance]
            for position in candidates:
                if 1 <= position <= total and position not in selected_positions:
                    selected_positions.append(position)
                if len(selected_positions) >= min(5, total):
                    break
            if len(selected_positions) >= min(5, total):
                break
        selected_positions.sort()

    selected = [(position, rows[position - 1]) for position in selected_positions]
    if args.format == "ncu-args":
        print(" ".join(f"--kernel-launch-id {row['kernel_launch_id']}" for _, row in selected))
    else:
        print(f"Matching BFS relax iterations: {total}")
        print("iteration\tkernel_launch_id\tmatching_kernel_index")
        for position, row in selected:
            print(
                f"{position}\t{row['kernel_launch_id']}\t"
                f"{row.get('matching_kernel_index', '')}"
            )
        print("NCU flags: " + " ".join(
            f"--kernel-launch-id {row['kernel_launch_id']}" for _, row in selected
        ))


if __name__ == "__main__":
    main()
