#!/usr/bin/env python3
"""Preserve immediate hints as an extra control and promote lookahead hints."""

import argparse
import csv
from pathlib import Path

from collect_swpf_motivation import KEY, summarize
from diagnose_swpf import write_csv


def read(path):
    return list(csv.DictReader(path.open()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "motivation-controls"
    immediate = output / "immediate-warp-cycle-breakdown.csv"
    if not immediate.exists():
        write_csv(immediate, read(output / "warp-cycle-breakdown.csv"))
    initial = read(immediate)
    lookahead = read(root / "motivation-lookahead/warp-cycle-breakdown.csv")
    replacement = {tuple(r[k] for k in KEY): r for r in lookahead}
    assert len(initial) == 32 and len(replacement) == 8
    combined = [replacement[tuple(r[k] for k in KEY)] if r["mode"] == "prefetch_only" else r
                for r in initial]
    write_csv(output / "warp-cycle-breakdown.csv", combined)
    extras = [{**r, "mode": "prefetch_immediate"} for r in initial if r["mode"] == "prefetch_only"]
    write_csv(output / "all-prefetch-controls.csv", combined + extras)
    summarize(root, output)
    print("Merged 32 main comparisons; preserved 8 immediate-hint extra controls")


if __name__ == "__main__":
    main()
