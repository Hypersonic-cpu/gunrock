#!/usr/bin/env python3
"""Export native software-prefetch comparison figures for a motivation section."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.native_root.resolve()
    with (root / "best-variants.csv").open(newline="") as stream:
        best = list(csv.DictReader(stream))
    with (root / "swpf-comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [f"{row['app'].upper()}\n{row['graph']}" for row in best]
    x = np.arange(len(best))
    figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    axis.bar(x - 0.18, [float(row["gp_speedup"]) for row in best],
             width=0.36, label="Best GP", color="#3973ac")
    axis.bar(x + 0.18, [float(row["spp_speedup"]) for row in best],
             width=0.36, label="Best SPP", color="#dc8734")
    axis.axhline(1.0, color="#444444", linewidth=1, linestyle="--")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylabel("Speedup over none (median of 10 runs)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(root / "best-prefetch-speedup.pdf")
    figure.savefig(root / "best-prefetch-speedup.png", dpi=300)
    plt.close(figure)

    variants = [(a, t, d) for a in ("gp", "spp") for t in ("l1", "l2")
                for d in (1, 2, 4, 8)]
    matrix = []
    for case in best:
        choices = {(row["swpf"], row["swpf_target"], int(row["swpf_distance"])):
                   float(row["speedup_over_none"]) for row in rows
                   if row["benchmark"] == case["app"] and row["graph"] == case["graph"]}
        matrix.append([choices[variant] for variant in variants])
    data = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    norm = TwoSlopeNorm(vmin=min(0.5, float(data.min())), vcenter=1.0,
                        vmax=max(1.5, float(data.max())))
    image = axis.imshow(data, cmap="RdBu", norm=norm, aspect="auto")
    axis.set_yticks(np.arange(len(best)),
                   [f"{row['app'].upper()} / {row['graph']}" for row in best])
    axis.set_xticks(np.arange(len(variants)),
                   [f"{a.upper()}\n{t.upper()}\nd={d}" for a, t, d in variants], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axis.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="Speedup over none (median of 10 runs)")
    figure.savefig(root / "all-prefetch-speedups.pdf")
    figure.savefig(root / "all-prefetch-speedups.png", dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
