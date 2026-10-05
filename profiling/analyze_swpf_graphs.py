#!/usr/bin/env python3
"""Count useful BC edges by BFS level from existing int/int/float CSR binaries."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from diagnose_swpf import write_csv
from run_native_suite import Suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = Suite.load(args.native_root / "suite-config.yaml")
    rows, summaries = [], []
    for graph in suite.graphs:
        path = suite.binary_root / graph.matrix_file
        n, m, edges = np.fromfile(path, dtype=np.int32, count=3)
        n, m, edges = int(n), int(m), int(edges)
        assert n == m and path.stat().st_size == 12 + 4 * (n + 1) + 8 * edges
        offsets = np.memmap(path, mode="r", dtype=np.int32, offset=12, shape=n + 1)
        indices = np.memmap(path, mode="r", dtype=np.int32, offset=12 + 4 * (n + 1), shape=edges)
        weights = np.memmap(path, mode="r", dtype=np.float32, offset=12 + 4 * (n + 1) + 4 * edges, shape=edges)
        print(f"CPU structural analysis: {graph.name}, {n} vertices, {edges} edges", flush=True)
        # The existing binary layout is read as-is. Directed=True follows the
        # CSR stored directions, including any symmetric expansion done at import.
        matrix = csr_matrix((np.ones(edges, dtype=np.float64), indices, offsets), shape=(n, n))
        distances = shortest_path(matrix, directed=True, unweighted=True, method="D", indices=13331)
        labels = np.where(np.isfinite(distances), distances, -1).astype(np.int32)
        depth = int(labels.max())
        counts, useful = np.zeros(depth + 1, dtype=np.int64), np.zeros(depth + 1, dtype=np.int64)
        degree = np.diff(offsets).astype(np.int64)
        for begin in range(0, n, 65536):
            end = min(n, begin + 65536)
            edge_begin, edge_end = int(offsets[begin]), int(offsets[end])
            owner = np.repeat(labels[begin:end], degree[begin:end])
            neighbor = labels[indices[edge_begin:edge_end]]
            valid = owner >= 0
            counts += np.bincount(owner[valid], minlength=depth + 1)
            kept = valid & (neighbor == owner + 1)
            useful += np.bincount(owner[kept], minlength=depth + 1)
        vertices = np.bincount(labels[labels >= 0], minlength=depth + 1)
        for level in range(depth + 1):
            rows.append(dict(graph=graph.name, source=13331, level=level,
                             forward_invocation=level + 1,
                             backward_invocation=depth - level + 1 if level > 0 else "",
                             frontier_vertices=int(vertices[level]), outgoing_edges=int(counts[level]),
                             next_level_edges=int(useful[level]),
                             next_level_edge_fraction=float(useful[level] / counts[level]) if counts[level] else 0,
                             mean_frontier_degree=float(counts[level] / vertices[level]) if vertices[level] else 0))
        summaries.append(dict(graph=graph.name, vertices=n, edges=edges, source=13331,
                              reached=int(np.count_nonzero(labels >= 0)), max_depth=depth,
                              min_weight=float(weights.min()), max_weight=float(weights.max()),
                              all_unit_weights=bool(np.all(weights == 1))))
        write_csv(args.output / "graph-level-structure.csv", rows)
        (args.output / "graph-structure.json").write_text(json.dumps(summaries, indent=2) + "\n")
        del matrix, distances, labels, offsets, indices, weights


if __name__ == "__main__":
    main()
