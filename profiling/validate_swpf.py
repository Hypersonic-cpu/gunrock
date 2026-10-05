#!/usr/bin/env python3
"""CPU-reference checks for every software prefetch variant before native runs."""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

from run_native_suite import Suite


ERRORS = re.compile(r"Number of errors\s*:\s*(\d+)")


def fixtures(folder):
    """Exercise weighted paths, empty rows, duplicate edges and tile crossings."""
    folder.mkdir(parents=True, exist_ok=True)
    cases = {
        "branching": (24, [(0, 1, 0.5), (0, 2, 0.25), (1, 3, 1.5),
                            (2, 3, 0.75), (3, 4, 1.25), (1, 4, 3.0),
                            (2, 3, 0.75), (3, 3, 1.0), (4, 5, 0.5),
                            (5, 6, 1.0), (8, 9, 2.0)]),
        "tile-crossing": (5000, [(0, v, 0.5 + (v % 7) * 0.25)
                                  for v in range(1, 3501)] +
                                 [(v, 4000, 0.5) for v in range(1, 3501)] +
                                 [(4000, 4001, 1.0), (4001, 4002, 1.0)]),
        "deep-chain": (1105, [(v, v + 1, 0.5) for v in range(1100)]),
    }
    paths = []
    for name, (vertices, edges) in cases.items():
        path = folder / (name + ".mtx")
        with path.open("w") as stream:
            stream.write("%%MatrixMarket matrix coordinate real general\n")
            stream.write(f"{vertices} {vertices} {len(edges)}\n")
            for src, dst, weight in edges:
                stream.write(f"{src + 1} {dst + 1} {weight}\n")
        paths.append((name, path, (0, vertices - 1)))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--build-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    suite = Suite.load(args.suite_config)
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS=str(args.threads),
               OMP_DYNAMIC="FALSE")
    if args.large:
        inputs = [(g.name, (suite.binary_root / g.matrix_file), (13331,))
                  for g in suite.graphs]
    else:
        inputs = fixtures(folder / "fixtures")
    rows = []
    fields = ["app", "graph", "source", "swpf", "target", "distance",
              "returncode", "errors", "status", "log"]
    for app in suite.applications:
        for graph, path, sources in inputs:
            for source in sources:
                for algorithm, target, distance in suite.swpf_variants:
                    variant = "none" if algorithm == "none" else f"{algorithm}-{target}-d{distance}"
                    output = folder / app.name / graph / f"src-{source}" / variant
                    output.mkdir(parents=True, exist_ok=True)
                    command = [str((args.build_bin / app.name).resolve()),
                               "--binary-in" if args.large else "--market", str(path),
                               "--src", str(source), "--num_runs", "2", "--validate",
                               "--advance", "merge_path", "--swpf", algorithm,
                               "--swpf-target", target, "--swpf-distance", str(distance)]
                    (output / "command.txt").write_text(" ".join(command) + "\n")
                    result = subprocess.run(command, env=env, text=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, timeout=1800)
                    log = output / "run.log"
                    log.write_text(result.stdout)
                    matches = ERRORS.findall(result.stdout)
                    errors = int(matches[-1]) if matches else -1
                    ok = result.returncode == 0 and errors == 0
                    rows.append(dict(app=app.name, graph=graph, source=source,
                                     swpf=algorithm, target=target, distance=distance,
                                     returncode=result.returncode, errors=errors,
                                     status="OK" if ok else "FAILED", log=str(log)))
                    with (folder / "validation-summary.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"[{len(rows)}] {app.name}/{graph}/src-{source}/{variant}: "
                          f"{rows[-1]['status']} errors={errors}", flush=True)
                    if not ok:
                        print(result.stdout[-4000:], flush=True)
                        return 1
    print(f"CPU validation passed: {len(rows)} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
