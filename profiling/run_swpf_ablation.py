#!/usr/bin/env python3
"""Validate and trace an isolated no-hint control, then replay key hotspots."""

import argparse
import csv
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
from types import SimpleNamespace

from diagnose_swpf import cases, run, trace, write_csv
from run_native_suite import Suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--case-plan", type=Path,
                        help="Reuse explicit configurations for comparison across revisions")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        try:
            descriptor = os.pidfd_open(args.wait_pid)
        except ProcessLookupError:
            descriptor = None
        if descriptor is not None:
            print(f"Waiting for primary profiler PID {args.wait_pid}; no GPU overlap.", flush=True)
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            poller.poll()
            os.close(descriptor)
    verified = list(csv.DictReader((root / "ncu-verified.csv").open()))
    assert len(verified) == len(json.loads((root / "ncu-plan.json").read_text()))
    output = root / "no-hint-timeline"
    binaries = root / "no-hint-control/bin"
    binary_root = Suite.load(args.native_root / "suite-config.yaml").binary_root
    plan = cases(args.native_root, binaries, binary_root)
    if args.case_plan:
        plan = json.loads(args.case_plan.read_text())
        for case in plan:
            case["command"][0] = str(binaries / case["app"])
    validation = []
    for i, case in enumerate(plan, 1):
        folder = output / "validation" / case["app"] / case["graph"] / case["variant"]
        print(f"Control CPU validation [{i}/{len(plan)}] {case['app']} {case['graph']} {case['variant']}", flush=True)
        run([*case["command"], "--validate"], folder, "validate")
        text = (folder / "validate.log").read_text()
        assert "Number of errors : 0" in text
        validation.append(dict(app=case["app"], graph=case["graph"], variant=case["variant"], status="OK"))
        write_csv(output / "validation-summary.csv", validation)
    trace(SimpleNamespace(output=output), plan)
    # Representative positive, neutral and negative cases. Match the original
    # exact function and invocation; this is not a kernel-name substring filter.
    points = [("bfs", "indochina-2004", "spp-l1-d8", "advance", "18"),
              ("bfs", "soc-orkut", "spp-l2-d8", "advance", "5"),
              ("sssp", "roadNet-CA", "gp-l2-d1", "advance", "154"),
              ("sssp", "indochina-2004", "spp-l1-d4", "advance", "18"),
              ("sssp", "soc-orkut", "spp-l1-d8", "advance", "5"),
              ("bc", "indochina-2004", "spp-l2-d4", "backward", "27"),
              ("bc", "soc-orkut", "spp-l1-d8", "forward", "5"),
              ("bc", "soc-orkut", "spp-l1-d8", "backward", "4")]
    invocations = list(csv.DictReader((output / "merge-path-invocations.csv").open()))
    results = []
    for i, point in enumerate(points, 1):
        app, graph, variant, phase, ordinal = point
        print(f"Control NCU [{i}/{len(points)}] {point}", flush=True)
        old = next(row for row in verified if tuple(row[k] for k in ("app", "graph", "variant", "phase", "invocation")) == point)
        original_folder = Path(old["report"]).parent
        folder = output / "ncu" / app / graph / variant / f"{phase}-{ordinal}"
        command = shlex.split((original_folder / "profile-command.txt").read_text())
        command[command.index("--export") + 1] = str(folder / "report")
        command = [str(binaries / app) if item.endswith("/build-v2/bin/" + app) else item for item in command]
        if not (folder / "report.ncu-rep").exists():
            run(command, folder, "profile")
        raw = folder / "raw.csv"
        with raw.open("w") as stream:
            subprocess.run([shutil.which("ncu"), "--import", str(folder / "report.ncu-rep"),
                            "--page", "raw", "--csv", "--print-units", "base",
                            "--print-kernel-base", "mangled"], stdout=stream, check=True)
        data = [row for row in csv.DictReader(raw.open()) if row["ID"].isdigit()]
        expected = next(row for row in invocations if tuple(row[k] for k in ("app", "graph", "variant", "phase", "invocation")) == point)
        assert len(data) == 1 and data[0]["Kernel Name"] == expected["mangled_name"]
        assert int(float(data[0]["launch__grid_size"].replace(",", ""))) == int(expected["grid_blocks"])
        results.append(dict(app=app, graph=graph, variant=variant, phase=phase, invocation=ordinal,
                            nsys_ms=expected["duration_ms"], grid_blocks=expected["grid_blocks"],
                            raw_csv=str(raw), report=str(folder / "report.ncu-rep")))
        write_csv(output / "ncu-verified.csv", results)


if __name__ == "__main__":
    main()
