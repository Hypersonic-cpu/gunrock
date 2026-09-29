#!/usr/bin/env python3
"""Run the native benchmark cases declared in a YAML suite file."""

import argparse
import csv
import datetime as dt
import json
import os
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parent.parent
GPU_TIME_RE = re.compile(r"GPU Elapsed Time\s*:\s*([-+0-9.eE]+)\s*\(ms\)")
TRIANGLE_RE = re.compile(r"Total Graph Traingles\s*:\s*(\d+)")
GRAPH_VERTICES_RE = re.compile(r"Graph vertices\s*:\s*(\d+)")
GRAPH_EDGES_RE = re.compile(r"Graph CSR edges\s*:\s*(\d+)")
VALID_REPEAT_MODES = {"runs_in_one_process", "fresh_processes"}
SUMMARY_FIELDS = [
    "benchmark", "graph", "load_balance_requested", "load_balance_effective",
    "vertices", "csr_edges", "samples", "gpu_median_ms", "gpu_mean_ms",
    "gpu_stddev_ms", "gpu_min_ms", "gpu_max_ms", "host_user_cpu_s",
    "host_system_cpu_s", "process_wall_s", "status", "notes",
]


class SuiteConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GraphSpec:
    name: str
    matrix_file: str
    vertices: int
    csr_edges: int

    @classmethod
    def from_yaml(cls, name, raw):
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise SuiteConfigError("each graph must map a name to a mapping")
        matrix_file = raw.get("matrix_file")
        vertices = raw.get("vertices")
        csr_edges = raw.get("csr_edges")
        if not isinstance(matrix_file, str) or not matrix_file:
            raise SuiteConfigError(f"{name}: matrix_file must be a non-empty string")
        for field, value in (("vertices", vertices), ("csr_edges", csr_edges)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SuiteConfigError(f"{name}: {field} must be a positive integer")
        return cls(name, matrix_file, vertices, csr_edges)


@dataclass(frozen=True)
class BenchmarkCase:
    graph: str
    source: int = 0
    reduce_all_triangles: bool = False

    @classmethod
    def from_yaml(cls, app_name, raw, graphs):
        if not isinstance(raw, dict):
            raise SuiteConfigError(f"{app_name}: each case must be a mapping")
        graph = raw.get("graph")
        if graph not in graphs:
            raise SuiteConfigError(f"{app_name}: unsupported or missing graph {graph!r}")
        source = raw.get("source", 0)
        if not isinstance(source, int) or isinstance(source, bool) or source < 0:
            raise SuiteConfigError(f"{app_name}/{graph}: source must be a non-negative integer")
        if source >= graphs[graph].vertices:
            raise SuiteConfigError(
                f"{app_name}/{graph}: source {source} is outside the configured vertex range "
                f"[0, {graphs[graph].vertices})"
            )
        reduce = raw.get("reduce_all_triangles", False)
        if not isinstance(reduce, bool):
            raise SuiteConfigError(f"{app_name}/{graph}: reduce_all_triangles must be true or false")
        return cls(graph=graph, source=source, reduce_all_triangles=reduce)


@dataclass(frozen=True)
class Application:
    name: str
    args: tuple
    repeat_mode: str
    cases: tuple
    load_balance_flag: str = ""
    load_balance_policy: str = ""
    fixed_load_balance: str = ""

    @classmethod
    def from_yaml(cls, name, raw, supported_policies, graphs):
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise SuiteConfigError("each application must map a name to a mapping")
        repeat_mode = raw.get("repeat_mode")
        if repeat_mode not in VALID_REPEAT_MODES:
            raise SuiteConfigError(f"{name}: repeat_mode must be one of {sorted(VALID_REPEAT_MODES)}")
        args = raw.get("args")
        if not isinstance(args, list) or not args or not all(isinstance(arg, str) for arg in args):
            raise SuiteConfigError(f"{name}: args must be a non-empty list of strings")
        raw_cases = raw.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise SuiteConfigError(f"{name}: cases must be a non-empty list")
        cases = tuple(BenchmarkCase.from_yaml(name, case, graphs) for case in raw_cases)
        if len({(case.graph, case.source) for case in cases}) != len(cases):
            raise SuiteConfigError(f"{name}: duplicate graph/source case")

        lb = raw.get("load_balance")
        if not isinstance(lb, dict):
            raise SuiteConfigError(f"{name}: load_balance must be a mapping")
        if "flag" in lb:
            flag, policy = lb["flag"], lb.get("policy")
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise SuiteConfigError(f"{name}: load_balance.flag must be a CLI flag")
            if not isinstance(policy, str) or policy not in supported_policies:
                raise SuiteConfigError(f"{name}: load_balance.policy must be supported")
            return cls(name, tuple(args), repeat_mode, cases, flag, policy, "")
        effective = lb.get("effective")
        if not isinstance(effective, str) or not effective:
            raise SuiteConfigError(f"{name}: set load_balance.flag/policy or describe load_balance.effective")
        return cls(name, tuple(args), repeat_mode, cases, "", "", effective)

    def effective_load_balance(self):
        return self.load_balance_policy if self.load_balance_flag else self.fixed_load_balance

    def requested_load_balance(self):
        return self.load_balance_policy if self.load_balance_flag else "not configurable"

    def command(self, executable, values):
        args = [arg.format_map(values) for arg in self.args]
        if self.load_balance_flag:
            args.extend((self.load_balance_flag, self.load_balance_policy))
        return [str(executable), *args]


@dataclass(frozen=True)
class Suite:
    path: Path
    runs: int
    supported_policies: tuple
    graphs: tuple
    applications: tuple

    @classmethod
    def load(cls, path):
        path = path.expanduser().resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SuiteConfigError(f"cannot read YAML suite config {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SuiteConfigError("suite config must be a YAML mapping")
        runs = raw.get("runs")
        if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1:
            raise SuiteConfigError("suite config 'runs' must be a positive integer")
        policies = raw.get("supported_load_balances")
        if not isinstance(policies, list) or not policies or not all(isinstance(x, str) for x in policies):
            raise SuiteConfigError("supported_load_balances must be a non-empty list of strings")
        graph_map = raw.get("graphs")
        if not isinstance(graph_map, dict) or not graph_map:
            raise SuiteConfigError("graphs must map selected graph names to matrix and size metadata")
        graphs = tuple(GraphSpec.from_yaml(name, graph) for name, graph in graph_map.items())
        graph_specs = {graph.name: graph for graph in graphs}
        app_map = raw.get("applications")
        if not isinstance(app_map, dict) or not app_map:
            raise SuiteConfigError("applications must contain at least one selected application")
        apps = tuple(
            Application.from_yaml(name, app, policies, graph_specs)
            for name, app in app_map.items()
        )
        return cls(path, runs, tuple(policies), graphs, apps)

    def cases(self):
        return [(app, case) for app in self.applications for case in app.cases]

    def graph(self, name):
        return next(graph for graph in self.graphs if graph.name == name)


class NativeRunner:
    def __init__(self, suite, output_dir, data_root, build_bin, cuda_home, device):
        self.suite = suite
        self.root = output_dir.expanduser().resolve()
        self.data_root = data_root.expanduser().resolve()
        self.build_bin = build_bin.expanduser().resolve()
        self.cuda_home = cuda_home.expanduser().resolve()
        self.device = str(device)
        self.rows = []
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = self.device
        self.env["CUDA_HOME"] = str(self.cuda_home)
        self.env["PATH"] = str(self.cuda_home / "bin") + os.pathsep + self.env.get("PATH", "")
        self.env["LD_LIBRARY_PATH"] = str(self.cuda_home / "lib64") + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")
        self.env.pop("GUNROCK_PROFILE_NVTX", None)

    def case_dir(self, app, case):
        return self.root / "full-suite" / "gunrock-v2" / app.name / case.graph / f"src-{case.source}" / "native"

    def input_path(self, case):
        return self.data_root / self.suite.graph(case.graph).matrix_file

    def values_for(self, app, case, folder):
        return {
            "matrix": str(self.input_path(case)),
            "source": case.source,
            "runs": self.suite.runs,
            "output_dir": str(folder),
            "reduce": str(case.reduce_all_triangles).lower(),
        }

    def command_for(self, app, case, folder):
        return app.command(self.build_bin / app.name, self.values_for(app, case, folder))

    def render_command(self, command):
        return "CUDA_VISIBLE_DEVICES=" + self.device + " " + shlex.join(command)

    def preflight(self):
        missing = []
        for app, case in self.suite.cases():
            executable = self.build_bin / app.name
            matrix = self.input_path(case)
            if not executable.is_file() or not os.access(executable, os.X_OK):
                missing.append(f"executable: {executable}")
            if not matrix.is_file():
                missing.append(f"matrix: {matrix}")
        if missing:
            raise SuiteConfigError("required benchmark inputs are missing:\n  " + "\n  ".join(missing))

    def write_dry_run(self):
        plan_cases = []
        for index, (app, case) in enumerate(self.suite.cases(), 1):
            folder = self.case_dir(app, case)
            folder.mkdir(parents=True, exist_ok=True)
            command = self.command_for(app, case, folder)
            rendered = self.render_command(command)
            if app.repeat_mode == "fresh_processes":
                rendered = f"{self.suite.runs} fresh processes; each: {rendered}"
            (folder / "command.txt").write_text(rendered + "\n", encoding="utf-8")
            (folder / "input.txt").write_text(str(self.input_path(case)) + "\n", encoding="utf-8")
            (folder / "run-count.txt").write_text(f"{self.suite.runs}\n", encoding="utf-8")
            plan_cases.append({
                "index": index,
                "application": app.name,
                "graph": case.graph,
                "source": case.source,
                "runs": self.suite.runs,
                "vertices": self.suite.graph(case.graph).vertices,
                "csr_edges": self.suite.graph(case.graph).csr_edges,
                "load_balance": app.effective_load_balance(),
                "command": rendered,
                "output_dir": str(folder),
            })
        plan = {
            "suite_config": str(self.suite.path),
            "runs": self.suite.runs,
            "selected_applications": [app.name for app in self.suite.applications],
            "cases": plan_cases,
            "commands_executed": False,
        }
        (self.root / "run-plan.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False), encoding="utf-8"
        )
        lines = [
            "# Native suite dry-run", "",
            f"Suite config: `{self.suite.path}`",
            f"Selected apps: {', '.join(app.name for app in self.suite.applications)}",
            f"Runs per case: {self.suite.runs}",
            "Benchmark commands executed: **no**.", "",
            "| # | App | Graph | Effective LB | Command |",
            "|---:|---|---|---|---|",
        ]
        for row in plan_cases:
            lines.append(
                f"| {row['index']} | {row['application']} | {row['graph']} | "
                f"{row['load_balance']} | `{row['command']}` |"
            )
        (self.root / "dry-run-plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Dry run only; no benchmark commands were executed.", flush=True)
        print(f"Plan: {self.root / 'dry-run-plan.md'}", flush=True)

    @staticmethod
    def child_usage():
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime, usage.ru_stime

    def execute(self, command):
        before = self.child_usage()
        start = time.perf_counter()
        process = subprocess.run(
            command, env=self.env, text=True, errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=21600,
        )
        wall = time.perf_counter() - start
        after = self.child_usage()
        return process, wall, after[0] - before[0], after[1] - before[1]

    def timing_stats(self, samples):
        return {
            "gpu_median_ms": statistics.median(samples),
            "gpu_mean_ms": statistics.mean(samples),
            "gpu_stddev_ms": statistics.pstdev(samples),
            "gpu_min_ms": min(samples),
            "gpu_max_ms": max(samples),
        }

    def run_case(self, app, case):
        folder = self.case_dir(app, case)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.txt").write_text(str(self.input_path(case)) + "\n", encoding="utf-8")
        (folder / "run-count.txt").write_text(f"{self.suite.runs}\n", encoding="utf-8")
        result_file = folder / "native.json"
        command = self.command_for(app, case, folder)
        samples, host_user, host_system, wall_total = [], 0.0, 0.0, 0.0

        if app.repeat_mode == "runs_in_one_process":
            (folder / "command.txt").write_text(self.render_command(command) + "\n", encoding="utf-8")
            process, wall_total, host_user, host_system = self.execute(command)
            (folder / "run.log").write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{app.name} exited {process.returncode}; inspect run.log")
            if not result_file.is_file():
                raise RuntimeError(f"{app.name} did not write native.json")
            native = json.loads(result_file.read_text(encoding="utf-8"))
            samples = [float(value) for value in native.get("process_times", [])]
            if len(samples) != self.suite.runs:
                raise RuntimeError(f"expected {self.suite.runs} timing samples, got {len(samples)}")
            vertices = int(native.get("num_vertices", -1))
            edges = int(native.get("num_edges", -1))
            graph = self.suite.graph(case.graph)
            if (vertices, edges) != (graph.vertices, graph.csr_edges):
                raise RuntimeError(
                    f"graph size mismatch: got V={vertices} E={edges}; "
                    f"YAML expects V={graph.vertices} E={graph.csr_edges}"
                )
            native["native_campaign"] = {
                "sample_count": len(samples),
                "gpu_times_ms": samples,
                "load_balance_requested": app.requested_load_balance(),
                "load_balance_effective": app.effective_load_balance(),
                "gpu_time_source": "CUDA events around each algorithm run",
                "host_user_cpu_s": host_user,
                "host_system_cpu_s": host_system,
                "process_wall_s": wall_total,
                "process_wall_scope": f"one process: Matrix Market load, graph construction, {self.suite.runs} runs, export",
                "profiling_counters_used": False,
            }
            result_file.write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")
        else:
            rendered = f"{self.suite.runs} fresh processes; each: {self.render_command(command)}\n"
            (folder / "command.txt").write_text(rendered, encoding="utf-8")
            triangle_totals = []
            graph = self.suite.graph(case.graph)
            for index in range(1, self.suite.runs + 1):
                process, wall, user, system = self.execute(command)
                wall_total += wall
                host_user += user
                host_system += system
                (folder / f"run-{index:02d}.log").write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(f"{app.name} process {index} exited {process.returncode}")
                vertices_match = GRAPH_VERTICES_RE.search(process.stdout)
                edges_match = GRAPH_EDGES_RE.search(process.stdout)
                if not vertices_match or not edges_match:
                    raise RuntimeError(
                        f"{app.name} process {index} did not report graph dimensions"
                    )
                vertices = int(vertices_match.group(1))
                edges = int(edges_match.group(1))
                if (vertices, edges) != (graph.vertices, graph.csr_edges):
                    raise RuntimeError(
                        f"graph size mismatch: got V={vertices} E={edges}; "
                        f"YAML expects V={graph.vertices} E={graph.csr_edges}"
                    )
                match = GPU_TIME_RE.search(process.stdout)
                if not match:
                    raise RuntimeError(f"{app.name} process {index} had no GPU elapsed time")
                samples.append(float(match.group(1)))
                match = TRIANGLE_RE.search(process.stdout)
                triangle_totals.append(int(match.group(1)) if match else None)
                print(f"    {app.name.upper()} sample {index}/{self.suite.runs}: GPU {samples[-1]:.3f} ms", flush=True)
            native = {
                "primitive": app.name,
                "graph_file": str(self.input_path(case)),
                "num_vertices": vertices,
                "num_edges": edges,
                "reduce_all_triangles": case.reduce_all_triangles,
                "process_times": samples,
                "avg_process_time": statistics.mean(samples),
                "stddev_process_time": statistics.pstdev(samples),
                "min_process_time": min(samples),
                "max_process_time": max(samples),
                "native_campaign": {
                    "sample_count": len(samples),
                    "gpu_times_ms": samples,
                    "load_balance_requested": app.requested_load_balance(),
                    "load_balance_effective": app.effective_load_balance(),
                    "gpu_time_source": "CUDA events around the algorithm run",
                    "host_user_cpu_s": host_user,
                    "host_system_cpu_s": host_system,
                    "process_wall_s": wall_total,
                    "process_wall_scope": f"sum of {self.suite.runs} fresh processes including matrix loading and graph construction",
                    "triangle_totals_reported": triangle_totals,
                    "profiling_counters_used": False,
                },
            }
            result_file.write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")

        note = ""
        if app.name == "tc" and app.repeat_mode == "fresh_processes":
            totals = native["native_campaign"]["triangle_totals_reported"]
            if any(value is not None for value in totals) and all(
                value == 0 for value in totals if value is not None
            ):
                note = "triangle total reported as 0"
        values = {
            **self.timing_stats(samples),
            "samples": len(samples),
            "vertices": self.suite.graph(case.graph).vertices,
            "csr_edges": self.suite.graph(case.graph).csr_edges,
            "host_user_cpu_s": host_user,
            "host_system_cpu_s": host_system,
            "process_wall_s": wall_total,
        }
        self.add_row(app, case, values, "OK", note)

    def add_row(self, app, case, values=None, status="FAILED", notes=""):
        values = values or {}
        graph = self.suite.graph(case.graph)
        vertices, edges = graph.vertices, graph.csr_edges
        row = {
            "benchmark": app.name,
            "graph": case.graph,
            "load_balance_requested": app.requested_load_balance(),
            "load_balance_effective": app.effective_load_balance(),
            "vertices": values.get("vertices", vertices),
            "csr_edges": values.get("csr_edges", edges),
            "samples": values.get("samples", 0),
            "gpu_median_ms": values.get("gpu_median_ms", ""),
            "gpu_mean_ms": values.get("gpu_mean_ms", ""),
            "gpu_stddev_ms": values.get("gpu_stddev_ms", ""),
            "gpu_min_ms": values.get("gpu_min_ms", ""),
            "gpu_max_ms": values.get("gpu_max_ms", ""),
            "host_user_cpu_s": values.get("host_user_cpu_s", ""),
            "host_system_cpu_s": values.get("host_system_cpu_s", ""),
            "process_wall_s": values.get("process_wall_s", ""),
            "status": status,
            "notes": notes.replace("|", "/"),
        }
        self.rows.append(row)
        self.write_summary()

    def write_summary(self):
        with (self.root / "native-summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        policies = ", ".join(
            f"`{app.name}={app.load_balance_policy}`"
            for app in self.suite.applications if app.load_balance_flag
        ) or "none"
        in_process = [app.name for app in self.suite.applications if app.repeat_mode == "runs_in_one_process"]
        fresh_process = [app.name for app in self.suite.applications if app.repeat_mode == "fresh_processes"]
        lines = [
            "# Gunrock v2 native timing summary", "",
            f"Run date: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"Load-balance policies: {policies}.",
            f"Primary statistic: median of {self.suite.runs} CUDA-event algorithm timings.", "",
            "| Benchmark | Graph matrix | Requested LB | Effective LB | V | CSR edges | N | GPU median (ms) | GPU mean ± SD (ms) | GPU min–max (ms) | Host user CPU (s) | Host system CPU (s) | Process wall (s) | Status | Notes |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in self.rows:
            def number(key):
                value = row.get(key, "")
                return f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(
                f"| {row['benchmark']} | {row['graph']} | {row['load_balance_requested']} | "
                f"{row['load_balance_effective']} | {row['vertices']} | {row['csr_edges']} | "
                f"{row['samples']} | {number('gpu_median_ms')} | {number('gpu_mean_ms')} ± "
                f"{number('gpu_stddev_ms')} | {number('gpu_min_ms')}–{number('gpu_max_ms')} | "
                f"{number('host_user_cpu_s')} | {number('host_system_cpu_s')} | "
                f"{number('process_wall_s')} | {row['status']} | {row['notes']} |"
            )
        lines += [
            "", "## Measurement details", "",
            f"- All {self.suite.runs} GPU samples per case are retained in native.json.",
            f"- Apps run {self.suite.runs} times in one process: {', '.join(in_process) or 'none'}.",
            f"- Apps run in {self.suite.runs} fresh processes: {', '.join(fresh_process) or 'none'}.",
            "- Host user/system CPU time and wall time include matrix loading, graph construction/upload, algorithm calls, and output.",
            "- Built with CUDA 12.6, SM 86, Release, and ESSENTIALS_COLLECT_METRICS=OFF. Nsight and hardware counters were not used.",
            "- No CPU reference validation was run.",
        ]
        selected = {app.name for app in self.suite.applications}
        if "bc" in selected:
            lines.insert(-1, "- BC uses hard-coded merge_path_v2.")
        if "pr" in selected:
            lines.insert(-1, "- PR uses edge-parallel parallel_for and has no advance load-balance setting.")
        (self.root / "native-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def command_output(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return ""
        return result.stdout.strip()

    def environment_info(self):
        return {
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "repository": str(REPO),
            "branch": self.command_output(["git", "-C", str(REPO), "branch", "--show-current"]),
            "commit": self.command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "suite_config": str(self.root / "suite-config.yaml"),
            "selected_applications": [app.name for app in self.suite.applications],
            "load_balance_policies": {
                app.name: app.effective_load_balance() for app in self.suite.applications
            },
            "runs_per_case": self.suite.runs,
            "data_root": str(self.data_root),
            "results_root": str(self.root),
            "build_bin": str(self.build_bin),
            "cuda_home": str(self.cuda_home),
            "cuda_visible_devices": self.device,
            "build_type": "Release",
            "cuda_architecture": "86",
            "essentials_collect_metrics": False,
            "nsys_used": False,
            "ncu_used": False,
            "profiling_counters_used": False,
            "gpu_info": self.command_output([
                "nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,memory.total", "--format=csv"
            ]),
            "nvcc_version": self.command_output([str(self.cuda_home / "bin/nvcc"), "--version"]),
        }

    def run(self, dry_run=False):
        self.preflight()
        if self.root.exists() and any(self.root.iterdir()):
            raise SuiteConfigError(f"output directory is not empty; refusing to overwrite: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.suite.path, self.root / "suite-config.yaml")
        if dry_run:
            self.write_dry_run()
            return 0

        metadata = self.environment_info()
        (self.root / "environment.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        (self.root / "native-summary.md").write_text("# Native run in progress\n", encoding="utf-8")
        cases = self.suite.cases()
        for index, (app, case) in enumerate(cases, 1):
            print(
                f"[{index}/{len(cases)}] {app.name} x {case.graph} "
                f"(N={self.suite.runs}, {app.repeat_mode})",
                flush=True,
            )
            try:
                self.run_case(app, case)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                folder = self.case_dir(app, case)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                self.add_row(app, case, status="FAILED", notes=str(exc))
        metadata["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        metadata["successful_cases"] = sum(row["status"] == "OK" for row in self.rows)
        metadata["failed_cases"] = sum(row["status"] != "OK" for row in self.rows)
        (self.root / "environment.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        print(
            f"Completed: {metadata['successful_cases']} succeeded, "
            f"{metadata['failed_cases']} failed",
            flush=True,
        )
        print(f"Summary: {self.root / 'native-summary.md'}", flush=True)
        return 0 if metadata["failed_cases"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run native benchmarks selected by the suite YAML."
    )
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new results directory; must be absent or empty")
    parser.add_argument("--suite-config", required=True, type=Path,
                        help="YAML suite file; only its applications are scheduled")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/data-8/Graph-Datasets/mtx"))
    parser.add_argument(
        "--build-bin",
        type=Path,
        default=Path(os.environ.get("GUNROCK_BUILD_BIN", REPO / "build" / "bin")),
        help="directory containing benchmark executables "
             "(default: GUNROCK_BUILD_BIN or <repo>/build/bin)",
    )
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.6"))
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--dry-run", action="store_true",
                        help="write commands and output paths without executing benchmarks")
    args = parser.parse_args()
    try:
        suite = Suite.load(args.suite_config)
        runner = NativeRunner(
            suite, args.output_dir, args.data_root, args.build_bin, args.cuda_home, args.device
        )
        return runner.run(dry_run=args.dry_run)
    except SuiteConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
