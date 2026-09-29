#!/usr/bin/env python3
"""Run YAML-selected native applications under Nsight profiling modes."""

import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import shlex
import sqlite3
import statistics
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


REPO = Path(__file__).resolve().parent.parent
PROFILE_ONE = Path(__file__).resolve().with_name("profile_one.sh")
MODES = ("nsys-trace", "nsys-metrics", "ncu")


class ProfileConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Graph:
    name: str
    matrix_file: str
    vertices: int
    csr_edges: int


@dataclass(frozen=True)
class Case:
    graph: str
    source: int
    reduce_all_triangles: bool


@dataclass(frozen=True)
class Application:
    name: str
    args: tuple
    cases: tuple
    load_balance_flag: str = ""
    load_balance_policy: str = ""
    fixed_load_balance: str = ""

    @classmethod
    def from_yaml(cls, name, raw, graphs, supported_load_balances):
        if not isinstance(raw, dict):
            raise ProfileConfigError(f"{name}: application must be a mapping")
        args = raw.get("args")
        cases_raw = raw.get("cases")
        if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
            raise ProfileConfigError(f"{name}: args must be a non-empty list of strings")
        if not isinstance(cases_raw, list) or not cases_raw:
            raise ProfileConfigError(f"{name}: cases must be a non-empty list")

        cases = []
        for raw_case in cases_raw:
            if not isinstance(raw_case, dict):
                raise ProfileConfigError(f"{name}: each case must be a mapping")
            raw_sources = raw_case.get("source", [0])
            if (isinstance(raw_sources, list) and len(raw_sources) > 1
                    and not any("{source}" in arg for arg in args)):
                raise ProfileConfigError(
                    f"{name}: multiple sources require an args entry containing {{source}}"
                )
            graph = raw_case.get("graph")
            if graph not in graphs:
                raise ProfileConfigError(f"{name}: unknown graph {graph!r}")
            sources = raw_case.get("source", [0])
            if isinstance(sources, int) and not isinstance(sources, bool):
                sources = [sources]  # Accept scalar values from older suite files.
            if (not isinstance(sources, list) or not sources
                    or any(not isinstance(source, int) or isinstance(source, bool) or source < 0
                           for source in sources)):
                raise ProfileConfigError(
                    f"{name}/{graph}: source must be a non-empty list of non-negative integers"
                )
            if len(set(sources)) != len(sources):
                raise ProfileConfigError(f"{name}/{graph}: source list contains duplicates")
            for source in sources:
                if source >= graphs[graph].vertices:
                    raise ProfileConfigError(
                        f"{name}/{graph}: source {source} is outside [0, {graphs[graph].vertices})"
                    )
            reduce = raw_case.get("reduce_all_triangles", False)
            if not isinstance(reduce, bool):
                raise ProfileConfigError(f"{name}/{graph}: reduce_all_triangles must be boolean")
            cases.extend(Case(graph, source, reduce) for source in sources)

        if len({(case.graph, case.source) for case in cases}) != len(cases):
            raise ProfileConfigError(f"{name}: duplicate graph/source case")

        load_balance = raw.get("load_balance")
        if not isinstance(load_balance, dict):
            raise ProfileConfigError(f"{name}: load_balance must be a mapping")
        flag = load_balance.get("flag", "")
        policy = load_balance.get("policy", "")
        effective = load_balance.get("effective", "")
        if flag:
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise ProfileConfigError(f"{name}: load_balance.flag must be a CLI option")
            if policy not in supported_load_balances:
                raise ProfileConfigError(f"{name}: unsupported load-balance policy {policy!r}")
            return cls(name, tuple(args), tuple(cases), flag, policy, "")
        if not isinstance(effective, str) or not effective:
            raise ProfileConfigError(f"{name}: expected load_balance.flag/policy or load_balance.effective")
        return cls(name, tuple(args), tuple(cases), "", "", effective)

    def command(self, executable, values):
        args = [arg.format_map(values) for arg in self.args]
        if self.load_balance_flag:
            args.extend((self.load_balance_flag, self.load_balance_policy))
        return [str(executable), *args]

    def effective_load_balance(self):
        return self.load_balance_policy if self.load_balance_flag else self.fixed_load_balance


@dataclass(frozen=True)
class GpuMetrics:
    metric_set: str
    frequencies_hz: tuple


@dataclass(frozen=True)
class Suite:
    path: Path
    runs: int
    graphs: dict
    applications: tuple
    gpu_metrics: GpuMetrics
    binary_root: Optional[Path] = None

    @classmethod
    def load(cls, path):
        path = path.expanduser().resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ProfileConfigError(f"cannot read YAML suite {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProfileConfigError("suite must be a YAML mapping")
        runs = raw.get("runs")
        if not isinstance(runs, int) or isinstance(runs, bool) or runs != 1:
            raise ProfileConfigError("profiling suite must set runs: 1")

        binary_root_value = raw.get("binary_root")
        binary_root = None
        if binary_root_value is not None:
            if not isinstance(binary_root_value, str) or not binary_root_value.strip():
                raise ProfileConfigError("binary_root must be a non-empty path when specified")
            binary_root = Path(binary_root_value).expanduser()
            if not binary_root.is_absolute():
                binary_root = path.parent / binary_root
            binary_root = binary_root.resolve()

        policy_list = raw.get("supported_load_balances")
        if not isinstance(policy_list, list) or not policy_list:
            raise ProfileConfigError("supported_load_balances must be a non-empty list")
        if not all(isinstance(value, str) and value for value in policy_list):
            raise ProfileConfigError("supported_load_balances entries must be non-empty strings")

        graph_data = raw.get("graphs")
        if not isinstance(graph_data, dict) or not graph_data:
            raise ProfileConfigError("graphs must map names to matrix metadata")
        graphs = {}
        for name, spec in graph_data.items():
            if not isinstance(spec, dict):
                raise ProfileConfigError(f"{name}: graph must be a mapping")
            matrix_file = spec.get("matrix_file")
            vertices = spec.get("vertices")
            edges = spec.get("csr_edges")
            if not isinstance(matrix_file, str) or not matrix_file:
                raise ProfileConfigError(f"{name}: matrix_file must be a non-empty string")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in (vertices, edges)):
                raise ProfileConfigError(f"{name}: vertices and csr_edges must be positive integers")
            graphs[name] = Graph(name, matrix_file, vertices, edges)

        app_data = raw.get("applications")
        if not isinstance(app_data, dict) or not app_data:
            raise ProfileConfigError("applications must contain selected applications")
        apps = tuple(
            Application.from_yaml(name, spec, graphs, policy_list)
            for name, spec in app_data.items()
        )
        metrics_data = raw.get("gpu_metrics")
        if not isinstance(metrics_data, dict):
            raise ProfileConfigError("gpu_metrics must define metric_set and frequencies_hz")
        metric_set = metrics_data.get("metric_set")
        frequencies = metrics_data.get("frequencies_hz")
        if not isinstance(metric_set, str) or not metric_set:
            raise ProfileConfigError("gpu_metrics.metric_set must be a non-empty string")
        if (not isinstance(frequencies, list) or not frequencies
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 10 or value > 200000 for value in frequencies)):
            raise ProfileConfigError("gpu_metrics.frequencies_hz must be a list in [10, 200000]")
        if frequencies != sorted(set(frequencies)):
            raise ProfileConfigError("gpu_metrics.frequencies_hz must be strictly increasing")
        gpu_metrics = GpuMetrics(metric_set, tuple(frequencies))
        return cls(path, runs, graphs, apps, gpu_metrics, binary_root)

    def cases(self):
        return [(app, case) for app in self.applications for case in app.cases]


class NsysTraceRunner:
    mode_dir = "nsys-trace"

    def __init__(self, suite, args):
        self.suite = suite
        self.root = args.output_dir.expanduser().resolve()
        self.data_root = args.data_root.expanduser().resolve()
        self.build_bin = args.build_bin.expanduser().resolve()
        self.device = str(args.device)
        self.nsys = shutil.which("nsys")
        self.sudo = shutil.which("sudo") or "sudo"
        self.only = self.parse_only(args.only)
        self.force = bool(getattr(args, "force", False))
        self.metadata_dir = None
        self.rows = []

    @staticmethod
    def parse_only(value):
        if not value:
            return None
        return {item.strip() for item in value.split(",") if item.strip()}

    def selected_cases(self):
        cases = self.suite.cases()
        if self.only is None:
            return cases
        selected = [
            (app, case) for app, case in cases
            if (app.name in self.only
                or f"{app.name}/{case.graph}" in self.only
                or f"{app.name}/{case.graph}/src-{case.source}" in self.only)
        ]
        known = {app.name for app, _ in cases} | {
            f"{app.name}/{case.graph}" for app, case in cases
        } | {
            f"{app.name}/{case.graph}/src-{case.source}" for app, case in cases
        }
        unknown = self.only - known
        if unknown:
            raise ProfileConfigError(f"unknown --only selection(s): {', '.join(sorted(unknown))}")
        if not selected:
            raise ProfileConfigError("--only selected no cases")
        return selected

    def case_dir(self, app, case):
        return self.root / app.name / case.graph / f"src-{case.source}"

    def mode_output_dir(self, app, case):
        return self.case_dir(app, case) / self.mode_dir

    def binary_path(self, graph):
        if self.suite.binary_root is None:
            raise ProfileConfigError(
                "suite config must set binary_root when an application uses {binary}"
            )
        binary_name = Path(graph.matrix_file).with_suffix(".bin").name
        return self.suite.binary_root / binary_name

    def prepare_run(self, cases):
        """Prepare selected mode outputs and create an invocation-specific metadata folder."""
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_root = self.root / "_meta"
        metadata_root.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z")
        for suffix in range(10000):
            name = timestamp if suffix == 0 else f"{timestamp}-{suffix}"
            candidate = metadata_root / name
            try:
                candidate.mkdir()
                self.metadata_dir = candidate
                break
            except FileExistsError:
                continue
        if self.metadata_dir is None:
            raise ProfileConfigError(f"cannot create a unique metadata directory in {metadata_root}")
        shutil.copyfile(self.suite.path, self.metadata_dir / "suite-config.yaml")

        for app, case in cases:
            output = self.mode_output_dir(app, case)
            if output.exists() or output.is_symlink():
                if self.force:
                    if output.is_symlink() or not output.is_dir():
                        output.unlink()
                    else:
                        shutil.rmtree(output)
                else:
                    raise ProfileConfigError(
                        f"output already exists for {app.name} x {case.graph} "
                        f"source={case.source} mode={self.mode_dir}: {output} "
                        "(use --force to overwrite this case/mode)"
                    )

    def preflight(self, cases):
        if not self.nsys:
            raise ProfileConfigError("nsys was not found on PATH")
        for app, case in cases:
            executable = self.build_bin / app.name
            graph = self.suite.graphs[case.graph]
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ProfileConfigError(f"executable missing or not executable: {executable}")
            if any("{matrix}" in arg for arg in app.args):
                matrix = self.data_root / graph.matrix_file
                if not matrix.is_file():
                    raise ProfileConfigError(f"matrix is missing: {matrix}")
            if any("{binary}" in arg for arg in app.args):
                binary = self.binary_path(graph)
                if not binary.is_file():
                    raise ProfileConfigError(f"binary CSR graph is missing: {binary}")
        version = subprocess.run(
            [self.sudo, "-n", self.nsys, "--version"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if version.returncode:
            raise ProfileConfigError(f"cannot run Nsight Systems with sudo -n:\n{version.stdout}")
        if not self.force:
            for app, case in cases:
                output = self.mode_output_dir(app, case)
                if output.exists() or output.is_symlink():
                    raise ProfileConfigError(
                        f"output already exists for {app.name} x {case.graph} "
                        f"source={case.source} mode={self.mode_dir}: {output} "
                        "(use --force to overwrite this case/mode)"
                    )

    def app_command(self, app, case, app_output):
        graph = self.suite.graphs[case.graph]
        values = {
            "matrix": str(self.data_root / graph.matrix_file),
            "source": case.source,
            "runs": self.suite.runs,
            "output_dir": str(app_output),
            "reduce": str(case.reduce_all_triangles).lower(),
        }
        if any("{binary}" in arg for arg in app.args):
            values["binary"] = str(self.binary_path(graph))
        return app.command(self.build_bin / app.name, values)

    @staticmethod
    def write_command(path, command):
        path.write_text(shlex.join(command) + "\n", encoding="utf-8")

    def nsys_command(self, app_command, report_base):
        child = [
            "env", f"CUDA_VISIBLE_DEVICES={self.device}", "GUNROCK_PROFILE_NVTX=1",
            *app_command,
        ]
        return [
            self.sudo, "-n", self.nsys, "profile",
            "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--stats=true", "--force-overwrite=true", "-o", str(report_base),
            *child,
        ]

    def app_result(self, app, app_output, graph):
        try:
            marker = app.args.index("--json_file")
        except ValueError:
            return
        if marker + 1 >= len(app.args):
            raise ProfileConfigError(f"{app.name}: --json_file has no filename")
        filename = app.args[marker + 1].format_map({
            "matrix": "", "source": 0, "runs": self.suite.runs,
            "output_dir": str(app_output), "reduce": "false",
        })
        result_file = app_output / filename
        if not result_file.is_file():
            raise ProfileConfigError(f"application result JSON is missing: {result_file}")
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileConfigError(f"cannot read application result {result_file}: {exc}") from exc
        if (int(result.get("num_vertices", -1)), int(result.get("num_edges", -1))) != (
            graph.vertices, graph.csr_edges
        ):
            raise ProfileConfigError(
                f"graph size mismatch for {result_file}: YAML expects "
                f"V={graph.vertices} E={graph.csr_edges}"
            )

    def write_kernel_table(self, report, trace_dir):
        csv_path = trace_dir / "kernel-launches.csv"
        raw_csv_path = trace_dir / "cuda-gpu-trace.csv"
        log_path = trace_dir / "nsys-stats.log"
        command = [
            self.sudo, "-n", self.nsys, "stats", "--quiet", "--force-export=true",
            "--report", "cuda_gpu_trace:mangled", "--format", "csv", "--output", "-",
            "--timeunit", "nsec", str(report),
        ]
        self.write_command(trace_dir / "stats-command.txt", command)
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            log_path.write_text(result.stderr + result.stdout, encoding="utf-8")
            raise ProfileConfigError(
                f"Nsight Systems could not export the kernel table (exit {result.returncode}); "
                f"see {log_path}"
            )
        log_path.write_text(result.stderr, encoding="utf-8")
        if not result.stdout.strip():
            raise ProfileConfigError("Nsight Systems produced an empty kernel launch table")
        raw_csv_path.write_text(result.stdout, encoding="utf-8")
        reader = csv.DictReader(io.StringIO(result.stdout))
        required = {"Start (ns)", "Duration (ns)", "GrdX", "Name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ProfileConfigError(
                "Nsight Systems CUDA trace CSV is missing kernel timing columns"
            )
        kernels = [row for row in reader if row.get("GrdX") and row.get("Name")]
        if not kernels:
            raise ProfileConfigError("Nsight Systems trace contains no CUDA kernel launches")
        kernels.sort(key=lambda row: int(row["Start (ns)"]))
        mangled_names = list(dict.fromkeys(row["Name"] for row in kernels))
        demangled_names = {name: name for name in mangled_names}
        cxxfilt = shutil.which("c++filt")
        if cxxfilt:
            demangled = subprocess.run(
                [cxxfilt, "-n"], input="\n".join(mangled_names) + "\n",
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if demangled.returncode == 0:
                values = demangled.stdout.splitlines()
                if len(values) == len(mangled_names):
                    demangled_names = dict(zip(mangled_names, values))
        kernel_fields = [
            *reader.fieldnames, "End (ns)", "kernel_launch_id",
            "matching_kernel_index", "demangled_name",
        ]
        matching_indices = {}
        for launch_id, row in enumerate(kernels, 1):
            try:
                row["End (ns)"] = str(int(row["Start (ns)"]) + int(row["Duration (ns)"]))
            except (TypeError, ValueError) as exc:
                raise ProfileConfigError("Nsight Systems emitted invalid kernel timestamps") from exc
            name = row["Name"]
            row["kernel_launch_id"] = str(launch_id)
            row["matching_kernel_index"] = str(matching_indices.get(name, 0))
            row["demangled_name"] = demangled_names[name]
            matching_indices[name] = matching_indices.get(name, 0) + 1
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=kernel_fields)
            writer.writeheader()
            writer.writerows(kernels)
        return csv_path, len(kernels)

    def run_case(self, app, case):
        case_root = self.case_dir(app, case)
        app_output = case_root / "application-output"
        trace_dir = case_root / "nsys-trace"
        app_output.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)

        graph = self.suite.graphs[case.graph]
        app_cmd = self.app_command(app, case, app_output)
        report_base = trace_dir / "report"
        report = Path(str(report_base) + ".nsys-rep")
        nsys_cmd = self.nsys_command(app_cmd, report_base)
        self.write_command(trace_dir / "command.txt", nsys_cmd)

        wrapper = [
            str(PROFILE_ONE), str(trace_dir / "run.log"), str(report), "--", *nsys_cmd,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device
        print(f"    {app.name} x {case.graph}: launching Nsys trace", flush=True)
        result = subprocess.run(wrapper, env=env, check=False)
        if result.returncode:
            raise ProfileConfigError(
                f"profile_one.sh failed with exit {result.returncode}; see {trace_dir / 'run.log'}"
            )

        self.app_result(app, app_output, graph)
        table, kernel_count = self.write_kernel_table(report, trace_dir)
        (trace_dir / "status.txt").write_text(
            f"status=OK\nreport={report}\nkernel_table={table}\n"
            f"kernel_launches={kernel_count}\nmetadata_dir={self.metadata_dir}\n",
            encoding="utf-8"
        )
        self.rows.append({
            "application": app.name,
            "graph": case.graph,
            "source": case.source,
            "load_balance": app.effective_load_balance(),
            "status": "OK",
            "report": str(report),
            "kernel_launch_table": str(table),
            "kernel_launches": kernel_count,
        })

    def write_summary(self):
        destination = self.metadata_dir / "profile-summary.csv"
        fields = [
            "application", "graph", "source", "load_balance", "status", "report",
            "kernel_launch_table", "kernel_launches",
        ]
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def run(self):
        cases = self.selected_cases()
        self.preflight(cases)
        self.prepare_run(cases)
        (self.metadata_dir / "environment.yaml").write_text(yaml.safe_dump({
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": "nsys-trace",
            "repository": str(REPO),
            "branch": subprocess.run(
                ["git", "-C", str(REPO), "branch", "--show-current"],
                text=True, stdout=subprocess.PIPE, check=False,
            ).stdout.strip(),
            "commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                text=True, stdout=subprocess.PIPE, check=False,
            ).stdout.strip(),
            "nsys": self.nsys,
            "build_bin": str(self.build_bin),
            "data_root": str(self.data_root),
            "cuda_visible_devices": self.device,
            "selected_cases": [f"{app.name}/{case.graph}/src-{case.source}" for app, case in cases],
        }, sort_keys=False), encoding="utf-8")

        failures = 0
        for index, (app, case) in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {app.name} x {case.graph} source={case.source}", flush=True)
            try:
                self.run_case(app, case)
            except Exception as exc:
                failures += 1
                case_root = self.case_dir(app, case)
                trace_dir = case_root / "nsys-trace"
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / "status.txt").write_text(
                    f"status=FAILED\nerror={exc}\nmetadata_dir={self.metadata_dir}\n",
                    encoding="utf-8",
                )
                self.rows.append({
                    "application": app.name,
                    "graph": case.graph,
                    "source": case.source,
                    "load_balance": app.effective_load_balance(),
                    "status": "FAILED",
                    "report": str(trace_dir / "report.nsys-rep"),
                    "kernel_launch_table": "",
                    "kernel_launches": 0,
                })
                print(f"    FAILED: {exc}", flush=True)
            self.write_summary()
        print(f"Completed: {len(cases) - failures} succeeded, {failures} failed", flush=True)
        print(f"Summary: {self.metadata_dir / 'profile-summary.csv'}", flush=True)
        return 1 if failures else 0


class NsysMetricsRunner(NsysTraceRunner):
    """Collect device-wide GPU metrics at progressively higher sample rates."""

    mode_dir = "nsys-metrics"

    def preflight(self, cases):
        super().preflight(cases)
        result = subprocess.run(
            [self.sudo, "-n", self.nsys, "profile",
             f"--gpu-metrics-devices={self.device}", "--gpu-metrics-set=help"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if result.returncode:
            raise ProfileConfigError(f"cannot list GPU metric sets:\n{result.stdout}")
        aliases = {
            line.split(":", 1)[0].strip()
            for line in result.stdout.splitlines() if ":" in line
        }
        if self.suite.gpu_metrics.metric_set not in aliases:
            raise ProfileConfigError(
                f"metric set {self.suite.gpu_metrics.metric_set!r} is unavailable for "
                f"GPU {self.device}; available sets: {', '.join(sorted(aliases))}"
            )

    def metrics_command(self, app_command, report_base, frequency_hz):
        child = [
            "env", f"CUDA_VISIBLE_DEVICES={self.device}", "GUNROCK_PROFILE_NVTX=1",
            *app_command,
        ]
        return [
            self.sudo, "-n", self.nsys, "profile",
            "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--stats=false", "--force-overwrite=true",
            f"--gpu-metrics-devices={self.device}",
            f"--gpu-metrics-set={self.suite.gpu_metrics.metric_set}",
            f"--gpu-metrics-frequency={frequency_hz}",
            "-o", str(report_base), *child,
        ]

    def export_sqlite(self, report, attempt_dir):
        database = attempt_dir / "report.sqlite"
        log = attempt_dir / "sqlite-export.log"
        command = [
            self.sudo, "-n", self.nsys, "export", "--type=sqlite",
            "--force-overwrite=true", "--quiet=true", "--output", str(database), str(report),
        ]
        self.write_command(attempt_dir / "sqlite-export-command.txt", command)
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log.write_text(result.stdout, encoding="utf-8")
        if result.returncode or not database.is_file() or not database.stat().st_size:
            raise ProfileConfigError(f"Nsight Systems SQLite export failed; see {log}")
        return database

    @staticmethod
    def table_columns(connection, table):
        return {
            row[1].lower(): row[1]
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }

    def write_metric_tables(self, database, attempt_dir, frequency_hz, run_log):
        metrics_csv = attempt_dir / "gpu-metrics.csv"
        summary_csv = attempt_dir / "gpu-metrics-summary.csv"
        summary_json = attempt_dir / "gpu-metrics-info.json"
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            tables = {
                row[0].lower() for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "gpu_metrics" not in tables or "target_info_gpu_metrics" not in tables:
                raise ProfileConfigError(
                    "SQLite export has no GPU_METRICS/TARGET_INFO_GPU_METRICS tables"
                )
            metric_columns = self.table_columns(connection, "GPU_METRICS")
            target_columns = self.table_columns(connection, "TARGET_INFO_GPU_METRICS")
            required_metrics = {"timestamp", "metricid", "value"}
            required_targets = {"metricid", "metricname"}
            if not required_metrics.issubset(metric_columns) or not required_targets.issubset(target_columns):
                raise ProfileConfigError("unrecognized Nsight Systems GPU metrics SQLite schema")

            device_column = metric_columns.get("typeid") or metric_columns.get("gpuid")
            raw_timestamp_column = metric_columns.get("rawtimestamp")
            selected_device = f'm."{device_column}"' if device_column else "''"
            selected_raw_timestamp = (
                f'm."{raw_timestamp_column}"' if raw_timestamp_column else 'm."timestamp"'
            )
            target_type_column = target_columns.get("typeid")
            source_column = target_columns.get("sourceid")
            if device_column and target_type_column:
                join_clause = f'm."{device_column}"=t."{target_type_column}"'
            else:
                join_clause = "1=1"
            if source_column and target_type_column:
                # Exclude synthetic EndTimestamp metadata rows, whose typeId
                # differs from the actual GPU metric sourceId.
                target_filter = f' WHERE t."{target_type_column}"=t."{source_column}"'
            else:
                target_filter = ""
            names = [row[0] for row in connection.execute(
                'SELECT DISTINCT "metricName" FROM "TARGET_INFO_GPU_METRICS" AS t' + target_filter
                + ' ORDER BY "metricName"'
            )]
            if not names:
                raise ProfileConfigError("GPU metric metadata contains no metric names")
            metric_units = {}
            for name in names:
                match = re.search(r"\[([^]]+)\]$", name)
                metric_units[name] = match.group(1) if match else ""
            unit_scales = {"MHz": 1e-6, "GB/s": 1e-9}
            query = (
                f'SELECT m."timestamp", {selected_raw_timestamp}, {selected_device}, '
                't."metricName", m."value" FROM "GPU_METRICS" AS m '
                f'JOIN "TARGET_INFO_GPU_METRICS" AS t ON m."metricId"=t."metricId" '
                f'AND {join_clause}' + target_filter + ' '
                'ORDER BY m."timestamp", ' + (f'm."{device_column}", ' if device_column else '') +
                't."metricName"'
            )
            writer_fields = ["timestamp_ns", "raw_timestamp_ns", "gpu_type_id", *names]
            counts = {name: 0 for name in names}
            sums = {name: 0.0 for name in names}
            minima = {name: None for name in names}
            maxima = {name: None for name in names}
            timestamps = set()
            sample_count = 0
            current_key = None
            current_raw_timestamp = None
            current_values = None

            with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=writer_fields)
                writer.writeheader()
                cursor = connection.execute(query)

                def write_sample():
                    nonlocal sample_count
                    if current_key is None:
                        return
                    timestamp, gpu_type = current_key
                    writer.writerow({
                        "timestamp_ns": timestamp,
                        "raw_timestamp_ns": current_raw_timestamp,
                        "gpu_type_id": gpu_type,
                        **current_values,
                    })
                    sample_count += 1

                for timestamp, raw_timestamp, gpu_type, name, value in cursor:
                    key = (int(timestamp), gpu_type or "")
                    if current_key != key:
                        write_sample()
                        current_key = key
                        current_raw_timestamp = int(raw_timestamp)
                        current_values = {}
                    timestamps.add(key[0])
                    if value is None:
                        continue
                    numeric = float(value) * unit_scales.get(metric_units[name], 1.0)
                    current_values[name] = numeric
                    counts[name] += 1
                    sums[name] += numeric
                    minima[name] = numeric if minima[name] is None else min(minima[name], numeric)
                    maxima[name] = numeric if maxima[name] is None else max(maxima[name], numeric)
                write_sample()
        finally:
            connection.close()

        if not sample_count:
            raise ProfileConfigError("GPU_METRICS contains no sample rows")
        with summary_csv.open("w", newline="", encoding="utf-8") as stream:
            fields = ["metric", "unit", "samples", "min", "mean", "max"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name in names:
                count = counts[name]
                writer.writerow({
                    "metric": name,
                    "unit": metric_units[name],
                    "samples": count,
                    "min": minima[name] if count else "",
                    "mean": sums[name] / count if count else "",
                    "max": maxima[name] if count else "",
                })

        ordered_timestamps = sorted(timestamps)
        intervals = [right - left for left, right in zip(ordered_timestamps, ordered_timestamps[1:])]
        expected_interval = 1_000_000_000 / frequency_hz
        median_interval = statistics.median(intervals) if intervals else 0
        gap_count = sum(interval > expected_interval * 3 for interval in intervals)
        gap_fraction = gap_count / len(intervals) if intervals else 1.0
        dram_metrics = sorted(
            name for name in names if ("dram" in name.lower() or "gpu memory" in name.lower())
            and any(word in name.lower() for word in ("bandwidth", "throughput"))
        )
        sm_metrics = sorted(
            name for name in names if "sm" in name.lower()
            or "warp" in name.lower() or "occupancy" in name.lower()
        )
        cache_metrics = sorted(
            name for name in names
            if "cache" in name.lower() or "l1" in name.lower() or "l2" in name.lower()
        )
        log_text = run_log.read_text(encoding="utf-8", errors="replace") if run_log.is_file() else ""
        overflow_warning = bool(re.search(
            r"(?i)(buffer\s+overflow|gpu metrics.{0,80}(missing data|overflow)|"
            r"(missing data|inconsistent data).{0,80}gpu metrics)", log_text
        ))
        cadence_ratio = median_interval / expected_interval if median_interval else 0
        usable = bool(
            dram_metrics and sm_metrics and len(ordered_timestamps) >= 3
            and not overflow_warning and gap_fraction <= 0.01
            and 0.25 <= cadence_ratio <= 2.5
        )
        info = {
            "requested_frequency_hz": frequency_hz,
            "metric_set": self.suite.gpu_metrics.metric_set,
            "gpu_metrics_scope": "device-wide; not attributable to one process/context",
            "value_normalization": "MHz=Hz/1e6; GB/s=bytes/s/1e9; other values are unchanged",
            "sample_rows": sample_count,
            "unique_timestamps": len(ordered_timestamps),
            "median_sample_interval_ns": median_interval,
            "observed_frequency_hz": 1_000_000_000 / median_interval if median_interval else None,
            "intervals_over_3x_requested_period": gap_count,
            "interval_gap_fraction": gap_fraction,
            "buffer_overflow_warning_in_log": overflow_warning,
            "dram_bandwidth_metrics": dram_metrics,
            "sm_or_warp_activity_metrics": sm_metrics,
            "cache_specific_metrics": cache_metrics,
            "usable_frequency": usable,
        }
        summary_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        return info, metrics_csv, summary_csv

    def run_case(self, app, case):
        case_root = self.case_dir(app, case)
        metrics_root = case_root / "nsys-metrics"
        attempts_root = metrics_root / "attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        graph = self.suite.graphs[case.graph]
        attempt_rows = []
        selected = None

        for frequency_hz in self.suite.gpu_metrics.frequencies_hz:
            attempt_dir = attempts_root / f"{frequency_hz:06d}Hz"
            app_output = attempt_dir / "application-output"
            app_output.mkdir(parents=True, exist_ok=True)
            report_base = attempt_dir / "report"
            report = Path(str(report_base) + ".nsys-rep")
            app_command = self.app_command(app, case, app_output)
            command = self.metrics_command(app_command, report_base, frequency_hz)
            self.write_command(attempt_dir / "command.txt", command)
            wrapper = [
                str(PROFILE_ONE), str(attempt_dir / "run.log"), str(report), "--", *command,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.device
            print(f"    {app.name} x {case.graph}: sampling at {frequency_hz / 1000:g} kHz", flush=True)
            result = subprocess.run(wrapper, env=env, check=False)
            if result.returncode:
                reason = f"profile_one.sh exited {result.returncode}"
                (attempt_dir / "status.txt").write_text(
                    f"status=FAILED\nfrequency_hz={frequency_hz}\nreason={reason}\n",
                    encoding="utf-8",
                )
                attempt_rows.append({
                    "frequency_hz": frequency_hz, "status": "FAILED", "usable": False,
                    "sample_rows": 0, "observed_frequency_hz": "", "gap_fraction": "",
                    "report": str(report), "reason": reason,
                })
                break

            try:
                self.app_result(app, app_output, graph)
                database = self.export_sqlite(report, attempt_dir)
                info, metrics_csv, summary_csv = self.write_metric_tables(
                    database, attempt_dir, frequency_hz, attempt_dir / "run.log"
                )
            except Exception as exc:
                reason = str(exc)
                (attempt_dir / "status.txt").write_text(
                    f"status=FAILED\nfrequency_hz={frequency_hz}\nreason={reason}\n",
                    encoding="utf-8",
                )
                attempt_rows.append({
                    "frequency_hz": frequency_hz, "status": "FAILED", "usable": False,
                    "sample_rows": 0, "observed_frequency_hz": "", "gap_fraction": "",
                    "report": str(report), "reason": reason,
                })
                break

            status = "USABLE" if info["usable_frequency"] else "GAPS_OR_WARNINGS"
            info["status"] = status
            (attempt_dir / "gpu-metrics-info.json").write_text(
                json.dumps(info, indent=2) + "\n", encoding="utf-8"
            )
            (attempt_dir / "status.txt").write_text(
                f"status={status}\nfrequency_hz={frequency_hz}\n"
                f"observed_frequency_hz={info['observed_frequency_hz']}\n"
                f"samples={info['unique_timestamps']}\nreport={report}\n"
                f"metrics_csv={metrics_csv}\nsummary_csv={summary_csv}\n",
                encoding="utf-8",
            )
            row = {
                "frequency_hz": frequency_hz,
                "status": status,
                "usable": info["usable_frequency"],
                "sample_rows": info["unique_timestamps"],
                "observed_frequency_hz": info["observed_frequency_hz"],
                "gap_fraction": info["interval_gap_fraction"],
                "report": str(report),
                "reason": "" if info["usable_frequency"] else "sampling gaps, warnings, or missing metrics",
            }
            attempt_rows.append(row)
            if info["usable_frequency"]:
                selected = {**row, "info": info, "metrics_csv": metrics_csv, "summary_csv": summary_csv}
            print(
                f"      {status}: {info['unique_timestamps']} samples, observed "
                f"{info['observed_frequency_hz']:.0f} Hz", flush=True,
            )

        attempts_csv = metrics_root / "attempts.csv"
        with attempts_csv.open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "frequency_hz", "status", "usable", "sample_rows", "observed_frequency_hz",
                "gap_fraction", "report", "reason",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(attempt_rows)

        if selected is None:
            raise ProfileConfigError(f"no usable GPU metrics frequency; see {attempts_csv}")
        selected_attempt = Path(selected["report"]).parent
        selected_info = selected["info"]
        (metrics_root / "selection.json").write_text(json.dumps({
            "status": "OK",
            "selected_frequency_hz": selected["frequency_hz"],
            "metric_set": self.suite.gpu_metrics.metric_set,
            "report": selected["report"],
            "metrics_csv": str(selected["metrics_csv"]),
            "summary_csv": str(selected["summary_csv"]),
            "unique_timestamps": selected_info["unique_timestamps"],
            "observed_frequency_hz": selected_info["observed_frequency_hz"],
            "dram_bandwidth_metrics": selected_info["dram_bandwidth_metrics"],
            "sm_or_warp_activity_metrics": selected_info["sm_or_warp_activity_metrics"],
            "cache_specific_metrics": selected_info["cache_specific_metrics"],
            "attempts_csv": str(attempts_csv),
        }, indent=2) + "\n", encoding="utf-8")
        (selected_attempt / "status.txt").write_text(
            (selected_attempt / "status.txt").read_text(encoding="utf-8") + "selected=true\n",
            encoding="utf-8",
        )
        self.rows.append({
            "application": app.name,
            "graph": case.graph,
            "source": case.source,
            "load_balance": app.effective_load_balance(),
            "status": "OK",
            "selected_frequency_hz": selected["frequency_hz"],
            "observed_frequency_hz": selected_info["observed_frequency_hz"],
            "samples": selected_info["unique_timestamps"],
            "report": selected["report"],
            "metrics_csv": str(selected["metrics_csv"]),
            "metrics_summary": str(selected["summary_csv"]),
        })

    def write_summary(self):
        destination = self.metadata_dir / "profile-summary.csv"
        fields = [
            "application", "graph", "source", "load_balance", "status",
            "selected_frequency_hz", "observed_frequency_hz", "samples",
            "report", "metrics_csv", "metrics_summary",
        ]
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def run(self):
        cases = self.selected_cases()
        self.preflight(cases)
        self.prepare_run(cases)
        (self.metadata_dir / "environment.yaml").write_text(yaml.safe_dump({
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": "nsys-metrics",
            "repository": str(REPO),
            "nsys": self.nsys,
            "build_bin": str(self.build_bin),
            "data_root": str(self.data_root),
            "cuda_visible_devices": self.device,
            "gpu_metric_device": self.device,
            "gpu_metric_scope": "device-wide; concurrent GPU activity is included",
            "gpu_metric_set": self.suite.gpu_metrics.metric_set,
            "frequency_candidates_hz": list(self.suite.gpu_metrics.frequencies_hz),
            "selected_cases": [f"{app.name}/{case.graph}/src-{case.source}" for app, case in cases],
        }, sort_keys=False), encoding="utf-8")

        failures = 0
        for index, (app, case) in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {app.name} x {case.graph} source={case.source}", flush=True)
            try:
                self.run_case(app, case)
            except Exception as exc:
                failures += 1
                metrics_root = self.case_dir(app, case) / "nsys-metrics"
                metrics_root.mkdir(parents=True, exist_ok=True)
                (metrics_root / "status.txt").write_text(
                    f"status=FAILED\nerror={exc}\n", encoding="utf-8"
                )
                self.rows.append({
                    "application": app.name,
                    "graph": case.graph,
                    "source": case.source,
                    "load_balance": app.effective_load_balance(),
                    "status": "FAILED",
                    "selected_frequency_hz": "",
                    "observed_frequency_hz": "",
                    "samples": 0,
                    "report": "",
                    "metrics_csv": "",
                    "metrics_summary": "",
                })
                print(f"    FAILED: {exc}", flush=True)
            self.write_summary()
        print(f"Completed: {len(cases) - failures} succeeded, {failures} failed", flush=True)
        print(f"Summary: {self.metadata_dir / 'profile-summary.csv'}", flush=True)
        return 1 if failures else 0


NCU_SECTIONS = (
    "SpeedOfLight", "MemoryWorkloadAnalysis", "WarpStateStats",
    "SchedulerStats", "LaunchStats", "Occupancy",
)
NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg", "sm__cycles_elapsed.avg.per_second",
    "lts__cycles_elapsed.avg", "lts__cycles_elapsed.avg.per_second",
    "dram__cycles_elapsed.avg", "dram__cycles_elapsed.avg.per_second",
    "l1tex__t_sectors.sum", "l1tex__t_sectors_lookup_hit.sum",
    "l1tex__t_sectors_lookup_miss.sum",
    "lts__t_requests.sum", "lts__t_requests_aperture_device_lookup_hit.sum",
    "lts__t_requests_aperture_device_lookup_miss.sum", "lts__d_sectors.sum",
    "dram__bytes.sum", "dram__bytes.sum.per_second",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
)
NCU_REQUIRED_RAW_METRICS = (
    "gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second",
    "lts__cycles_elapsed.avg.per_second", "dram__cycles_elapsed.avg.per_second",
    "l1tex__t_sectors_lookup_hit.sum", "l1tex__t_sectors_lookup_miss.sum",
    "lts__t_requests_aperture_device_lookup_hit.sum",
    "lts__t_requests_aperture_device_lookup_miss.sum", "lts__d_sectors.sum",
    "dram__bytes.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
)
NCU_SUMMARY_FIELDS = (
    "ncu_duration_ns", "sm_cycles", "sm_frequency_hz", "l2_frequency_hz",
    "dram_frequency_hz", "l1_sectors", "l1_hit_sectors", "l1_miss_sectors",
    "l1_hit_rate_pct",
    "l2_requests", "l2_hit_requests", "l2_miss_requests", "l2_data_sectors",
    "l2_hit_rate_pct", "l2_bandwidth_bytes_per_second", "dram_bytes",
    "dram_bandwidth_bytes_per_second",
    "sm_occupancy_pct", "active_warps_per_scheduler", "eligible_warps_per_scheduler",
    "issued_warps_per_scheduler",
    "stall_long_scoreboard_pct", "stall_short_scoreboard_pct", "stall_lg_throttle_pct",
    "stall_barrier_pct", "stall_wait_pct", "stall_not_selected_pct",
    "stall_selected_pct", "stall_branch_resolving_pct", "stall_drain_pct",
    "stall_membar_pct", "stall_math_pipe_throttle_pct", "stall_mio_throttle_pct",
    "stall_sleeping_pct", "stall_texture_throttle_pct", "quality_flags",
)
NCU_PROFILE_SUMMARY_FIELDS = (
    "application", "graph", "source", "load_balance", "kernel_launch_id",
    "matching_kernel_index", "demangled_kernel", "nsys_duration_ns", "ncu_report",
    "raw_metrics_csv", "verification", "status",
)
NCU_METRICS_SUMMARY_FIELDS = (
    "application", "graph", "source", "load_balance", "kernel_launch_id",
    "matching_kernel_index", "nsys_duration_ns", *NCU_SUMMARY_FIELDS,
    "ncu_report", "status",
)


class NcuRunner(NsysTraceRunner):
    """Profile explicit Nsys kernel-launch IDs with NCU application replay."""

    mode_dir = "ncu"

    def __init__(self, suite, args):
        super().__init__(suite, args)
        self.ncu = shutil.which("ncu")
        self.kernel_launch_table = args.kernel_launch_table.expanduser().resolve()
        self.launch_ids = args.kernel_launch_id

    def preflight(self, cases):
        if len(cases) != 1:
            raise ProfileConfigError(
                "NCU accepts launch IDs from one application/graph/source at a time; "
                "use --only app/graph/src-N to select one source"
            )
        if not self.ncu:
            raise ProfileConfigError("ncu was not found on PATH")
        if not self.launch_ids or any(value < 1 for value in self.launch_ids):
            raise ProfileConfigError("pass one or more positive --kernel-launch-id values")
        if len(set(self.launch_ids)) != len(self.launch_ids):
            raise ProfileConfigError("--kernel-launch-id values must be unique")
        if not self.kernel_launch_table.is_file():
            raise ProfileConfigError(f"kernel launch table is missing: {self.kernel_launch_table}")
        source_report = self.kernel_launch_table.parent / "report.nsys-rep"
        if not source_report.is_file():
            raise ProfileConfigError(f"source Nsys report is missing: {source_report}")
        super().preflight(cases)
        version = subprocess.run(
            [self.sudo, "-n", self.ncu, "--version"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if version.returncode:
            raise ProfileConfigError(f"cannot run Nsight Compute with sudo -n:\n{version.stdout}")
        section_result = subprocess.run(
            [self.sudo, "-n", self.ncu, "--list-sections"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if section_result.returncode:
            raise ProfileConfigError(f"cannot list NCU sections:\n{section_result.stdout}")
        missing_sections = [
            section for section in NCU_SECTIONS
            if not re.search(rf"(?m)^\s*{re.escape(section)}\s+", section_result.stdout)
        ]
        if missing_sections:
            raise ProfileConfigError(
                "installed NCU lacks required sections: " + ", ".join(missing_sections)
            )
        metric_result = subprocess.run(
            [self.sudo, "-n", self.ncu, "--query-metrics", "--metrics", ",".join(NCU_METRICS)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if metric_result.returncode:
            raise ProfileConfigError(
                "installed NCU/GPU cannot query the requested metrics:\n" + metric_result.stdout
            )
        source_suite = None
        trace_status = self.kernel_launch_table.parent / "status.txt"
        if trace_status.is_file():
            for line in trace_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("metadata_dir="):
                    metadata_dir = Path(line.partition("=")[2])
                    candidate = metadata_dir / "suite-config.yaml"
                    if candidate.is_file():
                        source_suite = candidate
                    break
        if source_suite is None:
            source_suite = next(
                (parent / "suite-config.yaml" for parent in self.kernel_launch_table.parents
                 if (parent / "suite-config.yaml").is_file()),
                None,
            )
        if source_suite:
            source_hash = hashlib.sha256(source_suite.read_bytes()).hexdigest()
            current_hash = hashlib.sha256(self.suite.path.read_bytes()).hexdigest()
            if source_hash != current_hash:
                raise ProfileConfigError(
                    "the Nsys launch table was captured from a different suite YAML; "
                    "capture a trace with the same application arguments/load-balance policy"
                )
        self.selected_launches(cases[0])

    @staticmethod
    def parse_launch_table(path):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [row for row in reader if row.get("Name") and row.get("GrdX")]
        rows.sort(key=lambda row: int(row["Start (ns)"]))
        next_by_name = {}
        launches = []
        for launch_id, row in enumerate(rows, 1):
            name = row["Name"]
            matching_index = int(row.get("matching_kernel_index") or next_by_name.get(name, 0))
            next_by_name[name] = matching_index + 1
            launches.append({
                "kernel_launch_id": launch_id,
                "matching_kernel_index": matching_index,
                "mangled_name": name,
                "demangled_name": row.get("demangled_name", name),
                "start_ns": int(row["Start (ns)"]),
                "duration_ns": int(row["Duration (ns)"]),
                "grid": [int(row[key]) for key in ("GrdX", "GrdY", "GrdZ")],
                "block": [int(row[key]) for key in ("BlkX", "BlkY", "BlkZ")],
            })
        return launches

    def selected_launches(self, case):
        launches = self.parse_launch_table(self.kernel_launch_table)
        by_id = {row["kernel_launch_id"]: row for row in launches}
        missing = [value for value in self.launch_ids if value not in by_id]
        if missing:
            raise ProfileConfigError(
                f"kernel launch ID(s) not found in {self.kernel_launch_table}: {missing}; "
                f"valid IDs are 1..{len(launches)}"
            )
        return [by_id[value] for value in self.launch_ids]

    def ncu_command(self, app_command, report_base, launch):
        return [
            self.sudo, "-n", self.ncu,
            "--devices", self.device,
            "--target-processes", "application-only",
            "--replay-mode", "application",
            "--app-replay-match", "grid",
            "--app-replay-mode", "relaxed",
            "--cache-control", "none",
            "--clock-control", "base",
            "--rename-kernels", "0",
            "--print-kernel-base", "mangled",
            "--kernel-name-base", "mangled",
            "--kernel-name", launch["mangled_name"],
            "--launch-skip", str(launch["matching_kernel_index"]),
            "--launch-count", "1",
            *[option for section in NCU_SECTIONS for option in ("--section", section)],
            "--metrics", ",".join(NCU_METRICS),
            "--force-overwrite", "-o", str(report_base), *app_command,
        ]

    def import_report(self, report, page, output_path):
        command = [
            self.sudo, "-n", self.ncu, "--import", str(report), "--page", page,
            "--print-units", "base", "--print-kernel-base", "mangled",
        ]
        if page == "raw":
            command.extend(("--csv",))
        else:
            command.extend(("--print-details", "all"))
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        output_path.write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise ProfileConfigError(f"NCU report import failed; see {output_path}")
        return result.stdout

    @staticmethod
    def verify_report(raw_text, details, launch, report, source_report, verification_path):
        try:
            rows = list(csv.DictReader(io.StringIO(raw_text)))
        except csv.Error as exc:
            raise ProfileConfigError(f"could not parse NCU raw metrics CSV: {exc}") from exc
        rows = [row for row in rows if row.get("Kernel Name")]
        errors = []
        if len(rows) != 1:
            errors.append(f"expected exactly one profiled kernel, found {len(rows)}")
        raw = rows[0] if rows else {}
        if raw and raw.get("Kernel Name") != launch["mangled_name"]:
            errors.append("reported mangled kernel does not match the requested kernel")
        expected_grid = f"({launch['grid'][0]}, {launch['grid'][1]}, {launch['grid'][2]})"
        expected_block = f"({launch['block'][0]}, {launch['block'][1]}, {launch['block'][2]})"
        if raw and raw.get("Grid Size") != expected_grid:
            errors.append(f"grid mismatch: expected {expected_grid}, got {raw.get('Grid Size')}")
        if raw and raw.get("Block Size") != expected_block:
            errors.append(f"block mismatch: expected {expected_block}, got {raw.get('Block Size')}")
        missing_metrics = [
            metric for metric in NCU_REQUIRED_RAW_METRICS
            if metric not in raw or raw.get(metric, "") in ("", "N/A", "n/a")
        ]
        if missing_metrics:
            errors.append("missing required raw metrics: " + ", ".join(missing_metrics))
        required_sections = {
            "GPU Speed Of Light Throughput", "Memory Workload Analysis",
            "Warp State Statistics", "Scheduler Statistics", "Launch Statistics",
            "Occupancy",
        }
        missing_sections = sorted(section for section in required_sections if section not in details)
        if missing_sections:
            errors.append("missing NCU sections: " + ", ".join(missing_sections))
        duration_ns = None
        try:
            duration_ns = float(raw["gpu__time_duration.sum"])
        except (KeyError, TypeError, ValueError):
            pass
        quality_flags = []
        if duration_ns is not None and duration_ns < 20_000:
            quality_flags.append("under_20us_multipass_sensitive")
        out_of_range_fields = []
        for field, raw_value in raw.items():
            if not field.endswith(".pct"):
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                continue
            if parsed < 0.0 or parsed > 100.0:
                out_of_range_fields.append(field)
        if out_of_range_fields:
            quality_flags.append("out_of_range_rate")
        verification = {
            "passed": not errors,
            "kernel_launch_id": launch["kernel_launch_id"],
            "matching_kernel_index": launch["matching_kernel_index"],
            "exact_mangled_kernel": launch["mangled_name"],
            "demangled_kernel": launch["demangled_name"],
            "nsys_start_ns": launch["start_ns"],
            "nsys_duration_ns": launch["duration_ns"],
            "ncu_duration_ns": duration_ns,
            "ncu_report": str(report),
            "source_nsys_report": str(source_report),
            "ncu_kernel_rows": len(rows),
            "grid": launch["grid"],
            "block": launch["block"],
            "required_sections": sorted(required_sections),
            "missing_sections": missing_sections,
            "required_raw_metrics": list(NCU_REQUIRED_RAW_METRICS),
            "missing_raw_metrics": missing_metrics,
            "raw_values": raw,
            "quality_flags": quality_flags,
            "out_of_range_rate_fields": out_of_range_fields,
            "errors": errors,
        }
        verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        if errors:
            raise ProfileConfigError("NCU verification failed: " + "; ".join(errors))

    @staticmethod
    def metric_summary(raw_text):
        rows = [row for row in csv.DictReader(io.StringIO(raw_text)) if row.get("Kernel Name")]
        if len(rows) != 1:
            raise ProfileConfigError(f"expected one NCU metrics row for summary, found {len(rows)}")
        raw = rows[0]

        def value(name):
            text_value = raw.get(name, "")
            try:
                return float(text_value)
            except (TypeError, ValueError):
                return ""

        duration_ns = value("gpu__time_duration.sum")
        duration_seconds = duration_ns / 1e9 if isinstance(duration_ns, float) else 0.0
        l2_sectors = value("lts__d_sectors.sum")
        dram_bytes = value("dram__bytes.sum")
        l2_bytes_per_second = (
            l2_sectors * 32.0 / duration_seconds
            if duration_seconds and isinstance(l2_sectors, float) else ""
        )
        dram_bytes_per_second = value("dram__bytes.sum.per_second")
        if not isinstance(dram_bytes_per_second, float):
            dram_bytes_per_second = (
                dram_bytes / duration_seconds
                if duration_seconds and isinstance(dram_bytes, float) else ""
            )
        metric_map = {
            "ncu_duration_ns": "gpu__time_duration.sum",
            "sm_cycles": "sm__cycles_elapsed.avg",
            "sm_frequency_hz": "sm__cycles_elapsed.avg.per_second",
            "l2_frequency_hz": "lts__cycles_elapsed.avg.per_second",
            "dram_frequency_hz": "dram__cycles_elapsed.avg.per_second",
            "l1_sectors": "l1tex__t_sectors.sum",
            "l1_hit_sectors": "l1tex__t_sectors_lookup_hit.sum",
            "l1_miss_sectors": "l1tex__t_sectors_lookup_miss.sum",
            "l1_hit_rate_pct": "l1tex__t_sector_hit_rate.pct",
            "l2_requests": "lts__t_requests.sum",
            "l2_hit_requests": "lts__t_requests_aperture_device_lookup_hit.sum",
            "l2_miss_requests": "lts__t_requests_aperture_device_lookup_miss.sum",
            "l2_data_sectors": "lts__d_sectors.sum",
            "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
            "dram_bytes": "dram__bytes.sum",
            "sm_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
            "active_warps_per_scheduler": "smsp__warps_active.avg.per_cycle_active",
            "eligible_warps_per_scheduler": "smsp__warps_eligible.avg.per_cycle_active",
            "issued_warps_per_scheduler": "smsp__issue_active.avg.per_cycle_active",
            "stall_long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "stall_short_scoreboard_pct": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "stall_lg_throttle_pct": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
            "stall_barrier_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "stall_wait_pct": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
            "stall_not_selected_pct": "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
            "stall_selected_pct": "smsp__warp_issue_stalled_selected_per_warp_active.pct",
            "stall_branch_resolving_pct": "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
            "stall_drain_pct": "smsp__warp_issue_stalled_drain_per_warp_active.pct",
            "stall_membar_pct": "smsp__warp_issue_stalled_membar_per_warp_active.pct",
            "stall_math_pipe_throttle_pct": "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
            "stall_mio_throttle_pct": "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
            "stall_sleeping_pct": "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
            "stall_texture_throttle_pct": "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
        }
        summary = {field: value(metric) for field, metric in metric_map.items()}
        summary["l2_bandwidth_bytes_per_second"] = l2_bytes_per_second
        summary["dram_bandwidth_bytes_per_second"] = dram_bytes_per_second
        quality_flags = []
        if isinstance(duration_ns, float) and duration_ns < 20_000:
            quality_flags.append("under_20us_multipass_sensitive")
        for name, raw_value in raw.items():
            if not name.endswith(".pct"):
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                continue
            if parsed < 0.0 or parsed > 100.0:
                quality_flags.append("out_of_range_rate")
                break
        summary["quality_flags"] = ";".join(quality_flags)
        return summary

    def run_launch(self, app, case, launch):
        case_root = self.case_dir(app, case) / "ncu" / f"launch-{launch['kernel_launch_id']:06d}"
        app_output = case_root / "application-output"
        ncu_dir = case_root / "ncu"
        app_output.mkdir(parents=True, exist_ok=True)
        ncu_dir.mkdir(parents=True, exist_ok=True)
        report_base = ncu_dir / "report"
        report = Path(str(report_base) + ".ncu-rep")
        app_command = self.app_command(app, case, app_output)
        command = self.ncu_command(app_command, report_base, launch)
        self.write_command(ncu_dir / "command.txt", command)
        wrapper = [
            str(PROFILE_ONE), str(ncu_dir / "run.log"), str(report), "--", *command,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device
        env["GUNROCK_PROFILE_NVTX"] = "1"
        display_name = launch["demangled_name"]
        if len(display_name) > 180:
            display_name = display_name[:177] + "..."
        print(
            f"    NCU kernel launch {launch['kernel_launch_id']} "
            f"(matching index {launch['matching_kernel_index']}): {display_name}", flush=True,
        )
        result = subprocess.run(wrapper, env=env, check=False)
        if result.returncode:
            raise ProfileConfigError(
                f"profile_one.sh failed with exit {result.returncode}; see {ncu_dir / 'run.log'}"
            )
        self.app_result(app, app_output, self.suite.graphs[case.graph])
        raw_path = ncu_dir / "metrics-raw.csv"
        details_path = ncu_dir / "report-details.txt"
        raw = self.import_report(report, "raw", raw_path)
        details = self.import_report(report, "details", details_path)
        source_report = self.kernel_launch_table.parent / "report.nsys-rep"
        self.verify_report(
            raw, details, launch, report, source_report, ncu_dir / "verification.json",
        )
        summary = self.metric_summary(raw)
        (ncu_dir / "status.txt").write_text(
            f"status=OK\nkernel_launch_id={launch['kernel_launch_id']}\n"
            f"matching_kernel_index={launch['matching_kernel_index']}\n"
            f"report={report}\n", encoding="utf-8"
        )
        return {
            "application": app.name,
            "graph": case.graph,
            "source": case.source,
            "load_balance": app.effective_load_balance(),
            "kernel_launch_id": launch["kernel_launch_id"],
            "matching_kernel_index": launch["matching_kernel_index"],
            "mangled_kernel": launch["mangled_name"],
            "demangled_kernel": launch["demangled_name"],
            "nsys_duration_ns": launch["duration_ns"],
            **summary,
            "ncu_report": str(report),
            "raw_metrics_csv": str(raw_path),
            "verification": str(ncu_dir / "verification.json"),
            "status": "OK",
        }

    def run(self):
        cases = self.selected_cases()
        self.preflight(cases)
        app, case = cases[0]
        launches = self.selected_launches(cases[0])
        self.prepare_run(cases)
        shutil.copyfile(self.kernel_launch_table, self.metadata_dir / "source-kernel-launches.csv")
        (self.metadata_dir / "environment.yaml").write_text(yaml.safe_dump({
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": "ncu",
            "ncu": self.ncu,
            "ncu_version": subprocess.run(
                [self.sudo, "-n", self.ncu, "--version"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            ).stdout.strip(),
            "repository": str(REPO),
            "build_bin": str(self.build_bin),
            "data_root": str(self.data_root),
            "cuda_visible_devices": self.device,
            "replay_mode": "application",
            "cache_control": "none (natural application cache history)",
            "clock_control": "base",
            "source_nsys_report": str(self.kernel_launch_table.parent / "report.nsys-rep"),
            "source_kernel_launch_table": str(self.kernel_launch_table),
            "kernel_launch_id_convention": "1-based position among CUDA kernel launches in Nsys start-time order",
            "selected_cases": [f"{app.name}/{case.graph}/src-{case.source}"],
            "selected_kernel_launch_ids": self.launch_ids,
            "sections": list(NCU_SECTIONS),
            "explicit_metrics": list(NCU_METRICS),
        }, sort_keys=False), encoding="utf-8")
        self.rows = []
        failures = 0
        for index, launch in enumerate(launches, 1):
            print(f"[{index}/{len(launches)}] {app.name} x {case.graph}", flush=True)
            try:
                self.rows.append(self.run_launch(app, case, launch))
            except Exception as exc:
                failures += 1
                self.rows.append({
                    "application": app.name, "graph": case.graph, "source": case.source,
                    "load_balance": app.effective_load_balance(),
                    "kernel_launch_id": launch["kernel_launch_id"],
                    "matching_kernel_index": launch["matching_kernel_index"],
                    "mangled_kernel": launch["mangled_name"],
                    "demangled_kernel": launch["demangled_name"],
                    "nsys_duration_ns": launch["duration_ns"],
                    "ncu_report": "", "raw_metrics_csv": "", "verification": "",
                    "status": f"FAILED: {exc}",
                })
                print(f"    FAILED: {exc}", flush=True)
            with (self.metadata_dir / "profile-summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=NCU_PROFILE_SUMMARY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
            with (self.metadata_dir / "ncu-metrics-summary.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=NCU_METRICS_SUMMARY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
        print(f"Completed: {len(launches) - failures} succeeded, {failures} failed", flush=True)
        print(f"Summary: {self.metadata_dir / 'profile-summary.csv'}", flush=True)
        return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="YAML-driven Gunrock Nsys/NCU profiling suite")
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--only",
        help="comma-separated application, application/graph, or application/graph/src-N selections",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output folders for the selected application/graph/source/mode cases",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data-8/Graph-Datasets/mtx"))
    parser.add_argument(
        "--build-bin", type=Path,
        default=Path(os.environ.get("GUNROCK_BUILD_BIN", REPO / "build" / "bin")),
        help="directory containing app executables (default: GUNROCK_BUILD_BIN or <repo>/build/bin)",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--kernel-launch-table", type=Path,
        help="kernel-launches.csv from a matching nsys-trace capture (required for --mode ncu)",
    )
    parser.add_argument(
        "--kernel-launch-id", type=int, action="append", default=[],
        help="1-based CUDA kernel launch ID from --kernel-launch-table; repeat to profile multiple launches",
    )
    args = parser.parse_args()

    try:
        suite = Suite.load(args.suite_config)
        runner_type = {
            "nsys-trace": NsysTraceRunner,
            "nsys-metrics": NsysMetricsRunner,
            "ncu": NcuRunner,
        }.get(args.mode)
        if runner_type is None:
            parser.error(f"mode {args.mode!r} is not implemented yet")
        if args.mode == "ncu" and not args.kernel_launch_table:
            parser.error("--kernel-launch-table is required for --mode ncu")
        return runner_type(suite, args).run()
    except ProfileConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
