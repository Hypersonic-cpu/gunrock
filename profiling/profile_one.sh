#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DATA_ROOT=${DATA_ROOT:-}
RESULTS_ROOT=${RESULTS_ROOT:-}
GPU_VISIBLE=${GPU_VISIBLE:-}
PROFILE_STAGE=${PROFILE_STAGE:-all}
PROFILE_MODES=${PROFILE_MODES:-natural,cold}
PROFILE_CONFIG=${PROFILE_CONFIG:-$SCRIPT_DIR/profile_config.json}
NEAR_EMPTY_FRACTION=${NEAR_EMPTY_FRACTION:-}
VALIDATION_MODE=${VALIDATION_MODE:-required}
VALIDATION_REASON=${VALIDATION_REASON:-}
PROFILE_REPETITIONS=${PROFILE_REPETITIONS:-1}
REDUCE_ALL_TRIANGLES=${REDUCE_ALL_TRIANGLES:-false}

usage() {
  sed -n '1,58p' "$0"
}

if (( $# < 2 )); then
  usage
  exit 2
fi
ALGORITHM=$1
GRAPH=$2
shift 2
SOURCE=0
if (( $# >= 1 )) && [[ $1 != --* ]]; then
  SOURCE=$1
  shift
fi

while (( $# )); do
  case "$1" in
    --stage) PROFILE_STAGE=$2; shift 2 ;;
    --modes) PROFILE_MODES=$2; shift 2 ;;
    --results-root) RESULTS_ROOT=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --near-empty-fraction) NEAR_EMPTY_FRACTION=$2; shift 2 ;;
    --validation-mode) VALIDATION_MODE=$2; shift 2 ;;
    --validation-reason) VALIDATION_REASON=$2; shift 2 ;;
    --profile-repetitions) PROFILE_REPETITIONS=$2; shift 2 ;;
    --reduce-all-triangles) REDUCE_ALL_TRIANGLES=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ALGORITHM" in bfs|sssp|bc|pr|tc) ;; *) echo "unsupported algorithm: $ALGORITHM" >&2; exit 2 ;; esac
case "$GRAPH" in soc-orkut|indochina-2004|roadNet-CA|hollywood-2009) ;; *) echo "unsupported graph: $GRAPH" >&2; exit 2 ;; esac
case "$PROFILE_STAGE" in all|validate|native|nsys|ncu) ;; *) echo "invalid stage" >&2; exit 2 ;; esac
case "$VALIDATION_MODE" in required|skipped) ;; *) echo "invalid validation mode" >&2; exit 2 ;; esac
[[ "$PROFILE_REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "profile repetitions must be a positive integer" >&2; exit 2; }
[[ -f "$PROFILE_CONFIG" ]] || { echo "missing config: $PROFILE_CONFIG" >&2; exit 1; }
[[ -n "$DATA_ROOT" ]] || { echo "set DATA_ROOT from AGENTS.local.md or --data-root" >&2; exit 2; }
[[ -n "$RESULTS_ROOT" ]] || { echo "set RESULTS_ROOT from AGENTS.local.md or --results-root" >&2; exit 2; }
[[ -n "$GPU_VISIBLE" ]] || { echo "set GPU_VISIBLE from AGENTS.local.md" >&2; exit 2; }

INPUT="$DATA_ROOT/$GRAPH.mtx"
APP="$REPO_ROOT/build/bin/$ALGORITHM"
BENCH="$REPO_ROOT/build/bin/${ALGORITHM}_bench"
[[ -f "$INPUT" && -x "$APP" && -x "$BENCH" ]] || { echo "input or selected binaries are missing" >&2; exit 1; }

# The active host's profiler privilege policy is documented in
# AGENTS.local.md. Resolve the profilers with `which` and invoke the resolved
# paths through sudo; the recorded command files retain that spelling.
NSYS=$(which nsys)
NCU=$(which ncu)
sudo -n "$NSYS" --version >/dev/null
sudo -n "$NCU" --version >/dev/null
NCU_SECTIONS=$(sudo -n "$NCU" --list-sections)
for section in SpeedOfLight MemoryWorkloadAnalysis WarpStateStats SchedulerStats LaunchStats; do
  printf '%s\n' "$NCU_SECTIONS" | rg -q "^${section}[[:space:]]" || {
    echo "required NCU section is unavailable: $section" >&2
    exit 1
  }
done

if [[ -z "$NEAR_EMPTY_FRACTION" ]]; then
  if [[ "$GRAPH" == soc-orkut ]]; then NEAR_EMPTY_FRACTION=0.0; else NEAR_EMPTY_FRACTION=0.05; fi
fi

CASE_DIR="$RESULTS_ROOT/full-suite/gunrock-v2/$ALGORITHM/$GRAPH/src-$SOURCE"
VALIDATION_DIR="$RESULTS_ROOT/validation/gunrock-v2/$ALGORITHM/$GRAPH/src-$SOURCE"
NATIVE_DIR="$CASE_DIR/native"
NSYS_DIR="$CASE_DIR/nsys"
INVOCATION_DIR="$CASE_DIR/invocations"
METADATA_DIR="$CASE_DIR/metadata"
mkdir -p "$NATIVE_DIR" "$NSYS_DIR" "$INVOCATION_DIR" "$METADATA_DIR"
cp -f "$PROFILE_CONFIG" "$METADATA_DIR/profile_config.json"

case "$GRAPH" in
  soc-orkut) EXPECTED_V=2997166; EXPECTED_E=212698418 ;;
  indochina-2004) EXPECTED_V=7414866; EXPECTED_E=194109311 ;;
  roadNet-CA) EXPECTED_V=1971281; EXPECTED_E=5533214 ;;
  hollywood-2009) EXPECTED_V=1139905; EXPECTED_E=115031232 ;;
esac

BENCH_ARGS=(--market "$INPUT" --devices 0 --profile)
if [[ "$ALGORITHM" == bc ]]; then
  BENCH_ARGS+=(--src="$SOURCE")
fi
if (( PROFILE_REPETITIONS > 1 )); then
  BENCH_ARGS+=(--profile-runs "$PROFILE_REPETITIONS")
fi
if [[ "$ALGORITHM" == tc ]]; then
  BENCH_ARGS+=(--reduce "$REDUCE_ALL_TRIANGLES")
fi

write_command() {
  local destination=$1
  shift
  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_VISIBLE"
    printf '%q ' "$@"
    printf '\n'
  } > "$destination"
}

check_graph_json() {
  python3 - "$1" "$EXPECTED_V" "$EXPECTED_E" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if int(data.get("num_vertices", -1)) != int(sys.argv[2]):
    raise SystemExit("vertex-count mismatch")
if int(data.get("num_edges", -1)) != int(sys.argv[3]):
    raise SystemExit("CSR-edge-count mismatch")
PY
}

write_validation_skip() {
  mkdir -p "$VALIDATION_DIR/metadata"
  cp -f "$PROFILE_CONFIG" "$VALIDATION_DIR/metadata/profile_config.json"
  python3 - "$VALIDATION_DIR/metadata/validation-skipped.json" \
    "$ALGORITHM" "$GRAPH" "$SOURCE" "$INPUT" "$VALIDATION_REASON" <<'PY'
import json
import sys
from pathlib import Path

destination, algorithm, graph, source, input_path, reason = sys.argv[1:]
Path(destination).write_text(json.dumps({
    "algorithm": algorithm,
    "graph": graph,
    "source": int(source),
    "input": input_path,
    "status": "SKIPPED",
    "skipped": True,
    "reason": reason or "validation explicitly disabled by profiling manifest",
}, indent=2) + "\n", encoding="utf-8")
PY
  printf 'validation=SKIPPED\nalgorithm=%s\ngraph=%s\nsource=%s\nreason=%s\ninput=%s\n' \
    "$ALGORITHM" "$GRAPH" "$SOURCE" "$VALIDATION_REASON" "$INPUT" \
    > "$VALIDATION_DIR/metadata/status.txt"
}

run_validation() {
  if [[ "$VALIDATION_MODE" == skipped ]]; then
    write_validation_skip
    return
  fi
  case "$ALGORITHM" in
    bfs|sssp) ;;
    *) echo "required validation is not implemented for $ALGORITHM; mark it skipped in the manifest" >&2; exit 1 ;;
  esac
  local out="$VALIDATION_DIR/native"
  mkdir -p "$out" "$VALIDATION_DIR/metadata"
  cp -f "$PROFILE_CONFIG" "$VALIDATION_DIR/metadata/profile_config.json"
  local cmd=("$APP" --market "$INPUT" --src "$SOURCE" --num_runs 1
    --advance_load_balance block_mapped --validate --export_metrics
    --json_dir "$out" --json_file validation.json)
  write_command "$VALIDATION_DIR/metadata/command.txt" "${cmd[@]}"
  if [[ ! -s "$out/validation.json" || ! -s "$out/run.log" ]] || ! rg -q 'Number of errors : 0' "$out/run.log"; then
    CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" "${cmd[@]}" 2>&1 | tee "$out/run.log"
  fi
  rg -q 'Number of errors : 0' "$out/run.log" || { echo "validation failed" >&2; exit 1; }
  check_graph_json "$out/validation.json"
  printf 'validation=PASS\nalgorithm=%s\ngraph=%s\nsource=%s\nvertices=%s\ncsr_entries=%s\ninput=%s\n' \
    "$ALGORITHM" "$GRAPH" "$SOURCE" "$EXPECTED_V" "$EXPECTED_E" "$INPUT" > "$VALIDATION_DIR/metadata/status.txt"
}

native_json_is_usable() {
  [[ -s "$NATIVE_DIR/native.json" ]] || return 1
  python3 - "$NATIVE_DIR/native.json" "$EXPECTED_V" "$EXPECTED_E" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if len(data.get("process_times", [])) != 10:
    raise SystemExit("native timing does not contain ten samples")
if int(data.get("num_vertices", -1)) != int(sys.argv[2]):
    raise SystemExit("native vertex-count mismatch")
if int(data.get("num_edges", -1)) != int(sys.argv[3]):
    raise SystemExit("native CSR-edge-count mismatch")
PY
}

native_command() {
  case "$ALGORITHM" in
    bfs|sssp|bc)
      printf '%s\0' "$APP" --market "$INPUT" --src "$SOURCE" --num_runs 10 \
        --advance_load_balance block_mapped --export_metrics \
        --json_dir "$NATIVE_DIR" --json_file native.json ;;
    pr)
      printf '%s\0' "$APP" --market "$INPUT" --num_runs 10 \
        --advance_load_balance block_mapped --export_metrics \
        --json_dir "$NATIVE_DIR" --json_file native.json ;;
    tc)
      printf '%s\0' "$APP" --market "$INPUT" --reduce "$REDUCE_ALL_TRIANGLES" ;;
  esac
}

read_native_command() {
  local -n destination=$1
  destination=()
  while IFS= read -r -d '' value; do destination+=("$value"); done < <(native_command)
}

write_tc_native_json() {
  python3 - "$NATIVE_DIR/native.json" "$NATIVE_DIR" "$EXPECTED_V" "$EXPECTED_E" \
    "$ALGORITHM" "$GRAPH" "$INPUT" "$PROFILE_REPETITIONS" "$REDUCE_ALL_TRIANGLES" <<'PY'
import json
import re
import shlex
import statistics
import sys
from pathlib import Path

destination, directory, vertices, edges, algorithm, graph, input_path, repetitions, reduce = sys.argv[1:]
times = []
pattern = re.compile(r"GPU Elapsed Time\s*:\s*([-+0-9.eE]+)\s*\(ms\)")
for path in sorted(Path(directory).glob("run-[0-9][0-9].log")):
    matches = pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise SystemExit(f"missing GPU elapsed time in {path}")
    times.append(float(matches[-1]))
if len(times) != 10:
    raise SystemExit(f"expected ten TC timing logs, found {len(times)}")
data = {
    "engine": "Essentials",
    "primitive": algorithm,
    "graph_type": "market",
    "num_edges": int(edges),
    "num_vertices": int(vertices),
    "srcs": [],
    "tags": [],
    "process_times": times,
    "graph_file": input_path,
    "avg_process_time": statistics.mean(times),
    "stddev_process_time": statistics.pstdev(times),
    "min_process_time": min(times),
    "max_process_time": max(times),
    "command_line": shlex.join(["build/bin/tc", "--market", input_path, "--reduce", reduce]),
    "native_generated_by": "profiling/profile_one.sh",
    "profile_repetitions": int(repetitions),
    "reduce_all_triangles": reduce == "true",
    "native_validation": "SKIPPED",
}
Path(destination).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

run_tc_native() {
  local cmd=()
  read_native_command cmd
  write_command "$NATIVE_DIR/command.txt" "${cmd[@]}"
  if ! native_json_is_usable; then
    rm -f "$NATIVE_DIR/native.json" "$NATIVE_DIR/run.log"
    : > "$NATIVE_DIR/run.log"
    for index in $(seq -w 1 10); do
      local log="$NATIVE_DIR/run-$index.log"
      if ! CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" "${cmd[@]}" > "$log" 2>&1; then
        cat "$log" >&2
        echo "TC native run failed at sample $index" >&2
        exit 1
      fi
      cat "$log" >> "$NATIVE_DIR/run.log"
    done
    write_tc_native_json
  fi
  native_json_is_usable
}

run_native() {
  if [[ "$ALGORITHM" == tc ]]; then
    run_tc_native
    return
  fi
  local cmd=()
  read_native_command cmd
  write_command "$NATIVE_DIR/command.txt" "${cmd[@]}"
  if ! native_json_is_usable; then
    CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" "${cmd[@]}" 2>&1 | tee "$NATIVE_DIR/run.log"
  fi
  native_json_is_usable
}

write_nsys_command() {
  {
    printf 'sudo "$(which nsys)" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true --force-overwrite=true -o %q env CUDA_VISIBLE_DEVICES=%q GUNROCK_PROFILE_NVTX=1 ' \
      "$NSYS_DIR/report" "$GPU_VISIBLE"
    printf '%q ' "$BENCH" "${BENCH_ARGS[@]}"
    printf '\n'
  } > "$NSYS_DIR/command.txt"
  printf 'sudo "$(which nsys)" stats --quiet --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum %q > %q\n' \
    "$NSYS_DIR/report.nsys-rep" "$NSYS_DIR/stats.txt" > "$NSYS_DIR/stats-command.txt"
}

run_nsys() {
  local report="$NSYS_DIR/report.nsys-rep"
  local database="$NSYS_DIR/report.sqlite"
  local recapture=false
  if [[ -s "$report" && -s "$NSYS_DIR/command.txt" ]]; then
    if (( PROFILE_REPETITIONS > 1 )) && ! rg -q -- "--profile-runs[[:space:]]+$PROFILE_REPETITIONS([[:space:]]|$)" "$NSYS_DIR/command.txt"; then
      recapture=true
    elif (( PROFILE_REPETITIONS == 1 )) && rg -q -- '--profile-runs([[:space:]]|$)' "$NSYS_DIR/command.txt"; then
      recapture=true
    fi
    if [[ "$ALGORITHM" == bc ]] && ! rg -q -- "--src(=|[[:space:]])$SOURCE([[:space:]]|$)" "$NSYS_DIR/command.txt"; then
      recapture=true
    fi
    if [[ "$ALGORITHM" == tc ]] && ! rg -q -- "--reduce[[:space:]]+$REDUCE_ALL_TRIANGLES([[:space:]]|$)" "$NSYS_DIR/command.txt"; then
      recapture=true
    fi
  fi
  write_nsys_command
  if [[ "$recapture" == true ]]; then
    rm -f "$NSYS_DIR/report.nsys-rep" "$NSYS_DIR/report.sqlite" "$NSYS_DIR/stats.txt" \
      "$NSYS_DIR/export.log" "$NSYS_DIR/run.log" "$INVOCATION_DIR/cuda_gpu_trace.csv" \
      "$INVOCATION_DIR/analysis.json" "$INVOCATION_DIR/analysis.txt" \
      "$INVOCATION_DIR/kernel-rankings.csv" "$INVOCATION_DIR/dominant-invocations.csv" \
      "$INVOCATION_DIR/selection.csv" "$INVOCATION_DIR/selection.tsv"
    rm -rf "$CASE_DIR/ncu"
  fi
  if [[ ! -s "$report" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" sudo -n "$NSYS" profile \
      --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true \
      --force-overwrite=true -o "$NSYS_DIR/report" \
      env CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" GUNROCK_PROFILE_NVTX=1 "$BENCH" \
      "${BENCH_ARGS[@]}" 2>&1 | tee "$NSYS_DIR/run.log"
  fi
  [[ -s "$report" ]] || { echo "Nsys report missing" >&2; exit 1; }
  if [[ ! -s "$database" ]]; then
    sudo -n "$NSYS" export --type=sqlite --output="$database" --force-overwrite=true "$report" > "$NSYS_DIR/export.log" 2>&1
  fi
  if [[ ! -s "$NSYS_DIR/stats.txt" ]]; then
    sudo -n "$NSYS" stats --quiet --force-export=true \
      --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
      "$report" > "$NSYS_DIR/stats.txt" 2>&1
  fi
  sudo -n "$NSYS" stats --quiet --force-export=true --report cuda_gpu_trace:mangled \
    --format csv --output - --timeunit nsec "$report" > "$INVOCATION_DIR/cuda_gpu_trace.csv"
  python3 "$SCRIPT_DIR/analyze_nsys_invocations.py" --database "$database" \
    --out-dir "$INVOCATION_DIR" --algorithm "$ALGORITHM" --graph "$GRAPH" \
    --near-empty-fraction "$NEAR_EMPTY_FRACTION"
}

write_ncu_command() {
  local destination=$1
  local target=$2
  local mode=$3
  local skip=$4
  local filter=$5
  local report=$6
  {
    printf 'sudo "$(which ncu)" --target-processes %q ' "$target"
    if [[ "$mode" == natural ]]; then
      printf '%s ' '--replay-mode' 'application' '--app-replay-match' 'grid' '--app-replay-mode' 'relaxed' '--cache-control' 'none'
    else
      printf '%s ' '--replay-mode' 'kernel' '--cache-control' 'all'
    fi
    printf '%s ' '--clock-control' 'base' '--disable-extra-suffixes' '--rename-kernels' '0' '--print-kernel-base' 'mangled'
    printf '%s ' '--section' 'SpeedOfLight' '--section' 'MemoryWorkloadAnalysis' '--section' 'WarpStateStats' '--section' 'SchedulerStats' '--section' 'LaunchStats'
    printf '%s %q %s %q %s %q %s %q ' \
      '--kernel-name-base' 'mangled' '--kernel-name' "$filter" '--launch-skip' "$skip" '--launch-count' '1'
    printf '%s ' '--force-overwrite'
    printf '%s %q ' '--export' "$report"
    printf '%q ' "$BENCH" "${BENCH_ARGS[@]}"
    printf '\n'
  } > "$destination"
}

run_ncu_point() {
  local mode=$1
  local role=$2
  local skip=$3
  local filter=$4
  local point="$CASE_DIR/ncu/$mode/$role"
  local report="$point/report.ncu-rep"
  mkdir -p "$point" "$CASE_DIR/ncu/commands"
  local common=()
  if [[ "$mode" == natural ]]; then
    common=(--replay-mode application --app-replay-match grid --app-replay-mode relaxed --cache-control none)
  else
    common=(--replay-mode kernel --cache-control all)
  fi
  common+=(--clock-control base --disable-extra-suffixes --rename-kernels 0 --print-kernel-base mangled
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section LaunchStats
    --kernel-name-base mangled --kernel-name "$filter" --launch-skip="$skip" --launch-count=1
    --force-overwrite --export "$point/report" "$BENCH" "${BENCH_ARGS[@]}")
  write_ncu_command "$point/command.txt" application-only "$mode" "$skip" "$filter" "$point/report"
  cp -f "$point/command.txt" "$CASE_DIR/ncu/commands/$mode-$role.txt"
  if [[ ! -s "$report" ]]; then
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" sudo -n "$NCU" --target-processes application-only "${common[@]}" 2>&1 | tee "$point/ncu.log"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )) || [[ ! -s "$report" ]]; then
      write_ncu_command "$point/command.txt" all "$mode" "$skip" "$filter" "$point/report"
      cp -f "$point/command.txt" "$CASE_DIR/ncu/commands/$mode-$role.txt"
      printf 'application-only failed (status=%s); reran with all\n' "$status" > "$point/target-processes-fallback.txt"
      set +e
      CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" sudo -n "$NCU" --target-processes all "${common[@]}" 2>&1 | tee "$point/ncu-all.log"
      status=${PIPESTATUS[0]}
      set -e
      (( status == 0 )) || { echo "Ncu failed for $ALGORITHM/$GRAPH/$mode/$role" >&2; exit 1; }
    fi
  fi
  [[ -s "$report" ]] || { echo "Ncu report missing" >&2; exit 1; }
  if [[ ! -s "$point/metrics-raw.csv" ]]; then
    sudo -n "$NCU" --import "$report" --page raw --csv --print-units base --print-kernel-base mangled > "$point/metrics-raw.csv"
  fi
  if [[ ! -s "$point/details.txt" ]]; then
    sudo -n "$NCU" --import "$report" --page details --print-kernel-base mangled > "$point/details.txt"
  fi
  python3 "$SCRIPT_DIR/verify_ncu_invocation.py" --selection "$INVOCATION_DIR/selection.tsv" \
    --invocations "$INVOCATION_DIR/dominant-invocations.csv" --role "$role" \
    --raw "$point/metrics-raw.csv" --details "$point/details.txt" --command "$point/command.txt" \
    --ncu-report "$report" --out "$point/verification.json"
}

run_ncu() {
  [[ -s "$INVOCATION_DIR/selection.tsv" ]] || { echo "missing invocation selection" >&2; exit 1; }
  IFS=',' read -r -a modes <<< "$PROFILE_MODES"
  for mode in "${modes[@]}"; do
    [[ "$mode" == natural || "$mode" == cold ]] || { echo "invalid Ncu mode" >&2; exit 2; }
    while IFS=$'\t' read -r role target index skip kernel demangled mangled filter rest; do
      [[ -n "$role" ]] || continue
      run_ncu_point "$mode" "$role" "$skip" "$filter"
    done < <(tail -n +2 "$INVOCATION_DIR/selection.tsv")
    printf 'complete\n' > "$CASE_DIR/ncu/$mode/status.txt"
  done
}

write_environment_metadata() {
  python3 - "$METADATA_DIR/environment.json" "$REPO_ROOT" "$GPU_VISIBLE" "$NSYS" "$NCU" "$PROFILE_CONFIG" \
    "$VALIDATION_MODE" "$VALIDATION_REASON" "$PROFILE_REPETITIONS" "$REDUCE_ALL_TRIANGLES" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

destination = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
gpu_visible = sys.argv[3]
nsys = sys.argv[4]
ncu = sys.argv[5]
profile_config = sys.argv[6]
validation_mode = sys.argv[7]
validation_reason = sys.argv[8]
profile_repetitions = int(sys.argv[9])
reduce_all_triangles = sys.argv[10] == "true"


def capture(*command):
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
    except OSError as error:
        return f"unavailable: {error}"
    return result.stdout.strip()


native_path = destination.parent.parent / "native" / "native.json"
native = {}
if native_path.is_file():
    try:
        native = json.loads(native_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        native = {}

data = {
    "repository": str(repo_root),
    "branch": capture("git", "-C", str(repo_root), "branch", "--show-current"),
    "git_commit_sha": capture("git", "-C", str(repo_root), "rev-parse", "HEAD"),
    "cuda_visible_devices": gpu_visible,
    "gpu_query": capture("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,clocks.sm,clocks.mem", "--format=csv,noheader"),
    "cuda_compiler": capture("nvcc", "--version"),
    "cmake": capture("cmake", "--version"),
    "nsys_path": nsys,
    "nsys_version": capture("sudo", "-n", nsys, "--version"),
    "ncu_path": ncu,
    "ncu_version": capture("sudo", "-n", ncu, "--version"),
    "ncu_required_sections": ["SpeedOfLight", "MemoryWorkloadAnalysis", "WarpStateStats", "SchedulerStats", "LaunchStats"],
    "ncu_required_sections_checked": True,
    "native_gpuinfo": native.get("gpuinfo", {}),
    "native_compiler": native.get("compiler", ""),
    "native_compiler_version": native.get("compiler_version", ""),
    "profile_config": profile_config,
    "validation_mode": validation_mode,
    "validation_reason": validation_reason,
    "profile_repetitions": profile_repetitions,
    "reduce_all_triangles": reduce_all_triangles,
    "profiler_command_policy": {
        "nsys": 'sudo "$(which nsys)"',
        "ncu": 'sudo "$(which ncu)"',
        "capture_mode": "host-native; no sandbox",
    },
}
destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

case "$PROFILE_STAGE" in all|validate) run_validation ;; esac
case "$PROFILE_STAGE" in all|native) run_native ;; esac
case "$PROFILE_STAGE" in all|nsys) run_nsys ;; esac
case "$PROFILE_STAGE" in all|ncu) run_ncu ;; esac

write_environment_metadata

if [[ -s "$CASE_DIR/ncu/natural/status.txt" && -s "$CASE_DIR/ncu/cold/status.txt" ]]; then
  python3 "$SCRIPT_DIR/summarize_invocation_profiles.py" --case-dir "$CASE_DIR"
  printf 'complete\n' > "$METADATA_DIR/status.txt"
else
  printf 'stage=%s\n' "$PROFILE_STAGE" > "$METADATA_DIR/status.txt"
fi
printf 'Completed v2 %s/%s source=%s stage=%s\n' "$ALGORITHM" "$GRAPH" "$SOURCE" "$PROFILE_STAGE"
