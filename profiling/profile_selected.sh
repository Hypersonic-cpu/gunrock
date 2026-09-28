#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RESULTS_ROOT=${RESULTS_ROOT:-}
DATA_ROOT=${DATA_ROOT:-}
REFERENCE_RESULTS_ROOT=${REFERENCE_RESULTS_ROOT:-}
PROFILE_STAGE=${PROFILE_STAGE:-all}
PROFILE_MODES=${PROFILE_MODES:-natural,cold}
GPU_VISIBLE=${GPU_VISIBLE:-}
PROFILE_CONFIG=${PROFILE_CONFIG:-$SCRIPT_DIR/profile_config.json}
ONLY=${ONLY:-}

usage() {
  sed -n '1,42p' "$0"
}

while (( $# )); do
  case "$1" in
    --stage) PROFILE_STAGE=$2; shift 2 ;;
    --modes) PROFILE_MODES=$2; shift 2 ;;
    --results-root) RESULTS_ROOT=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --reference-root) REFERENCE_RESULTS_ROOT=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE_STAGE" in all|validate|native|nsys|ncu) ;; *) echo "invalid stage" >&2; exit 2 ;; esac
[[ -f "$PROFILE_CONFIG" ]] || { echo "missing config: $PROFILE_CONFIG" >&2; exit 1; }
[[ -n "$DATA_ROOT" ]] || { echo "set DATA_ROOT from AGENTS.local.md" >&2; exit 2; }
[[ -n "$RESULTS_ROOT" ]] || { echo "set RESULTS_ROOT from AGENTS.local.md or --results-root" >&2; exit 2; }
[[ -n "$REFERENCE_RESULTS_ROOT" ]] || { echo "set REFERENCE_RESULTS_ROOT from AGENTS.local.md" >&2; exit 2; }
[[ -n "$GPU_VISIBLE" ]] || { echo "set GPU_VISIBLE from AGENTS.local.md" >&2; exit 2; }
mkdir -p "$RESULTS_ROOT"
cp -f "$PROFILE_CONFIG" "$RESULTS_ROOT/profile_config.json"
cp -Lf "$REPO_ROOT/PROFILING_TASKS.md" "$RESULTS_ROOT/PROFILING_TASKS.md"

printf 'GPU_VISIBLE=%q DATA_ROOT=%q RESULTS_ROOT=%q REFERENCE_RESULTS_ROOT=%q PROFILE_CONFIG=%q %q --stage %q --modes %q\n' \
  "$GPU_VISIBLE" "$DATA_ROOT" "$RESULTS_ROOT" "$REFERENCE_RESULTS_ROOT" "$PROFILE_CONFIG" \
  "$SCRIPT_DIR/profile_selected.sh" "$PROFILE_STAGE" "$PROFILE_MODES" \
  > "$RESULTS_ROOT/full-suite-command.txt"

selected() {
  [[ -z "$ONLY" ]] && return 0
  local candidate=$1
  IFS=',' read -r -a wanted <<< "$ONLY"
  for value in "${wanted[@]}"; do
    [[ "$value" == "$candidate" ]] && return 0
  done
  return 1
}

mapfile -t CASE_ROWS < <(python3 - "$PROFILE_CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
for item in config["cases"]:
    validation = item.get("validation", {})
    reason = str(validation.get("reason", "")).replace("\t", " ").replace("\n", " ")
    print("\t".join((
        item["algorithm"],
        item["graph"],
        str(item.get("source", 0)),
        validation.get("mode", "required"),
        str(item.get("profile_repetitions", config.get("profile_repetitions_default", 1))),
        "true" if item.get("reduce_all_triangles", False) else "false",
        reason,
    )))
PY
)
(( ${#CASE_ROWS[@]} > 0 )) || { echo "profile config has no cases" >&2; exit 1; }

for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r algorithm graph source validation_mode repetitions reduce reason <<< "$row"
  case_id="$algorithm/$graph"
  selected "$case_id" || continue
  echo "== profiling $algorithm/$graph source=$source validation=$validation_mode repetitions=$repetitions =="
  PROFILE_STAGE="$PROFILE_STAGE" PROFILE_MODES="$PROFILE_MODES" GPU_VISIBLE="$GPU_VISIBLE" \
    DATA_ROOT="$DATA_ROOT" \
    RESULTS_ROOT="$RESULTS_ROOT" PROFILE_CONFIG="$PROFILE_CONFIG" \
    "$SCRIPT_DIR/profile_one.sh" "$algorithm" "$graph" "$source" \
    --stage "$PROFILE_STAGE" --modes "$PROFILE_MODES" \
    --validation-mode "$validation_mode" --validation-reason "$reason" \
    --profile-repetitions "$repetitions" --reduce-all-triangles "$reduce"
done

python3 - "$RESULTS_ROOT" "$PROFILE_STAGE" "$PROFILE_MODES" \
  "$PROFILE_CONFIG" "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
modes = [mode for mode in sys.argv[3].split(",") if mode]
config_path = Path(sys.argv[4])
repo_root = Path(sys.argv[5])
config = json.loads(config_path.read_text(encoding="utf-8"))
cases = config["cases"]
roles = ("p10", "p50", "p90", "peak", "tail")

validation_rows = []
required_pass = True
validation_complete = True
all_complete = True
for item in cases:
    algorithm = item["algorithm"]
    graph = item["graph"]
    source = item.get("source", 0)
    validation_policy = item.get("validation", {})
    required = validation_policy.get("mode", "required") == "required"
    case = root / "full-suite" / "gunrock-v2" / algorithm / graph / f"src-{source}"
    valid = root / "validation" / "gunrock-v2" / algorithm / graph / f"src-{source}"
    status_path = valid / "metadata" / "status.txt"
    status_text = status_path.read_text(encoding="utf-8") if status_path.is_file() else ""
    status = "PASS" if "validation=PASS" in status_text else (
        "SKIPPED" if "validation=SKIPPED" in status_text else "MISSING"
    )
    validation_json = valid / "native" / "validation.json"
    skipped_json = valid / "metadata" / "validation-skipped.json"
    pass_case = status == "PASS" and validation_json.is_file()
    if pass_case:
        pass_case = "Number of errors : 0" in (valid / "native" / "run.log").read_text(encoding="utf-8")
    skip_case = status == "SKIPPED" and skipped_json.is_file()
    if required:
        required_pass &= pass_case
        validation_complete &= pass_case
    else:
        validation_complete &= skip_case
    reason = validation_policy.get("reason", "")
    if skipped_json.is_file():
        try:
            reason = json.loads(skipped_json.read_text(encoding="utf-8")).get("reason", reason)
        except json.JSONDecodeError:
            pass
    row = {
        "algorithm": algorithm,
        "graph": graph,
        "source": source,
        "case_dir": str(valid),
        "validation_required": required,
        "validation_status": status,
        "passed": True if pass_case else (False if required else None),
        "accepted_for_profiling": pass_case if required else skip_case,
        "skipped": not required,
        "skip_reason": reason if not required else "",
    }
    validation_rows.append(row)

    if stage == "all":
        required_artifacts = [
            case / "native" / "native.json",
            case / "nsys" / "report.nsys-rep",
            case / "nsys" / "report.sqlite",
            case / "invocations" / "analysis.json",
            case / "invocations" / "selection.tsv",
            case / "summary" / "ncu-five-point.csv",
        ]
        required_artifacts += [
            case / "ncu" / mode / role / "verification.json"
            for mode in modes for role in roles
        ]
        all_complete &= all(path.is_file() for path in required_artifacts)

(root / "validation").mkdir(parents=True, exist_ok=True)
validation_payload = {
    "passed": required_pass,
    "complete": validation_complete,
    "required_case_count": sum(row["validation_required"] for row in validation_rows),
    "skipped_case_count": sum(row["skipped"] for row in validation_rows),
    "cases": validation_rows,
    "policy": config.get("validation_policy", ""),
}
(root / "validation" / "validation-pass.json").write_text(
    json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8"
)
skip_count = validation_payload["skipped_case_count"]
validation_text = "Validation result: " + ("PASS" if required_pass else "FAIL") + "\n"
validation_text += f"Required validation cases passed: {sum(row['passed'] is True for row in validation_rows)}\n"
validation_text += f"Explicitly skipped validation cases: {skip_count}\n"
(root / "validation" / "validation-pass.txt").write_text(validation_text, encoding="utf-8")

if stage == "all" and all_complete and validation_complete:
    marker = {
        "complete": True,
        "repository": str(repo_root),
        "stage": stage,
        "modes": modes,
        "cases": [
            {
                "algorithm": item["algorithm"],
                "graph": item["graph"],
                "source": item.get("source", 0),
                "validation": item.get("validation", {}),
            }
            for item in cases
        ],
        "validation_skipped_cases": [
            {"algorithm": row["algorithm"], "graph": row["graph"], "reason": row["skip_reason"]}
            for row in validation_rows if row["skipped"]
        ],
    }
    (root / "full-suite").mkdir(parents=True, exist_ok=True)
    (root / "full-suite" / "suite-complete.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    (root / "full-suite" / "suite-complete.txt").write_text("Suite result: COMPLETE\n", encoding="utf-8")
else:
    for stale in (root / "full-suite" / "suite-complete.json", root / "full-suite" / "suite-complete.txt"):
        stale.unlink(missing_ok=True)
PY

summary_ready=true
for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r algorithm graph source validation_mode repetitions reduce reason <<< "$row"
  if [[ ! -s "$RESULTS_ROOT/full-suite/gunrock-v2/$algorithm/$graph/src-$source/summary/ncu-five-point.csv" ]]; then
    summary_ready=false
  fi
done
if [[ "$summary_ready" == true && ( "$PROFILE_STAGE" == all || "$PROFILE_STAGE" == ncu ) ]]; then
  python3 "$SCRIPT_DIR/summarize_architecture.py" \
    --root "$RESULTS_ROOT" --group full-suite \
    --reference-root "$REFERENCE_RESULTS_ROOT"
fi

echo "Selected v2 profiling stage complete: $PROFILE_STAGE"
