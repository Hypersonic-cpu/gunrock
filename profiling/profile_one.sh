#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: profile_one.sh LOG_FILE RESULT_FILE -- COMMAND [ARG ...]

Run one profiling command, save its combined stdout/stderr to LOG_FILE, and
require RESULT_FILE to be non-empty when the command exits successfully.
USAGE
}

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  usage >&2
  exit 2
fi

log_file=$1
result_file=$2
shift 3

mkdir -p "$(dirname -- "$log_file")" "$(dirname -- "$result_file")"

if "$@" >"$log_file" 2>&1; then
  status=0
else
  status=$?
fi

if (( status != 0 )); then
  printf 'profile_one: command exited with status %d; log: %s\n' "$status" "$log_file" >&2
  cat "$log_file" >&2
  exit "$status"
fi

if [[ ! -s "$result_file" ]]; then
  printf 'profile_one: expected result is missing or empty: %s\n' "$result_file" >&2
  cat "$log_file" >&2
  exit 1
fi

printf 'profile_one: OK %s\n' "$result_file"
