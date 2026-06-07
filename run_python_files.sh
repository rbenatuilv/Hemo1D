#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Edit this list to choose which Python files to run.
PYTHON_FILES=(
  "examples/real_network.py"
  "examples/stent_vessel_coupling.py"
)

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-run_logs/$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"

pids=()
files=()
logs=()

cleanup() {
  if ((${#pids[@]})); then
    echo "Stopping running jobs..."
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for file in "${PYTHON_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing file: $file" >&2
    exit 1
  fi
done

for file in "${PYTHON_FILES[@]}"; do
  log_file="$LOG_DIR/${file//\//__}.log"
  echo "Starting $file -> $log_file"
  "$PYTHON_BIN" "$file" >"$log_file" 2>&1 &

  pids+=("$!")
  files+=("$file")
  logs+=("$log_file")
done

status=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  file="${files[$index]}"
  log_file="${logs[$index]}"

  if wait "$pid"; then
    echo "OK: $file"
  else
    exit_code=$?
    echo "FAILED: $file exited with code $exit_code. See $log_file" >&2
    status=1
  fi
done

if ((status == 0)); then
  echo "All jobs finished successfully. Logs are in $LOG_DIR"
else
  echo "One or more jobs failed. Logs are in $LOG_DIR" >&2
fi

exit "$status"
