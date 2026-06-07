#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON:-python}"
STATUS_INTERVAL="${STATUS_INTERVAL:-30}"
LOG_DIR="${LOG_DIR:-examples/outputs/aortic_endograft_logs}"
T_END="${T_END:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

mkdir -p "$LOG_DIR"

declare -a JOB_LABELS=()
declare -a JOB_METHODS=()
declare -a JOB_FLUXES=()
declare -a JOB_OUTPUT_DIRS=()
declare -a JOB_LOGS=()
declare -a JOB_PIDS=()
declare -a JOB_STATUS=()
SHUTTING_DOWN=0

cleanup_jobs() {
    local pid

    for pid in "${JOB_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    for pid in "${JOB_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

handle_interrupt() {
    if [[ "$SHUTTING_DOWN" -eq 1 ]]; then
        return
    fi

    SHUTTING_DOWN=1
    echo
    echo "Interrupted; stopping running aortic-endograft jobs..."
    cleanup_jobs
    exit 130
}

trap handle_interrupt INT TERM

add_job() {
    local method="$1"
    local flux="$2"

    local method_label
    local output_dir

    if [[ "$method" == "dg" ]]; then
        method_label="dg_${flux}"
        output_dir="examples/outputs/aortic_endograft/method_${method}_${flux}"
    else
        method_label="cg"
        output_dir="examples/outputs/aortic_endograft/method_${method}"
    fi

    JOB_LABELS+=("aortic_endograft:${method_label}")
    JOB_METHODS+=("$method")
    JOB_FLUXES+=("$flux")
    JOB_OUTPUT_DIRS+=("$output_dir")
    JOB_LOGS+=("${LOG_DIR}/aortic_endograft_${method_label}.log")
}

run_aortic_endograft() {
    local method="$1"
    local flux="$2"
    local output_dir="$3"

    "$PYTHON_BIN" - "$method" "$flux" "$output_dir" "$T_END" <<'PY'
import importlib.util
import sys
from pathlib import Path

method, flux, output_dir, t_end = sys.argv[1:]

spec = importlib.util.spec_from_file_location(
    "aortic_endograft_example",
    Path("examples/aortic_endograft.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module.METHOD = method
module.DG_FLUX = flux
module.OUTPUT_DIR = Path(output_dir)
if t_end:
    module.T_END = float(t_end)

module.main()
PY
}

add_job cg lxf
add_job dg hll
add_job dg lxf

print_status() {
    printf "\n[%s] Aortic-endograft job status\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "%-30s %-12s %s\n" "job" "status" "log"
    printf "%-30s %-12s %s\n" "---" "------" "---"

    local index label pid status
    for index in "${!JOB_LABELS[@]}"; do
        label="${JOB_LABELS[$index]}"
        pid="${JOB_PIDS[$index]}"
        status="${JOB_STATUS[$index]}"

        if [[ "$status" == "ongoing" ]] && ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid"; then
                JOB_STATUS[$index]="completed"
            else
                JOB_STATUS[$index]="failed"
            fi
            status="${JOB_STATUS[$index]}"
        fi

        printf "%-30s %-12s %s\n" "$label" "$status" "${JOB_LOGS[$index]}"
    done
}

all_finished() {
    local status
    for status in "${JOB_STATUS[@]}"; do
        if [[ "$status" == "ongoing" ]]; then
            return 1
        fi
    done
    return 0
}

echo "Launching ${#JOB_LABELS[@]} aortic-endograft jobs."
echo "Logs: ${LOG_DIR}"
echo "Set STATUS_INTERVAL=<seconds> to change polling cadence."
echo "Set T_END=<seconds> to override examples/aortic_endograft.py's default t_end."

for index in "${!JOB_LABELS[@]}"; do
    {
        printf "Method: %s\n" "${JOB_METHODS[$index]}"
        printf "DG flux: %s\n" "${JOB_FLUXES[$index]}"
        printf "Output: %s\n\n" "${JOB_OUTPUT_DIRS[$index]}"
        run_aortic_endograft \
            "${JOB_METHODS[$index]}" \
            "${JOB_FLUXES[$index]}" \
            "${JOB_OUTPUT_DIRS[$index]}"
    } >"${JOB_LOGS[$index]}" 2>&1 &

    JOB_PIDS[$index]="$!"
    JOB_STATUS[$index]="ongoing"
done

print_status

while ! all_finished; do
    sleep "$STATUS_INTERVAL"
    print_status
done

print_status

failed=0
for status in "${JOB_STATUS[@]}"; do
    if [[ "$status" == "failed" ]]; then
        failed=1
    fi
done

if [[ "$failed" -eq 0 ]]; then
    echo "All aortic-endograft jobs completed."
else
    echo "Some aortic-endograft jobs failed. Check the log files above."
fi

trap - INT TERM
exit "$failed"
