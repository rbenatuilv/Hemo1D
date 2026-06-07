#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON:-python}"
STATUS_INTERVAL="${STATUS_INTERVAL:-30}"
LOG_DIR="${LOG_DIR:-analysis/outputs/convergence_logs}"
T_END="${T_END:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

mkdir -p "$LOG_DIR"

declare -a JOB_LABELS=()
declare -a JOB_COMMANDS=()
declare -a JOB_PIDS=()
declare -a JOB_STATUS=()
declare -a JOB_LOGS=()
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
    echo "Interrupted; stopping running convergence jobs..."
    cleanup_jobs
    exit 130
}

trap handle_interrupt INT TERM

add_job() {
    local script="$1"
    local method_label="$2"
    local outlet="$3"
    shift 3

    local stem="${script%.py}"
    local label="${stem}:${method_label}:${outlet}"
    local log_file="${LOG_DIR}/${stem}_${method_label}_${outlet}.log"
    local cmd=("$PYTHON_BIN" "analysis/${script}" "$@" "--outlet-model" "$outlet")

    if [[ -n "$T_END" ]]; then
        cmd+=("--t-end" "$T_END")
    fi

    JOB_LABELS+=("$label")
    JOB_COMMANDS+=("$(printf "%q " "${cmd[@]}")")
    JOB_LOGS+=("$log_file")
}

for script in convergence_single.py convergence_three_vessel.py; do
    for outlet in nonreflecting capillary; do
        add_job "$script" "cg" "$outlet" --method cg
        add_job "$script" "dg_lxf" "$outlet" --method dg --dg-flux lxf
        add_job "$script" "dg_hll" "$outlet" --method dg --dg-flux hll
    done
done

print_status() {
    printf "\n[%s] Convergence job status\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "%-48s %-12s %s\n" "job" "status" "log"
    printf "%-48s %-12s %s\n" "---" "------" "---"

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

        printf "%-48s %-12s %s\n" "$label" "$status" "${JOB_LOGS[$index]}"
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

echo "Launching ${#JOB_LABELS[@]} convergence jobs."
echo "Logs: ${LOG_DIR}"
echo "Set STATUS_INTERVAL=<seconds> to change polling cadence."
echo "Set T_END=<seconds> to override each script's default t_end."

for index in "${!JOB_LABELS[@]}"; do
    log_file="${JOB_LOGS[$index]}"
    cmd="${JOB_COMMANDS[$index]}"

    {
        printf "Command: %s\n\n" "$cmd"
        eval "$cmd"
    } >"$log_file" 2>&1 &

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
    echo "All convergence jobs completed."
else
    echo "Some convergence jobs failed. Check the log files above."
fi

trap - INT TERM
exit "$failed"
