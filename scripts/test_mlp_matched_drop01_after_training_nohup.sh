#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s TRAIN_PID\n' "$0" >&2
    exit 1
fi

TRAIN_PID="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$ROOT/runs/mlp_matched_drop01_seed42_test_queue.log"

setsid nohup bash -c '
    set -euo pipefail
    train_pid="$1"
    root="$2"
    script_dir="$3"

    while kill -0 "$train_pid" 2>/dev/null; do
        sleep 60
    done

    history="$root/runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/training_history.json"
    if [[ ! -f "$history" ]]; then
        printf "Multi-task training did not finish successfully: %s is missing\n" "$history" >&2
        exit 1
    fi

    printf "[%s] Starting single-task test\n" "$(date "+%F %T")"
    bash "$script_dir/test_single_mlp_matched_drop01_mse_seed42.sh"

    printf "[%s] Starting multi-task test\n" "$(date "+%F %T")"
    bash "$script_dir/test_multitask_mlp_matched_drop01_norm_w_denoise_b005_seed42.sh"

    printf "[%s] All tests complete\n" "$(date "+%F %T")"
' bash "$TRAIN_PID" "$ROOT" "$SCRIPT_DIR" > "$LOG" 2>&1 < /dev/null &

printf 'Test queue started: PID=%s\nLog: %s\n' "$!" "$LOG"
