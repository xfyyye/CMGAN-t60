#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUEUE_LOG="$ROOT/runs/mlp_matched_seed42_queue.log"

mkdir -p "$ROOT/runs"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
setsid nohup bash "$SCRIPT_DIR/run_mlp_matched_seed42_queue.sh" \
    > "$QUEUE_LOG" 2>&1 < /dev/null &

printf 'Training queue started: PID=%s\nLog: %s\n' "$!" "$QUEUE_LOG"
