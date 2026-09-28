#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/configs/t60_multitask/mlp_mse_norm_w_denoise_b005_seed42.yaml"
RUN_DIR="$REPO_ROOT/runs/mlp_multitask_norm_w_denoise_b005_seed42"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python environment not found: %s\n' "$PYTHON" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTHON="$PYTHON" \
setsid nohup bash "$REPO_ROOT/train_multitask.sh" \
    --no-conda \
    -c "$CONFIG" \
    > "$RUN_DIR/nohup.log" 2>&1 < /dev/null &

printf 'Training started: PID=%s\nLog: %s\n' "$!" "$RUN_DIR/nohup.log"
