#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SINGLE_RUN="$ROOT/runs/single_t60_mlp_matched_drop01_mse_seed42"
MULTI_RUN="$ROOT/runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42"

mkdir -p "$SINGLE_RUN" "$MULTI_RUN"

printf '[%s] Starting single-task matched MLP dropout=0.1\n' "$(date '+%F %T')"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHON="$PYTHON" \
    bash "$ROOT/train.sh" --no-conda \
    -c "$ROOT/configs/t60_single/mlp_matched_drop01_mse_seed42.yaml" \
    > "$SINGLE_RUN/nohup.log" 2>&1

printf '[%s] Single-task complete; starting multi-task matched MLP dropout=0.1\n' "$(date '+%F %T')"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHON="$PYTHON" \
    bash "$ROOT/train_multitask.sh" --no-conda \
    -c "$ROOT/configs/t60_multitask/mlp_matched_drop01_mse_norm_w_denoise_b005_seed42.yaml" \
    > "$MULTI_RUN/nohup.log" 2>&1

printf '[%s] All matched MLP dropout=0.1 training complete\n' "$(date '+%F %T')"
