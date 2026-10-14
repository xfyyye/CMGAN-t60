#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/test_multitask.py" \
    --config "$ROOT/configs/t60_multitask/mlp_matched_drop01_mse_norm_w_denoise_b005_seed42.yaml" \
    --model_path "$ROOT/runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/best_model.pth" \
    --save_dir "$ROOT/runs/mlp_matched_drop01_multitask_norm_w_denoise_b005_seed42/test" \
    --gpu_id 0
