#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/train.py" \
    --config "$ROOT/configs/t60_single/mlp_matched_mse_seed42.yaml" \
    --test_only \
    --model_path "$ROOT/runs/single_t60_mlp_matched_mse_seed42/best_model.pth" \
    --save_dir "$ROOT/runs/single_t60_mlp_matched_mse_seed42/test" \
    --gpu_id 0
