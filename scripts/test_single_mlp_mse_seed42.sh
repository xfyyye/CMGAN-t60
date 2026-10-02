#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -g) GPU_ID="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
    esac
done

if [[ -n "${T60_DATASET_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--dataset_root "$T60_DATASET_ROOT")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/train.py" \
    --config "$ROOT/configs/t60_single/mlp_mse_seed42.yaml" \
    --test_only \
    --model_path "$ROOT/runs/single_t60_mlp_mse_seed42/best_model.pth" \
    --save_dir "$ROOT/runs/single_t60_mlp_mse_seed42/test" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"
