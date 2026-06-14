#!/usr/bin/env bash
# 测试 KAN 多任务实验: w_balanced (alpha=0.5, beta=0.5)
# Usage: ./scripts/test_multitask_kan_w_balanced.sh -g 0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

PYTHON="${PYTHON:-python}"
GPU_ID=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -g) GPU_ID="$2"; shift 2 ;;
        --no-conda) shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "Test kan_multitask_w_balanced on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/test_multitask.py" \
    --config  "$ROOT/configs/t60_multitask/kan_mse_w_balanced.yaml" \
    --model_path "$ROOT/runs/kan_multitask_w_balanced/best_model.pth" \
    --save_dir   "$ROOT/runs/kan_multitask_w_balanced/test" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"

echo "测试完成: kan_multitask_w_balanced"
