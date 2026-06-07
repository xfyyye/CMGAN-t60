#!/usr/bin/env bash
# 去噪性能测试: kan_multitask_w_denoise (alpha=1.0, beta=0.1)
# Usage: ./scripts/test_multitask_denoise_kan_w_denoise.sh -g 0

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

echo "Denoise Test: kan_multitask_w_denoise on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/test_multitask_denoise.py" \
    --config       "$ROOT/configs/t60_multitask/kan_mse_w_denoise.yaml" \
    --model_path   "$ROOT/runs/kan_multitask_w_denoise/best_model.pth" \
    --save_dir     "$ROOT/runs/kan_multitask_w_denoise/test_denoise" \
    --experiment_name kan_multitask_w_denoise \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"

echo "去噪测试完成: kan_multitask_w_denoise"
