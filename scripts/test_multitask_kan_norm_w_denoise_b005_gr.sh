#!/usr/bin/env bash
# 测试 GR 实验 T60 估计: kan_multitask_norm_w_denoise_b005_gr
# Usage: ./scripts/test_multitask_kan_norm_w_denoise_b005_gr.sh -g 0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

PYTHON="${PYTHON:-/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python}"
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

echo "Test kan_multitask_norm_w_denoise_b005_gr on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$ROOT/test_multitask.py" \
    --config     "$ROOT/configs/t60_multitask/kan_mse_norm_w_denoise_b005_gr.yaml" \
    --model_path "$ROOT/runs/kan_multitask_norm_w_denoise_b005_gr/best_model.pth" \
    --save_dir   "$ROOT/runs/kan_multitask_norm_w_denoise_b005_gr/test" \
    --gpu_id 0 \
    "${EXTRA_ARGS[@]}"

echo "测试完成: kan_multitask_norm_w_denoise_b005_gr"
