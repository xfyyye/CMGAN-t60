#!/usr/bin/env bash
# Gradient Remedy 实验：主模型 β=0.05 + GR 开启
# 与 scripts/multitask_kan_norm_w_denoise_b005.sh 唯一差异 = gradient_remedy.use: true
#
# Usage:
#   setsid nohup bash scripts/multitask_kan_norm_w_denoise_b005_gr.sh \
#       > runs/kan_multitask_norm_w_denoise_b005_gr/nohup.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU 0/1 NVLink 异常，多任务实验默认用 GPU 2/3。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHON="${PYTHON:-/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python}"
export CONDA_ENV="${CONDA_ENV:-demucs_xxn}"

exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_norm_w_denoise_b005_gr.yaml" \
    "$@"
