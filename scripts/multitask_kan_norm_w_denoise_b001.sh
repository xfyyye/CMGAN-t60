#!/usr/bin/env bash
# EMA归一化 + 权重组合: alpha=1.0 (denoise) / beta=0.01 (T60) —— 极端去噪主导（β 扫描，找触底）
# 镜像 scripts/multitask_kan_norm_w_denoise.sh，默认 GPU 2/3。
#
# Usage:
#   setsid nohup bash scripts/multitask_kan_norm_w_denoise_b001.sh \
#       > runs/kan_multitask_norm_w_denoise_b001/nohup.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHON="${PYTHON:-/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python}"
export CONDA_ENV="${CONDA_ENV:-demucs_xxn}"

exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_norm_w_denoise_b001.yaml" \
    "$@"
