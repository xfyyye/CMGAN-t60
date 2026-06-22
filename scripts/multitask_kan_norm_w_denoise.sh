#!/usr/bin/env bash
# EMA归一化 + 权重组合: alpha=1.0 (denoise) / beta=0.1 (T60) —— 去噪主导
# 镜像 scripts/multitask_kan_w_denoise.sh，默认 GPU 2/3（GPU 0/1 NVLink 异常）。
#
# Usage:
#   # 前台跑（默认 GPU 2,3）
#   ./scripts/multitask_kan_norm_w_denoise.sh
#   # 后台跑（推荐长时训练）
#   setsid nohup bash scripts/multitask_kan_norm_w_denoise.sh \
#       > runs/kan_multitask_norm_w_denoise/nohup.log 2>&1 &
#   # 覆盖 GPU / 数据集
#   ./scripts/multitask_kan_norm_w_denoise.sh -g 2,3 -d /path/to/T60_Dataset_v7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU 0/1 NVLink 异常（传输耗时 ~636ms），多任务实验默认用 GPU 2/3。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
# Python 环境（继承 conda base 所有包）
export PYTHON="${PYTHON:-/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python}"
export CONDA_ENV="${CONDA_ENV:-demucs_xxn}"

exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_norm_w_denoise.yaml" \
    "$@"
