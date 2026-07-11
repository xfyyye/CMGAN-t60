#!/usr/bin/env bash
# 消融实验：MLP head vs Fourier-KAN head（同一 multitask 配置 α=1.0,β=0.05）
# 镜像 scripts/multitask_kan_norm_w_denoise_b005.sh，仅 head 类型不同。
#
# Usage:
#   setsid nohup bash scripts/multitask_kan_norm_w_denoise_mlphead_b005.sh \
#       > runs/kan_multitask_norm_w_denoise_mlphead_b005/nohup.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_norm_w_denoise_mlphead_b005.yaml" \
    "$@"
