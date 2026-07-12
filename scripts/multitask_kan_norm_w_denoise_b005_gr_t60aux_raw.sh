#!/usr/bin/env bash
# GR attempt2: 原始 loss + T60 作辅（主=去噪）
# Usage:
#   setsid nohup bash scripts/multitask_kan_norm_w_denoise_b005_gr_t60aux_raw.sh \
#       > runs/kan_multitask_norm_w_denoise_b005_gr_t60aux_raw/nohup.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONDA_ENV="${CONDA_ENV:-demucs_xxn}"

exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_norm_w_denoise_b005_gr_t60aux_raw.yaml" \
    "$@"
