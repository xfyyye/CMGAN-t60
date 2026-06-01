#!/usr/bin/env bash
# 权重组合: alpha=1.0 (denoise) / beta=0.1 (T60) —— 偏向去噪
# Usage: ./scripts/multitask_kan_w_denoise.sh -g 0,1 [-d /path/to/T60_Dataset_v7]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_w_denoise.yaml" "$@"
