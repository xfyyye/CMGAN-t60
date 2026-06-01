#!/usr/bin/env bash
# 权重组合: alpha=0.5 (denoise) / beta=0.5 (T60) —— 弱平衡
# Usage: ./scripts/multitask_kan_w_balanced.sh -g 0,1 [-d /path/to/T60_Dataset_v7]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_multitask.sh" \
    -c "${SCRIPT_DIR}/../configs/t60_multitask/kan_mse_w_balanced.yaml" "$@"
