#!/usr/bin/env bash
# 权重组合: alpha=0.1 (denoise) / beta=1.0 (T60) —— master 默认，偏 T60
# 用法: ./scripts/grad_probe_w_t60.sh -g 0,1 -d /path/to/T60_Dataset_v7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_grad_probe.sh" \
    -e grad_probe_w_t60 "$@" \
    -- --alpha 0.1 --beta 1.0
