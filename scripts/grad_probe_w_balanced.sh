#!/usr/bin/env bash
# 权重组合: alpha=0.5 (denoise) / beta=0.5 (T60) —— 弱平衡
# 用法: ./scripts/grad_probe_w_balanced.sh -g 0,1 -d /path/to/T60_Dataset_v7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_grad_probe.sh" \
    -e grad_probe_w_balanced "$@" \
    -- --alpha 0.5 --beta 0.5
