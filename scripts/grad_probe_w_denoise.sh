#!/usr/bin/env bash
# 权重组合: alpha=1.0 (denoise) / beta=0.1 (T60) —— 反向偏移，偏去噪
# 用法: ./scripts/grad_probe_w_denoise.sh -g 0,1 -d /path/to/T60_Dataset_v7

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_grad_probe.sh" \
    -e grad_probe_w_denoise "$@" \
    -- --alpha 1.0 --beta 0.1
