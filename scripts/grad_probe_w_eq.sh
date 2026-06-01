#!/usr/bin/env bash
# 权重组合: alpha=1.0 (denoise) / beta=1.0 (T60) —— 中性基线
# 用法: ./scripts/grad_probe_w_eq.sh -g 0,1 -d /path/to/T60_Dataset_v7
# 注: 不要在调用时自己加 `--`，固定的 --alpha/--beta 会附加在末尾

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../train_grad_probe.sh" \
    -e grad_probe_w_eq "$@" \
    -- --alpha 1.0 --beta 1.0
