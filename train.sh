#!/usr/bin/env bash
# Usage: ./train.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

set -euo pipefail

CONFIG="${CONFIG:-configs/t60_single/mlp_mse.yaml}"
DATASET_ROOT_OVERRIDE="${T60_DATASET_ROOT:-}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
CONDA_ENV="${CONDA_ENV:-demucs_xxn}"
PYTHON="${PYTHON:-python}"
USE_CONDA=1
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 [-c CONFIG] [-d DATASET_PATH] [-g GPUS] [--conda-env ENV] [--no-conda] [-- extra train.py args]"
    echo "  -c CONFIG       YAML config path. Default: configs/t60_single/mlp_mse.yaml"
    echo "  -d DATASET_PATH Optional override for data.dataset_root in YAML"
    echo "  -g GPUS         CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift 2 ;;
        -d|--dataset-root) DATASET_ROOT_OVERRIDE="$2"; shift 2 ;;
        -g) GPUS="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --no-conda) USE_CONDA=0; shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "配置文件不存在: $CONFIG"
    exit 1
fi

if [[ -n "$DATASET_ROOT_OVERRIDE" && ! -d "$DATASET_ROOT_OVERRIDE" ]]; then
    echo "数据集目录不存在: $DATASET_ROOT_OVERRIDE"
    exit 1
fi

if [[ -n "$DATASET_ROOT_OVERRIDE" ]]; then
    export T60_DATASET_ROOT="$DATASET_ROOT_OVERRIDE"
fi

if [[ "$USE_CONDA" == "1" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [[ "$USE_CONDA" == "1" ]]; then
    echo "未找到 conda，使用当前 Python 环境"
fi

cmd=("$PYTHON" train.py --config "$CONFIG" --train_only)
if [[ -n "$DATASET_ROOT_OVERRIDE" ]]; then
    cmd+=(--dataset_root "$DATASET_ROOT_OVERRIDE")
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "Train: config=$CONFIG gpus=$GPUS"
if [[ -n "$DATASET_ROOT_OVERRIDE" ]]; then
    echo "Dataset override: $DATASET_ROOT_OVERRIDE"
else
    echo "Dataset: read from YAML data.dataset_root"
fi

CUDA_VISIBLE_DEVICES="$GPUS" "${cmd[@]}"

echo ""
echo "训练完成: $CONFIG"
