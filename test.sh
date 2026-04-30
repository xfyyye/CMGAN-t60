#!/usr/bin/env bash
# CMGAN T60 测试脚本 (单GPU)
# 用法: ./test.sh -m 模型路径 [-d /path/to/T60_Dataset_v7] [-g GPU_ID]

set -euo pipefail

MODEL_PATH=""
GPU_ID=0
EXPERIMENT="CMGAN_t60_multitask"
BATCH_SIZE=8
# 本机默认数据集路径；可把下一行空字符串改成 "/path/to/T60_Dataset_v7"，也可用 -d 覆盖。
T60_DATASET_ROOT="${T60_DATASET_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-demucs_xxn}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -e) EXPERIMENT="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        -d|--dataset-root) T60_DATASET_ROOT="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请指定模型路径 (-m)"
    echo "用法: ./test.sh -m runs/CMGAN_t60_multitask/best_model.pth -g 0"
    exit 1
fi

if [[ -z "$T60_DATASET_ROOT" ]]; then
    echo "错误: 请指定数据集路径 (-d) 或设置 T60_DATASET_ROOT"
    exit 1
fi

if [[ ! -d "$T60_DATASET_ROOT" ]]; then
    echo "数据集目录不存在: $T60_DATASET_ROOT"
    exit 1
fi

export T60_DATASET_ROOT

SAVE_DIR="runs/${EXPERIMENT}/test"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
else
    echo "未找到 conda 命令，默认使用当前 Python 环境"
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
    --model_path "$MODEL_PATH" \
    --dataset_root "$T60_DATASET_ROOT" \
    --gpu_id 0 \
    --batch_size $BATCH_SIZE \
    --experiment_name "$EXPERIMENT" \
    --save_dir "$SAVE_DIR"

echo ""
echo "测试完成! 结果: $SAVE_DIR"
