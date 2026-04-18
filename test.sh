#!/bin/bash
# CMGAN T60 测试脚本 (单GPU)
# 用法: ./test.sh -m 模型路径 -g GPU_ID

set -e

MODEL_PATH=""
GPU_ID=0
EXPERIMENT="CMGAN_t60_multitask"
BATCH_SIZE=8

while [[ $# -gt 0 ]]; do
    case $1 in
        -m) MODEL_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -e) EXPERIMENT="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请指定模型路径 (-m)"
    echo "用法: ./test.sh -m runs/CMGAN_t60_multitask/best_model.pth -g 0"
    exit 1
fi

SAVE_DIR="runs/${EXPERIMENT}/test"

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate demucs_xxn

CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
    --model_path "$MODEL_PATH" \
    --gpu_id 0 \
    --batch_size $BATCH_SIZE \
    --experiment_name "$EXPERIMENT" \
    --save_dir "$SAVE_DIR"

echo ""
echo "测试完成! 结果: $SAVE_DIR"
