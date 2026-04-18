#!/bin/bash
# CMGAN T60 多任务训练 + 测试 (双GPU DataParallel)
# 用法:
#   ./train.sh                       # 训练 + 测试 (默认双GPU)
#   ./train.sh --train-only          # 仅训练
#   ./train.sh --test-only -m 路径   # 仅测试
#   ./train.sh -e 实验名 -b 16       # 指定实验名和batch_size

set -e

# 默认参数
EXPERIMENT="CMGAN_t60_multitask"
GPUS="0,1"
BATCH_SIZE=4
EPOCHS=100
LR=5e-4
ALPHA=0.1
BETA=1.0
AUDIO_LENGTH=4.0
ACCUM_STEPS=8
N_TSCB=2
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e) EXPERIMENT="$2"; shift 2 ;;
        -g) GPUS="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --n_tscb) N_TSCB="$2"; shift 2 ;;
        --audio-length) AUDIO_LENGTH="$2"; shift 2 ;;
        --train-only) EXTRA_ARGS="$EXTRA_ARGS --train_only"; shift ;;
        --test-only) EXTRA_ARGS="$EXTRA_ARGS --test_only"; shift ;;
        -m) EXTRA_ARGS="$EXTRA_ARGS --model_path $2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

SAVE_DIR="runs/${EXPERIMENT}"

echo "============================================"
echo "CMGAN T60 多任务训练 (DataParallel)"
echo "============================================"
echo "  实验名:   $EXPERIMENT"
echo "  GPUs:     $GPUS"
echo "  Batch:    $BATCH_SIZE × ${ACCUM_STEPS}累积 = 等效$((BATCH_SIZE * ACCUM_STEPS))"
echo "  Epochs:   $EPOCHS"
echo "  LR:       $LR"
echo "  α(去噪):  $ALPHA"
echo "  β(T60):   $BETA"
echo "  音频长度: ${AUDIO_LENGTH}s"
echo "  TSCB层数: $N_TSCB"
echo "  累积步数: $ACCUM_STEPS"
echo "  保存目录: $SAVE_DIR"
echo "============================================"

source /home/ps/anaconda3/etc/profile.d/conda.sh
conda activate demucs_xxn

CUDA_VISIBLE_DEVICES=$GPUS python train.py \
    --experiment_name "$EXPERIMENT" \
    --batch_size $BATCH_SIZE \
    --max_epochs $EPOCHS \
    --lr $LR \
    --alpha $ALPHA \
    --beta $BETA \
    --audio_length $AUDIO_LENGTH \
    --accum_steps $ACCUM_STEPS \
    --n_tscb $N_TSCB \
    --save_dir "$SAVE_DIR" \
    $EXTRA_ARGS

echo ""
echo "完成! 结果保存在: $SAVE_DIR"
