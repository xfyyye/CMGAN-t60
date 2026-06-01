#!/usr/bin/env bash
# Usage: ./train_grad_probe.sh [-d /path/to/T60_Dataset_v7] [-e EXP] [-g 0,1] [-b 4] [-- extra train_grad_probe.py args]
#
# 与 train.sh 完全等价，只是调用 train_grad_probe.py 以启用梯度夹角探针。

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-expname}" # 修改为你想要的实验名称，输出log和模型都会保存在 runs/expname 目录下
# 本机默认数据集路径；可把下一行空字符串改成 "/path/to/T60_Dataset_v7"，也可用 -d 覆盖。
T60_DATASET_ROOT="${T60_DATASET_ROOT:-/path/to/T60_Dataset_v7}" # 在这里更改T60_Dataset_v7路径
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}" # 修改为你要使用的GPU ID，例如 "0" 或 "0,1"
BATCH_SIZE="${BATCH_SIZE:-4}" # 修改为你想要的批大小
MAX_EPOCHS="${MAX_EPOCHS:-100}" # 修改为你想要的训练轮数
LR="${LR:-5e-4}" # 修改为你想要的学习率，默认5e-4
ACCUM_STEPS="${ACCUM_STEPS:-8}" # 梯度累积步数
N_TSCB="${N_TSCB:-2}" # 默认两个 TSCB 模块
AUDIO_LENGTH="${AUDIO_LENGTH:-4.0}"
PROBE_LOG_EVERY="${PROBE_LOG_EVERY:-20}" # 梯度探针每多少 train step 触发一次
CONDA_ENV="${CONDA_ENV:-demucs_xxn}"
PYTHON="${PYTHON:-python}"
USE_CONDA=1
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 [-d DATASET_PATH] [-e EXP] [-g GPUS] [-b BATCH] [--conda-env ENV] [-- extra train_grad_probe.py args]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -e) EXPERIMENT="$2"; shift 2 ;;
        -g) GPUS="$2"; shift 2 ;;
        -b) BATCH_SIZE="$2"; shift 2 ;;
        -d|--dataset-root) T60_DATASET_ROOT="$2"; shift 2 ;;
        --epochs|--max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --accum-steps) ACCUM_STEPS="$2"; shift 2 ;;
        --n-tscb|--n_tscb) N_TSCB="$2"; shift 2 ;;
        --audio-length) AUDIO_LENGTH="$2"; shift 2 ;;
        --probe-log-every) PROBE_LOG_EVERY="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --no-conda) USE_CONDA=0; shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

SAVE_DIR="runs/${EXPERIMENT}"

if [[ -z "$T60_DATASET_ROOT" ]]; then
    echo "未指定数据集路径: 请在 train_grad_probe.sh 中设置 T60_DATASET_ROOT，或运行时传入 -d"
    usage
    exit 1
fi

if [[ ! -d "$T60_DATASET_ROOT" ]]; then
    echo "数据集目录不存在: $T60_DATASET_ROOT"
    exit 1
fi

export T60_DATASET_ROOT

if [[ "$USE_CONDA" == "1" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [[ "$USE_CONDA" == "1" ]]; then
    echo "未找到 conda，使用当前 Python 环境"
fi

echo "Train (grad-probe): exp=$EXPERIMENT gpus=$GPUS batch=$BATCH_SIZE accum=$ACCUM_STEPS probe_every=$PROBE_LOG_EVERY dataset=$T60_DATASET_ROOT"

CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" train_grad_probe.py \
    --dataset_root "$T60_DATASET_ROOT" \
    --experiment_name "$EXPERIMENT" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --audio_length $AUDIO_LENGTH \
    --accum_steps $ACCUM_STEPS \
    --n_tscb $N_TSCB \
    --probe_log_every $PROBE_LOG_EVERY \
    --save_dir "$SAVE_DIR" \
    --train_only \
    "${EXTRA_ARGS[@]}"

echo ""
echo "训练完成: $SAVE_DIR"
