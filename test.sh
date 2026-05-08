#!/usr/bin/env bash
# Usage: ./test.sh -c configs/t60_single/mlp_mse.yaml -m runs/single_t60_mlp_mse/best_model.pth -g 0

set -euo pipefail

CONFIG="${CONFIG:-configs/t60_single/mlp_mse.yaml}"
MODEL_PATH=""
GPU_ID=0
DATASET_ROOT_OVERRIDE="${T60_DATASET_ROOT:-}"
EXPERIMENT_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
SAVE_DIR_OVERRIDE=""
CONDA_ENV="${CONDA_ENV:-demucs_xxn}"
PYTHON="${PYTHON:-python}"
USE_CONDA=1
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 -m MODEL_PATH [-c CONFIG] [-d DATASET_PATH] [-g GPU_ID] [--conda-env ENV] [--no-conda] [-- extra test.py args]"
    echo "  -c CONFIG       YAML config path. Default: configs/t60_single/mlp_mse.yaml"
    echo "  -m MODEL_PATH   Checkpoint path, e.g. runs/single_t60_mlp_mse/best_model.pth"
    echo "  -d DATASET_PATH Optional override for data.dataset_root in YAML"
    echo "  -g GPU_ID       Physical GPU id exposed to this test process"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG="$2"; shift 2 ;;
        -m|--model-path) MODEL_PATH="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -e|--experiment-name) EXPERIMENT_OVERRIDE="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE_OVERRIDE="$2"; shift 2 ;;
        -d|--dataset-root) DATASET_ROOT_OVERRIDE="$2"; shift 2 ;;
        --save-dir) SAVE_DIR_OVERRIDE="$2"; shift 2 ;;
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

if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: 请指定模型路径 (-m)"
    usage
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

cmd=("$PYTHON" test.py --config "$CONFIG" --model_path "$MODEL_PATH" --gpu_id 0)
if [[ -n "$DATASET_ROOT_OVERRIDE" ]]; then
    cmd+=(--dataset_root "$DATASET_ROOT_OVERRIDE")
fi
if [[ -n "$EXPERIMENT_OVERRIDE" ]]; then
    cmd+=(--experiment_name "$EXPERIMENT_OVERRIDE")
fi
if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE_OVERRIDE")
fi
if [[ -n "$SAVE_DIR_OVERRIDE" ]]; then
    cmd+=(--save_dir "$SAVE_DIR_OVERRIDE")
elif [[ -n "$EXPERIMENT_OVERRIDE" ]]; then
    cmd+=(--save_dir "runs/${EXPERIMENT_OVERRIDE}/test")
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "Test: config=$CONFIG model=$MODEL_PATH gpu=$GPU_ID"
if [[ -n "$DATASET_ROOT_OVERRIDE" ]]; then
    echo "Dataset override: $DATASET_ROOT_OVERRIDE"
else
    echo "Dataset: read from YAML data.dataset_root"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}"

echo ""
echo "测试完成"
