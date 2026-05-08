# CMGAN-T60 — 盲估混响时间 T60

将 [CMGAN](https://github.com/ruizhecao96/CMGAN)（基于 Conformer 的 Metric-GAN 语音增强框架）适配为**单任务 T60 混响时间回归框架**。增强 Decoder 和判别器已全部移除，保留 DenseEncoder + TSCB 声学 backbone，在顶部新增轻量级 `T60HeadWithPool` 回归头。

> **当前分支**: `exp/single-t60` — 专注于从含噪混响语音中估计 T60，不做语音增强。详细设计决策见 [`AGENTS.md`](AGENTS.md)。

---

## 实验结果（CMGAN\_t60\_2tscb\_v2，2 层 TSCB 模型）

| 测试集 | RMSE (ms) | MAE (ms) | Pearson *r* | R² |
|---|---|---|---|---|
| test1 — 仿真 RIR + 已见噪声 | 109.73 | 67.99 | 0.9483 | 0.8961 |
| test2 — 真实 RIR + 已见噪声 | 118.82 | 75.19 | 0.9383 | 0.8783 |
| test3 — 仿真 RIR + 未见噪声 | 110.50 | 73.92 | 0.9505 | 0.8951 |
| test4 — 真实 RIR + 未见噪声 | 113.33 | 73.83 | 0.9448 | 0.8877 |
| **四集平均** | **113.10** | **72.73** | **0.9455** | **0.8893** |

---

## 模型架构

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100) → (B, 2, 641, 201)  [real, imag]
 → DenseEncoder              → (B, 64, 641, 101)  [频率轴 stride=2 下采样]
 → TSCB × N                  → (B, 64, 641, 101)  [N=2 或 4]
 → T60HeadWithPool
     mean(F) → MultiStatsPool(avg+max+std, T) → FC(192→128→64→1) → Sigmoid
 → t60_pred (B,)  [归一化到 0–1]
```

| 变体 | TSCB 层数 | 参数量 |
|---|---|---|
| `TSCNet_T60Estimator` | 4 | ~1.87 M |
| `TSCNet_T60Estimator_2TSCB` | 2 | ~1.35 M |

---

## 环境要求

- Python ≥ 3.9
- CUDA 11.8 / 12.x（GPU 训练）

```bash
pip install -r requirements.txt
```

主要依赖：`torch>=2.0`、`torchaudio>=2.0`、`einops>=0.6`、`scipy>=1.10`、`soundfile>=0.12`、`PyYAML>=6.0`、`swanlab>=0.3`（可选，实验追踪）。

---

## 数据集

`T60_Dataset_v7` — 23 GB，16 kHz 单声道，每条约 4 秒：

```
T60_Dataset_v7/
  train/      40,000 条
  eval/        5,136 条
  test1/       1,080 条  （仿真 RIR + 已见噪声）
  test2/       1,080 条  （真实 RIR + 已见噪声）
  test3/       1,080 条  （仿真 RIR + 未见噪声）
  test4/       1,080 条  （真实 RIR + 未见噪声）
```

目录名即标签：

```
speech000001_reverb_0.214_0dB/   →  T60 = 0.214 s，SNR = 0 dB
```

在 `configs/t60_single/*.yaml` 的 `data.dataset_root` 中设置数据集路径，或在运行时通过 `-d /path/to/T60_Dataset_v7` 临时覆盖。

---

## 快速开始

### 训练

```bash
conda activate demucs_xxn

# 单配置训练（使用 GPU 0,1）
./train.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

# 训练完成后自动评估 test1-test4
./train-and-test.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

# 临时覆盖数据集路径（不修改 YAML）
./train.sh -c configs/t60_single/mlp_mse.yaml -d /path/to/T60_Dataset_v7 -g 0,1

# 通过 -- 传入额外的 train.py 参数
./train.sh -c configs/t60_single/mlp_mse.yaml -- --batch_size 2 --accum_steps 16
```

并行运行多个损失函数实验：

```bash
./train.sh -c configs/t60_single/mlp_mse.yaml   -g 0,1 &
./train.sh -c configs/t60_single/mlp_mae.yaml   -g 2,3 &
./train.sh -c configs/t60_single/mlp_huber.yaml -g 4,5 &
```

### 仅评估

```bash
./test.sh -c configs/t60_single/mlp_mse.yaml \
          -m runs/single_t60_mlp_mse/best_model.pth \
          -g 0
```

---

## 配置说明

所有超参数统一在 `configs/t60_single/*.yaml` 中维护：

```yaml
experiment:
  name: single_t60_mlp_mse    # 实验名；输出目录默认是 output_root/name
  output_root: runs            # 权重、日志、测试结果的根目录

model:
  n_tscb: 2                   # 2（更快）或 4（接近原始 CMGAN 深度）

data:
  dataset_root: /path/to/T60_Dataset_v7
  n_fft: 400
  hop_length: 100
  audio_length: 4.0           # 固定音频长度（秒）
  target_sr: 16000
  t60_min: 0.1
  t60_max: 1.5

train:
  batch_size: 4               # 按显存调整
  accum_steps: 8              # 等效 batch = batch_size × accum_steps = 32
  max_epochs: 100
  lr: 0.0005
  weight_decay: 0.00001
  early_stop_patience: 15     # val loss 连续多少轮不提升则早停
  num_workers: 4

loss:
  name: mse                   # mse | mae | huber

logging:
  swanlab_project: T60_Estimation
```

---

## 默认超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size` | 4 | Conformer O(T²) 显存限制 |
| `accum_steps` | 8 | 等效 batch = 32 |
| `lr` | 5e-4 | AdamW 学习率 |
| `weight_decay` | 1e-5 | AdamW 权重衰减 |
| `max_epochs` | 100 | 配合早停 |
| `early_stop_patience` | 15 | 监控验证集 loss |
| `n_fft` / `hop` | 400 / 100 | 4s/16kHz 约 641 帧 |
| `audio_length` | 4.0 s | 固定长度输入 |
| T60 范围 | 0.1 – 1.5 s | Min-max 归一化到 [0, 1] |
| 损失函数 | MSE | 可选 `mse` / `mae` / `huber` |

---

## 输出目录结构

```
runs/{experiment_name}/
  best_model.pth                       # 验证集最优 checkpoint
  training_history.json                # 每轮训练/验证指标
  test/
    evaluation_results_test{1-4}.json  # 含逐样本 + 分组指标
    evaluation_summary.json            # 四测试集平均汇总
```

评估 JSON 格式与 `/compare` skill 的 **Format B** 兼容。

---

## 评估指标

对 test1–test4 各运行一次，输出：

- **总体指标**：RMSE (ms)、MAE (ms)、Bias (ms)、Pearson *r*、R²、平均相对误差 (%)
- **T60 分组**：`[0.1–0.3, 0.3–0.5, …, 1.3–1.5]` s，每组的 RMSE / MAE / *r*
- **SNR 分组**：`[-5, 0, 5, 10, 15, 20]` dB，每组的 RMSE / MAE / *r*

---

## 文件说明

| 文件 | 来源 | 说明 |
|---|---|---|
| `models/conformer.py` | 原始 CMGAN，未改动 | Conformer block（attention + FFN + conv） |
| `models/generator.py` | 原始 CMGAN，未改动 | DenseEncoder + TSCB |
| `models/generator_t60.py` | **新增** | `TSCNet_T60Estimator` + `T60HeadWithPool` |
| `dataset.py` | **新增** | `T60_Dataset_v7` 数据加载器（复数 STFT + T60 标签） |
| `train.py` | **新增** | 单任务训练循环（AMP + 梯度累积 + SwanLab） |
| `test.py` | **新增** | test1–test4 评估，含分组指标 |
| `utils.py` | 原始 CMGAN | `power_compress`、`kaiming_init`、`LearnableSigmoid` |
| `configs/t60_single/` | **新增** | MSE / MAE / Huber 实验 YAML 配置 |
| `train.sh` | **新增** | 训练启动脚本 |
| `train-and-test.sh` | **新增** | 训练后自动测试 |
| `test.sh` | **新增** | 仅评估启动脚本 |

---

## 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 输入格式 | 原始复数 STFT（real + imag + mag → 3 通道） | 保留相位信息 |
| 能量归一化 | `c = sqrt(T / Σwav²)` | 与上游 CMGAN 一致，训练/推理行为一致 |
| T60 归一化 | Min-max [0.1, 1.5] → [0, 1] | 匹配 Sigmoid 输出范围 |
| GAN 判别器 | **已移除** | 回归任务不需要对抗训练 |
| 增强 Decoder | **已移除** | 单任务分支不保留去噪梯度路径 |
| 混合精度 AMP | 开启 | Conformer attention 为 O(T²)，4 s 音频需要 FP16 |
| 梯度累积 | 默认 8 步 | 等效 batch = 32，在显存受限时扩大有效 batch |

### 显存说明

Conformer self-attention 是 O(T²)。T=641、F=101，4 层 × 2（time+freq）= 8 次 attention 操作：

- batch=1 时单层 attention 约 330 MB
- OOM 时可减小 `batch_size`、缩短 `audio_length`，或减少 TSCB 层数（4→2）

