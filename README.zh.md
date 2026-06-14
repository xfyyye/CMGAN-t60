# CMGAN-T60 多任务 — 盲估混响时间（去噪辅助任务）

将 [CMGAN](https://github.com/ruizhecao96/CMGAN)（基于 Conformer 的 Metric-GAN 语音增强框架）改造为**多任务联合训练框架**，同时完成 T60 混响时间估计与语音去噪。GAN 判别器已移除，DenseEncoder + TSCB backbone 由 FourierKAN T60 回归头与保留的 Mask/Complex Decoder（去噪辅助任务）共享。

> **分支**：`exp/kan-multitask` — 多任务联合训练为主线；单任务 T60 实验（`configs/t60_single/`）作为**消融对照基线**。

---

## 核心结果

### T60 估计（四测试集均值）

| 模型 | RMSE (ms) ↓ | MAE (ms) ↓ | Pearson *r* ↑ | R² ↑ |
|---|---|---|---|---|
| 单任务（消融基线） | 107.70 | 62.14 | 0.9488 | 0.8996 |
| 多任务 w_t60 (α=0.1, β=1.0) | 107.84 | 62.79 | 0.9489 | 0.8993 |
| 多任务 w_eq (α=1.0, β=1.0) | 105.64 | 62.04 | 0.9520 | 0.9032 |
| 多任务 w_balanced (α=0.5, β=0.5) | 100.28 | 62.12 | 0.9569 | 0.9128 |
| **多任务 w_denoise (α=1.0, β=0.1)** | **96.41** | **58.69** | **0.9594** | **0.9195** |

w_denoise 相比单任务基线 **RMSE 下降 10.5%**，参数量仅增加 6.6%。

### 去噪质量 — w_denoise（四测试集均值）

| 指标 | 含噪输入 | 增强后 | Δ |
|---|---|---|---|
| PESQ (wb) ↑ | 2.721 | **3.583** | +0.862 |
| STOI ↑ | 0.847 | **0.937** | +0.090 |
| SI-SDR (dB) ↑ | 15.91 | **21.83** | +5.92 |

---

## 模型架构

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100)  → (B, 2, T=641, F=201)
 → power_compress             → (B, 2, T, F)
 → [mag, real, imag] 拼接    → (B, 3, T, F)
 → DenseEncoder               → (B, 64, T, F/2=101)  ┐
 → TSCB_1, TSCB_2             → (B, 64, T, F/2)      ┤ 共享 backbone
      ├─► T60Head (FourierKAN) → t60_pred (B,)         ┘ 主任务
      ├─► MaskDecoder          → mask (B, 1, T, F)     ┐
      └─► ComplexDecoder       → complex (B, 2, T, F)  ┘ 辅助去噪
                               → iSTFT → 增强语音
```

| 模块 | 参数量 |
|---|---|
| DenseEncoder | ~260K（共享） |
| TSCB × 2 | ~516K（共享） |
| T60Head (FourierKAN) | ~86K |
| MaskDecoder + ComplexDecoder | ~27K |
| **合计** | **~1,406K** |

---

## 梯度探针分析

为理解去噪辅助任务提升 T60 估计的机制，在共享 backbone 中嵌入梯度探针（每 20 步记录一次）：

| 实验 | cos 均值 | neg% | r = ‖g_T60‖/‖g_denoise‖ |
|---|---|---|---|
| w_denoise | +0.030 | ~37% | **0.33** |
| w_balanced | +0.028 | ~41% | 0.89 |
| w_eq | +0.031 | ~40% | 0.93 |
| w_t60 | +0.024 | ~42% | 1.83 |

**核心发现**：梯度方向（cos ≈ 0，近似正交，无系统性冲突）在各组实验间几乎一致；关键差异在于梯度幅度比 r。T60 RMSE 与 r 高度相关（Pearson r = 0.885）：r 越小，T60 估计越准。当去噪主导共享层（r=0.33），其丰富的监督信号驱动编码器学习细粒度时频表征，从而也提升了 T60 回归精度。

完整分析（线性探针、t-SNE、CKA）详见 `EXPERIMENT_REPORT.md` 与 `ANALYSIS_GUIDE.md`。

---

## 环境要求

- Python ≥ 3.9，CUDA 11.8 / 12.x

```bash
pip install -r requirements.txt
```

主要依赖：`torch>=2.0`、`torchaudio>=2.0`、`einops>=0.6`、`scipy>=1.10`、`soundfile>=0.12`、`PyYAML>=6.0`、`swanlab>=0.3`（可选）。

---

## 数据集

`T60_Dataset_v7_4s` — 23 GB，16 kHz 单声道，每条约 4 秒：

```
T60_Dataset_v7_4s/
  train/      40,000 条
  eval/        5,136 条
  test1/       1,080 条  （仿真 RIR + 已见噪声）
  test2/       1,080 条  （真实 RIR + 已见噪声）
  test3/       1,080 条  （仿真 RIR + 未见噪声）
  test4/       1,080 条  （真实 RIR + 未见噪声）
```

每个样本目录（`speech{id}_reverb_{T60:.3f}_{SNR}dB/`）包含：
- `{name}.wav` — 含噪混响语音（模型输入）
- `{name}_denoised.wav` — 纯混响无噪语音（去噪 target）
- `{name}_noise.wav` — 噪声分量（不使用）

在 `configs/t60_multitask/*.yaml` 中设置 `data.dataset_root`。

---

## 快速开始

### 多任务训练

```bash
PYTHON=/path/to/venv/bin/python

# 后台启动（完全脱离终端）
setsid nohup bash scripts/multitask_kan_w_denoise.sh \
  > runs/kan_multitask_w_denoise/nohup.log 2>&1 &
```

### 多任务评估

```bash
# T60 估计（test1–4）
bash scripts/test_multitask_kan_w_denoise.sh -g 0

# 去噪质量（PESQ / STOI / SI-SDR）
bash scripts/test_multitask_denoise_kan_w_denoise.sh -g 0
```

### 消融对照（单任务基线）

```bash
bash train.sh -c configs/t60_single/kan_v2_mae_clip1_log.yaml -g 0,1
bash test.sh  -c configs/t60_single/kan_v2_mae_clip1_log.yaml \
              -m runs/single_t60_kan_v2_mae_clip1_log/best_model.pth -g 0
```

### 分析脚本

```bash
# 梯度探针可视化（无需 GPU）
python analyze_gradients.py --out_dir figures/gradient

# 表征分析（需要 GPU）
python analyze_representations.py --gpu 0 --n_samples 500 \
  --out_dir figures/representation
```

---

## 配置说明

多任务配置位于 `configs/t60_multitask/*.yaml`，核心字段：

```yaml
experiment:
  name: kan_multitask_w_denoise

model:
  n_tscb: 2
  t60_head_type: fourier_kan

data:
  dataset_root: /path/to/T60_Dataset_v7_4s

train:
  batch_size: 16
  accum_steps: 1
  max_epochs: 100
  lr: 0.0005
  early_stop_patience: 15

loss:
  alpha: 1.0   # 去噪任务权重
  beta:  0.1   # T60 任务权重
```

---

## 输出目录结构

```
runs/kan_multitask_{name}/
  best_model.pth
  training_history.json
  nohup_train.log
  test/                         ← T60 评估结果
    evaluation_results_test{1-4}.json
    evaluation_summary.json
  test_denoise/                 ← 去噪评估结果
    denoise_evaluation.json

figures/
  gradient/                     ← 梯度探针图（5 张）
  representation/               ← 线性探针、t-SNE、CKA（4 张）

demo_audio/                     ← 含噪 / 纯净 / 增强音频样本
```

---

## 文件说明

### 多任务主线

| 文件 | 说明 |
|---|---|
| `train_multitask.py` | 多任务训练循环（梯度探针 + SwanLab 日志） |
| `test_multitask.py` | T60 评估 — test1–4，标准指标 + T60/SNR 分 bin |
| `test_multitask_denoise.py` | 去噪评估 — PESQ / STOI / SI-SDR |
| `infer_demo.py` | 推理 demo — 生成含噪 / 纯净 / 增强三路音频 |
| `analyze_gradients.py` | 梯度探针可视化（cos/r 时间序列、直方图、柱状图） |
| `analyze_representations.py` | 表征分析（线性探针、t-SNE、CKA） |
| `models/generator_t60.py` | `TSCNet_KAN_MultiTask_2TSCB`（多任务）+ `TSCNet_T60Estimator_2TSCB`（消融） |
| `dataset.py` | `CMGANT60MultiTaskDataset`（多任务）/ `CMGANT60Dataset`（消融） |
| `configs/t60_multitask/` | 四组多任务实验配置 |
| `scripts/` | 各实验训练/测试启动脚本 |
| `EXPERIMENT_REPORT.md` | 完整实验报告（含所有指标和梯度分析） |
| `ANALYSIS_GUIDE.md` | 分析图表解读指南 |

### 消融对照（单任务）

| 文件 | 说明 |
|---|---|
| `train.py` / `test.py` | 单任务训练/评估脚本 |
| `train.sh` / `test.sh` / `train-and-test.sh` | 单任务启动脚本 |
| `configs/t60_single/` | 单任务实验配置 |
| `runs/single_t60_*/` | 单任务权重和结果（消融基线） |

---

## 致谢

本工作基于 [CMGAN](https://github.com/ruizhecao96/CMGAN)（Ruizhe Cao et al.）构建。

```bibtex
@inproceedings{cao2022cmgan,
  title     = {CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement},
  author    = {Ruizhe Cao and Sherif Abdulatif and Bin Yang},
  booktitle = {Interspeech},
  year      = {2022}
}
```
