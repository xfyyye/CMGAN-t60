# AGENTS.md — CMGAN 多任务 T60 估计 + 去噪联合训练

> **分支**: `exp/kan-multitask` — 主任务为**多任务联合训练**（T60 估计 + 去噪重建）。
> 单任务 T60 实验（`configs/t60_single/` + `train.py`）作为**消融对照基线**保留在本分支，不是主线。

## 项目目标

本分支在 CMGAN（Conformer-Based Metric GAN）backbone 基础上，实验**去噪辅助任务对 T60 估计的影响**：

- **主任务**：从含噪混响语音中估计 T60 混响时间（0.1–1.5 秒）
- **辅助任务**：去噪重建（noisy_reverb → clean_reverb），作为共享 backbone 的额外监督信号
- **研究问题**：去噪梯度如何影响共享层特征表征，进而改善 T60 估计精度

## 架构概览

```
原始 CMGAN (TSCNet)              多任务版本 (TSCNet_KAN_MultiTask_2TSCB)
┌────────────────┐               ┌────────────────┐
│ DenseEncoder   │               │ DenseEncoder   │ ← 共享，未改
│ (260K params)  │               │ (260K params)  │
├────────────────┤               ├────────────────┤
│ TSCB × 4       │               │ TSCB × 2       │ ← 共享，双分支 Conformer
│ (1.03M params) │               │ (516K params)  │
├────────────────┤               │       │         │
│ MaskDecoder    │               │  ┌────┴──────┐  │
│ ComplexDecoder │  →  保留      │  │  T60Head  │  │ ← 新增，FourierKAN，~86K
└────────────────┘               │  │ (Sigmoid) │  │
Discriminator (GAN)  ❌ 已移除   │  MaskDecoder   │ ← 保留，去噪辅助任务
                                 │  ComplexDecoder│
                                 └────────────────┘
总参数量：~1,406K（+6.6% vs 单任务 ~1,319K）
```

### 维度流（4 秒音频，16kHz）

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100)    → (B, 2, T=641, F=201)
 → power_compress               → (B, 2, T, F)
 → [mag, real, imag] cat        → (B, 3, T, F)
 → DenseEncoder                 → (B, 64, T, F/2=101)   [共享]
 → TSCB_1, TSCB_2               → (B, 64, T, F/2)       [共享]
      ├─► T60Head (FourierKAN)  → t60_pred (B,)
      ├─► MaskDecoder           → mask (B, 1, T, F)
      └─► ComplexDecoder        → complex_out (B, 2, T, F)
                                → iSTFT → enhanced wav
```

## 实验配置

四组多任务实验，通过 α/β 控制去噪/T60 的梯度主导权：

| 实验名 | α (denoise) | β (T60) | 设计意图 |
|---|---|---|---|
| `w_denoise` | 1.0 | 0.1 | 去噪强监督主导共享层（**最优配置**） |
| `w_balanced` | 0.5 | 0.5 | 均衡权重 |
| `w_eq` | 1.0 | 1.0 | 等权基线 |
| `w_t60` | 0.1 | 1.0 | T60 梯度主导，验证退化情况 |

单任务最优（`single_t60_kan_v2_mae_clip1_log`，RMSE=107.70ms）作为消融基线对照。

## 关键实验结论

**T60 估计（四测试集均值）：**

| 模型 | RMSE (ms) ↓ | MAE (ms) ↓ | Pearson r ↑ | R² ↑ |
|---|---|---|---|---|
| 单任务（消融基线） | 107.70 | 62.14 | 0.9488 | 0.8996 |
| w_t60 | 107.84 | 62.79 | 0.9489 | 0.8993 |
| w_eq | 105.64 | 62.04 | 0.9520 | 0.9032 |
| w_balanced | 100.28 | 62.12 | 0.9569 | 0.9128 |
| **w_denoise** | **96.41** | **58.69** | **0.9594** | **0.9195** |

**核心机制**：梯度幅度比 r = ‖g_T60‖/‖g_denoise‖ 是决定因素（r 越小性能越好），梯度方向近似正交（cos≈0，无系统性冲突）。详见 `EXPERIMENT_REPORT.md` 和 `ANALYSIS_GUIDE.md`。

## 文件说明

### 多任务主线

| 文件 | 说明 |
|---|---|
| `train_multitask.py` | 多任务训练脚本（含 swanlab 日志、梯度探针） |
| `test_multitask.py` | T60 评估（test1–4，标准指标 + 分 bin） |
| `test_multitask_denoise.py` | 去噪评估（PESQ / STOI / SI-SDR） |
| `infer_demo.py` | 推理 demo（生成 noisy/clean/enhanced 三路音频） |
| `train_multitask.sh` | 多任务训练启动器 |
| `analyze_gradients.py` | 梯度探针可视化（cos/r 时间序列、分布、分层对比） |
| `analyze_representations.py` | 表征分析（线性探针、t-SNE、CKA） |
| `models/generator_t60.py` | `TSCNet_KAN_MultiTask_2TSCB`（多任务）+ `TSCNet_T60Estimator_2TSCB`（单任务） |
| `dataset.py` | `CMGANT60Dataset`（单任务）/ `CMGANT60MultiTaskDataset`（多任务） |
| `configs/t60_multitask/*.yaml` | 四组多任务实验配置 |
| `scripts/` | 各实验训练/测试启动脚本 |
| `runs/kan_multitask_*/` | 训练权重、日志、测试结果 |
| `figures/` | 梯度分析和表征分析图表 |
| `demo_audio/` | 推理 demo 音频样本 |
| `EXPERIMENT_REPORT.md` | 完整实验报告（含所有指标、梯度分析摘要） |
| `ANALYSIS_GUIDE.md` | 分析结果解读指南（含图表说明） |

### 消融对照（单任务）

| 文件 | 说明 |
|---|---|
| `train.py` / `test.py` | 单任务训练/评估脚本 |
| `train.sh` / `test.sh` / `train-and-test.sh` | 单任务启动脚本 |
| `configs/t60_single/*.yaml` | 单任务实验配置 |
| `runs/single_t60_*/` | 单任务权重和结果（消融基线） |

## 数据集

数据集路径在 `configs/t60_multitask/*.yaml` 的 `data.dataset_root` 中配置。

```
T60_Dataset_v7_4s/
  train/      40,000 samples
  eval/        5,136 samples
  test1/       1,080 samples  (仿真 RIR + 已见噪声)
  test2/       1,080 samples  (真实 RIR + 已见噪声)
  test3/       1,080 samples  (仿真 RIR + 未见噪声)
  test4/       1,080 samples  (真实 RIR + 未见噪声)
```

每个样本目录（`speech{id}_reverb_{T60:.3f}_{SNR}dB/`）下：
- `{name}.wav` — 含噪混响（模型输入）
- `{name}_denoised.wav` — 纯混响无噪声（去噪任务 target）
- `{name}_noise.wav` — 噪声分量（不使用）

## 运行命令

### 多任务训练

```bash
# Python 环境（继承 conda base 所有包）
PYTHON=/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python

# 后台启动训练（setsid nohup 完全脱离父进程）
setsid nohup bash scripts/multitask_kan_w_denoise.sh \
  > runs/kan_multitask_w_denoise/nohup.log 2>&1 &
```

### 多任务测试

```bash
# T60 估计评估
bash scripts/test_multitask_kan_w_denoise.sh -g 0

# 去噪质量评估
bash scripts/test_multitask_denoise_kan_w_denoise.sh -g 0
```

### 消融对照（单任务）

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
python analyze_representations.py --gpu 2 --n_samples 500 \
  --out_dir figures/representation
```

## 注意事项

- **GPU 限制**：GPU 0/1 NVLink 异常（传输耗时 ~636ms），多任务实验用 GPU 2/3
- **Python 环境**：`/mnt/tidal-sh01/usr/chuan/youling/demucs_xxn/bin/python`
- **SwanLab 项目**：`T60_KAN_MultiTask`，账号 `xfyyye`
- **长时训练**：务必用 `setsid nohup` 启动，避免进程被工具超时 kill
- **显存**：Conformer attention O(T²)，batch=16 需要约 40GB；OOM 时减小 batch_size 或缩短 audio_length
