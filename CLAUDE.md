# CLAUDE.md — CMGAN T60 混响时间估计

> **分支提示**: 当前代码位于 `exp/single-t60` 分支，已经聚焦为单任务 T60 估计。不要按多任务去噪版本理解本分支；增强 decoder、去噪 loss 和 `test_denoise.py` 已移除。

## 项目目标

当前分支 `exp/single-t60` 将开源语音增强框架 CMGAN (Conformer-Based Metric GAN) 适配为 **单任务 T60 混响时间估计** 框架。任务目标是从含噪混响语音中估计 T60 值（0.1~1.5 秒）。

本分支已移除增强 decoder 和去噪训练路径，不再进行 noisy_reverb → clean_reverb 的辅助任务。

## 架构概览

```
原始 CMGAN (TSCNet)          单任务版本 (TSCNet_T60Estimator)
┌────────────────┐           ┌────────────────┐
│ DenseEncoder   │           │ DenseEncoder   │ ← 原始代码，未改
│ (260K params)  │           │ (260K params)  │
├────────────────┤           ├────────────────┤
│ TSCB × 4       │           │ TSCB × N       │ ← 原始代码，未改；默认 N=2，可选 N=4
│ (1.03M params) │           │ (1.03M params) │    双分支 Conformer (time+freq)
│                │           │       │         │
│                │           │       ├─────────┤
│                │           │       │ T60Head │ ← 唯一新增: 33K params
│                │           │       │ (Sigmoid)│    MultiStatsPool(avg+max+std) → FC
├────────────────┤           └────────────────┘
│ MaskDecoder    │           ❌ 已移除
│ ComplexDecoder │           ❌ 已移除
└────────────────┘
Discriminator (GAN)          ❌ 去掉，不需要
```

**4 层 TSCB 单任务模型约 1.32M 参数**，默认训练脚本使用 2 层 TSCB 以降低显存和训练时间。

### 维度流 (4 秒音频, 16kHz)

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100) → (B, 2, 641, 201)  [real, imag]
 → DenseEncoder              → (B, 64, 641, 101)  [频率轴 stride=2 下采样]
 → TSCB × N                  → (B, 64, 641, 101)  [不变，默认 N=2]
 → T60Head: mean(F) → MultiStatsPool(T) → FC(192→128→64→1) → Sigmoid → (B,)
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 功率压缩 | **不做** mag^0.3 | 对 T60 回归无意义，Conv2d 第一层可自适应数值范围 |
| 输入格式 | 原始复数 STFT | 简化数据流，与其他实验保持一致 |
| 能量归一化 | `c = sqrt(T / Σwav²)` | 沿用 CMGAN 原始设计，保证训练/推理一致 |
| T60 归一化 | min-max [0.1, 1.5] → [0, 1] | Sigmoid 输出范围匹配 |
| GAN | **去掉 Discriminator** | T60 回归不需要对抗训练 |
| Enhancement decoder | **去掉 MaskDecoder/ComplexDecoder** | 单任务分支不保留去噪梯度路径 |
| 损失函数 | T60 only | 支持 `mse` / `mae` / `huber`，默认 `mse` |
| AMP | 开启 | Conformer attention O(T²)，4 秒音频需要混合精度 |
| 梯度累积 | 默认 8 步 | 等效 batch = batch_size × accum_steps，按显存调整 |

### 显存分析

Conformer 的 self-attention 是 O(T²)：T=641, F=101 时，每个 TSCB 对 (B×101, 641, 64) 做 attention，batch=1 时单层 attention 约 330MB。4 层 × 2(time+freq) = 8 层 attention。

**对比其他模型:**
- Two-stage (GLU-TCM): O(T) 显存，batch=8 无压力
- CMGAN (Conformer): O(T²) 显存，batch=1 已用 ~15GB

## 文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `models/conformer.py` | 原始 CMGAN | Conformer block (attention + FFN + conv module) |
| `models/generator.py` | 原始 CMGAN | DenseEncoder, TSCB, MaskDecoder, ComplexDecoder |
| `models/generator_t60.py` | **新增** | TSCNet_T60Estimator = DenseEncoder + TSCB + T60HeadWithPool |
| `dataset.py` | **新增** | T60_Dataset_v7 数据加载器 (复数 STFT + T60 label) |
| `train.py` | **新增** | 单任务 T60 训练 (AMP + 梯度累积 + SwanLab) |
| `test.py` | **新增** | test1-4 评估 (标准T60指标, JSON 与 /compare 兼容) |
| `utils.py` | 原始 CMGAN | power_compress/uncompress, kaiming_init, LearnableSigmoid |
| `configs/t60_single/*.yaml` | **新增** | 单任务 T60 实验配置 |
| `train.sh` | **新增** | 一键训练脚本 |
| `train-and-test.sh` | **新增** | 一键训练并测试脚本 |
| `test.sh` | **新增** | 一键测试脚本 |
| `requirements.txt` | **新增** | Python 依赖 |

## 数据集

**数据集路径配置**: `configs/t60_single/*.yaml` 中的 `data.dataset_root` (23GB)

迁移到新机器时，优先修改 YAML 里的 `data.dataset_root`。运行时也可以用 `-d/--dataset-root` 临时覆盖该路径；`T60_DATASET_ROOT` 只作为兼容环境变量保留。

```
T60_Dataset_v7/
  train/      40,000 samples
  eval/        5,136 samples
  test1/       1,080 samples (simulated RIR + seen noise)
  test2/       1,080 samples (real RIR + seen noise)
  test3/       1,080 samples (simulated RIR + unseen noise)
  test4/       1,080 samples (real RIR + unseen noise)
```

每个样本: `speech{id}_reverb_{T60:.3f}_{SNR}dB/` 下有三个 wav (16kHz mono, ~4s)
- `{name}.wav` — 含噪混响 (模型输入，本分支唯一使用的音频)
- `{name}_denoised.wav` — 纯混响 (本分支忽略)
- `{name}_noise.wav` — 噪声分量 (本分支忽略)

## 运行命令

### 训练
```bash
conda activate demucs_xxn
# 推荐用 config 切换实验变量
./train.sh -c configs/t60_single/mlp_mse.yaml -g 0,1
./train.sh -c configs/t60_single/mlp_mae.yaml -g 2,3
./train.sh -c configs/t60_single/mlp_huber.yaml -g 4,5

# 训练后自动测试 test1-test4
./train-and-test.sh -c configs/t60_single/mlp_mse.yaml -g 0,1

# 临时覆盖 config 中的少量参数
./train.sh -c configs/t60_single/mlp_mse.yaml -d /path/to/T60_Dataset_v7 -- --batch_size 2
```

### 测试
```bash
./test.sh -c configs/t60_single/mlp_mse.yaml -m runs/single_t60_mlp_mse/best_model.pth -g 0
```

### 训练超参数 (默认)

默认配置文件位于 `configs/t60_single/`，训练超参数优先在 YAML 中维护。

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 4 | 受 Conformer 显存限制，按机器显存调整 |
| accum_steps | 8 | 默认等效 batch=32 |
| lr | 5e-4 | AdamW |
| max_epochs | 100 | 配合早停 patience=15 |
| loss | mse | 可选 mse / mae / huber |
| audio_length | 4.0 秒 | |
| n_fft / hop | 400 / 100 | CMGAN 原始参数 |

## 评估指标

对 test1-test4 各跑一次，输出:
- 总体: RMSE(ms), MAE(ms), Bias(ms), Pearson r, R², Mean Rel Error(%)
- T60 分组: bins [0.1-0.3, 0.3-0.5, ..., 1.3-1.5] 每组的 RMSE/MAE/r
- SNR 分组: [-5, 0, 5, 10, 15, 20] dB 每组的 RMSE/MAE/r

结果 JSON 保存到 `runs/{experiment}/test/`，格式与 `/compare` skill 的 Format B 兼容。

## 与其他实验的关系

本项目是 `/adapt-t60` skill 的一次执行实例。该 skill 定义了将任意语音增强框架适配为 T60 估计的标准化流程。

已有实验对比文档在原实验环境中维护；迁移时按实际位置同步或另行指定。

## 已知问题

1. **显存**: 4 秒音频 + Conformer attention 导致 batch_size 受限。如需更大 batch，可：
   - 缩短音频 `--audio_length 2.0` (原始 CMGAN 用 2 秒)
   - 减少 TSCB 层数 (4→2)
   - 使用更大显存的 GPU
2. **STFT 参数差异**: CMGAN 用 n_fft=400, hop=100；其他实验多用 n_fft=320, hop=160。频率分辨率不同，结果不直接横向对比。
