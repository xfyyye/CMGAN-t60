# CLAUDE.md — CMGAN T60 混响时间估计

## 项目目标

将开源语音增强框架 CMGAN (Conformer-Based Metric GAN) 适配为 **T60 混响时间估计** 的多任务训练框架。同时完成两个任务：
1. **T60 回归**：从含噪混响语音中估计 T60 值（0.1~1.5 秒）
2. **去噪**：去除噪声，保留混响语音（noisy_reverb → clean_reverb）

## 架构概览

```
原始 CMGAN (TSCNet)          改造后 (TSCNet_MultiTask)
┌────────────────┐           ┌────────────────┐
│ DenseEncoder   │           │ DenseEncoder   │ ← 原始代码，未改
│ (260K params)  │           │ (260K params)  │
├────────────────┤           ├────────────────┤
│ TSCB × 4       │           │ TSCB × 4       │ ← 原始代码，未改
│ (1.03M params) │           │ (1.03M params) │    双分支 Conformer (time+freq)
│                │           │       │         │
│                │           │       ├─────────┤
│                │           │       │ T60Head │ ← 唯一新增: 33K params
│                │           │       │ (Sigmoid)│    MultiStatsPool(avg+max+std) → FC
├────────────────┤           ├────────────────┤
│ MaskDecoder    │           │ MaskDecoder    │ ← 原始代码，未改
│ ComplexDecoder │           │ ComplexDecoder │
└────────────────┘           └────────────────┘
Discriminator (GAN)          ❌ 去掉，不需要
```

**总参数: 1.87M** (T60 头仅占 1.8%)

### 维度流 (4 秒音频, 16kHz)

```
wav (B, 64000)
 → STFT(n_fft=400, hop=100) → (B, 2, 641, 201)  [real, imag]
 → DenseEncoder              → (B, 64, 641, 101)  [频率轴 stride=2 下采样]
 → TSCB × 4                  → (B, 64, 641, 101)  [不变]
 → T60Head: mean(F) → MultiStatsPool(T) → FC(192→128→64→1) → Sigmoid → (B,)
 → MaskDecoder + ComplexDecoder → (B, 1, 641, 201) × 2  [去噪输出]
```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 功率压缩 | **不做** mag^0.3 | 对 T60 回归无意义，Conv2d 第一层可自适应数值范围 |
| 输入格式 | 原始复数 STFT | 简化数据流，与其他实验保持一致 |
| 能量归一化 | `c = sqrt(T / Σwav²)` | 沿用 CMGAN 原始设计，保证训练/推理一致 |
| T60 归一化 | min-max [0.1, 1.5] → [0, 1] | Sigmoid 输出范围匹配 |
| GAN | **去掉 Discriminator** | T60 回归不需要对抗训练 |
| 损失函数 | α×denoise + β×T60 | 默认 α=0.1, β=1.0 (T60 为主任务) |
| AMP | 开启 | Conformer attention O(T²)，4 秒音频需要混合精度 |
| 梯度累积 | 8 步 × batch=1 | 显存限制：单卡 batch=1 约 15GB |

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
| `models/generator_t60.py` | **新增** | TSCNet_MultiTask = TSCNet + T60HeadWithPool |
| `dataset.py` | **新增** | T60_Dataset_v7 数据加载器 (复数 STFT + 多任务) |
| `train.py` | **新增** | 多任务训练 (AMP + 梯度累积 + SwanLab) |
| `test.py` | **新增** | test1-4 评估 (标准T60指标, JSON 与 /compare 兼容) |
| `utils.py` | 原始 CMGAN | power_compress/uncompress, kaiming_init, LearnableSigmoid |
| `train.sh` | **新增** | 一键训练脚本 |
| `test.sh` | **新增** | 一键测试脚本 |
| `requirements.txt` | **新增** | Python 依赖 |

## 数据集

**路径**: `/mnt/st16t/xxn/program/dataset/T60_Dataset_v7` (23GB)

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
- `{name}.wav` — 含噪混响 (模型输入)
- `{name}_denoised.wav` — 纯混响 (去噪 target)
- `{name}_noise.wav` — 噪声分量

关系: `noisy = clean_reverb + noise`，去噪任务是去除噪声保留混响。

## 运行命令

### 训练
```bash
conda activate demucs_xxn
# 单卡
CUDA_VISIBLE_DEVICES=1 python train.py -e CMGAN_t60_v1 --batch_size 1 --accum_steps 8
# 双卡
CUDA_VISIBLE_DEVICES=0,1 python train.py -e CMGAN_t60_v1 --batch_size 2 --accum_steps 8

# 或用脚本
./train.sh -e CMGAN_t60_v1 -g 1 -b 1
```

### 测试
```bash
python test.py --model_path runs/CMGAN_t60_v1/best_model.pth --gpu_id 0
```

### 训练超参数 (默认)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 1-2 | 受 Conformer 显存限制 |
| accum_steps | 8 | 等效 batch=8 或 16 |
| lr | 5e-4 | AdamW |
| max_epochs | 100 | 配合早停 patience=15 |
| α (去噪) | 0.1 | 去噪为辅助任务 |
| β (T60) | 1.0 | T60 为主任务 |
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

已有实验对比见: `/mnt/st16t/xxn/program/experiment_comparison.md`

## 已知问题

1. **显存**: 4 秒音频 + Conformer attention 导致 batch_size 受限。如需更大 batch，可：
   - 缩短音频 `--audio_length 2.0` (原始 CMGAN 用 2 秒)
   - 减少 TSCB 层数 (4→2)
   - 使用更大显存的 GPU
2. **STFT 参数差异**: CMGAN 用 n_fft=400, hop=100；其他实验多用 n_fft=320, hop=160。频率分辨率不同，结果不直接横向对比。
