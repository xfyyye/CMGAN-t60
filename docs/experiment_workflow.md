# CMGAN T60 混响时间估计 — 完整实验流程文档

> **分支**: `exp/single-t60`
> **目标**: 将开源语音增强框架 CMGAN (Conformer-Based Metric GAN) 适配为**单任务 T60 混响时间盲估计**系统
> **任务定义**: 从含噪混响语音中盲估计 T60 值（范围 0.1~1.5 秒）

---

## 一、项目总览

### 1.1 项目定位

本项目是 `/adapt-t60` skill 的一次执行实例。该 skill 定义了将任意语音增强框架适配为 T60 估计的标准化流程。

核心改动：**去除 CMGAN 的语音增强 decoder 和 GAN 判别器**，在 backbone (DenseEncoder + TSCB) 之后接一个轻量 T60 回归头，实现端到端的 T60 盲估计。

### 1.2 文件结构

```
CMGAN_estimate_t60/
├── train.py                    # 主训练脚本
├── test.py                     # 评估脚本 (test1-test4)
├── dataset.py                  # T60_Dataset_v7 数据加载器
├── utils.py                    # 工具函数 (power_compress, kaiming_init, LearnableSigmoid)
├── requirements.txt            # Python 依赖
├── train.sh                    # 一键训练脚本
├── test.sh                     # 一键测试脚本
├── train-and-test.sh           # 训练+自动评估脚本
├── CLAUDE.md                   # 项目指南
├── README.md / README.zh.md    # 项目说明文档
│
├── models/
│   ├── conformer.py            # [原始] Conformer block (Shaw 相对位置编码, ConvModule)
│   ├── generator.py            # [原始] DenseEncoder, TSCB, DilatedDenseNet
│   ├── generator_t60.py        # [新增] TSCNet_T60Estimator, T60HeadWithPool, T60HeadFourierKAN
│   └── fourier_kan.py          # [新增] FourierKANLayer, FourierKANBlock
│
├── configs/t60_single/         # 15 个 YAML 实验配置
│   ├── mlp_mse.yaml            # MLP + MSE (基线)
│   ├── mlp_mae.yaml            # MLP + MAE (基线)
│   ├── mlp_huber.yaml          # MLP + Huber (基线)
│   ├── mlp_mae_log.yaml        # MLP + MAE + log-scale
│   ├── kan_mse.yaml            # Fourier-KAN v1 + MSE
│   ├── kan_mae.yaml            # Fourier-KAN v1 + MAE
│   ├── kan_huber.yaml          # Fourier-KAN v1 + Huber
│   ├── kan_v2_mse.yaml         # Fourier-KAN v2 + MSE
│   ├── kan_v2_mae.yaml         # Fourier-KAN v2 + MAE
│   ├── kan_v2_mse_clip1.yaml   # Fourier-KAN v2 + MSE + clip=1.0
│   ├── kan_v2_mae_clip1.yaml   # Fourier-KAN v2 + MAE + clip=1.0
│   ├── kan_v2_mse_log.yaml     # Fourier-KAN v2 + MSE + log-scale
│   ├── kan_v2_huber_delta0.05.yaml  # Fourier-KAN v2 + Huber(δ=0.05)
│   ├── kan_v2_mae_clip1_log.yaml     # Fourier-KAN v2 + MAE + clip + log (🏆最佳)
│   ├── kan_v2_mae_clip1_log_1kan.yaml
│   └── kan_v2_mae_clip1_1kan.yaml
│
├── docs/
│   ├── fourier_kan_implementation_notes.md   # v1→v2→v2.1→v2.2 完整实现历史
│   ├── fourier_kan_v1v2_analysis.md          # v1/v2 与损失函数交互分析
│   ├── t60_fourier_kan_coding_plan.md        # Fourier-KAN head 编码计划
│   └── CMGAN-t60_git_branch_8gpu_experiment_plan.md  # 8-GPU 并行实验计划
│
├── runs/                       # 实验输出目录 (每个实验一个子目录)
│   ├── single_t60_mlp_mse/
│   ├── single_t60_mlp_mae/
│   ├── single_t60_kan_v2_mae_clip1_log/  # 最佳实验
│   └── ...                       # 共 14 个实验
│
└── swanlog/                    # SwanLab 实验追踪日志
```

### 1.3 文件来源标注

| 文件 | 来源 | 说明 |
|------|------|------|
| `models/conformer.py` | 原始 CMGAN | Conformer block, 未修改 |
| `models/generator.py` | 原始 CMGAN | DenseEncoder, TSCB, 未修改 |
| `models/generator_t60.py` | **新增** | T60 估计器 + 两种回归头 |
| `models/fourier_kan.py` | **新增** | Fourier-KAN 层与块 |
| `dataset.py` | **新增** | T60_Dataset_v7 数据加载 |
| `train.py` | **新增** | 训练脚本 |
| `test.py` | **新增** | 评估脚本 |
| `utils.py` | 原始 CMGAN | power_compress 等工具函数 |

---

## 二、模型架构

### 2.1 架构对比：原始 CMGAN → T60 估计版

```
原始 CMGAN (TSCNet)                    单任务 T60 版本
┌────────────────┐                     ┌────────────────┐
│ DenseEncoder   │                     │ DenseEncoder   │ ← 原始代码，未改
│ (260K params)  │                     │ (260K params)  │
├────────────────┤                     ├────────────────┤
│ TSCB × 4       │                     │ TSCB × N       │ ← 默认 N=2，可选 4
│ (1.03M params) │                     │       │        │
│                │                     │       ├────────┤
│                │                     │       │T60Head │ ← 唯一新增
│                │                     │       │(Sigmoid)│
├────────────────┤                     └────────────────┘
│ MaskDecoder    │                     ❌ 已移除
│ ComplexDecoder │                     ❌ 已移除
└────────────────┘
Discriminator (GAN)                    ❌ 已去除
```

### 2.2 数据流维度（4 秒音频, 16kHz）

```
输入: noisy_reverb.wav (B, 64000) @ 16kHz, 4s
  │
  ├─ 能量归一化: c = sqrt(T / sum(wav²))  →  wav *= c
  │
  ├─ STFT(n_fft=400, hop=100, window=hamming)
  │   → (B, 2, 641, 201)  [real, imag]
  │
  ├─ 功率压缩: power_compress (mag^0.3, 保留相位)
  │   → (B, 2, 641, 201) [压缩后]
  │
  ├─ 提取幅度: mag = sqrt(real² + imag²)
  ├─ 拼接 3 通道: [mag, real, imag]
  │   → (B, 3, 641, 201)
  │
  ├─ DenseEncoder (3ch → 64ch, F stride=2 下采样)
  │   → (B, 64, 641, 101)
  │
  ├─ TSCB × N (双分支 Conformer: time + freq)
  │   → (B, 64, 641, 101)
  │
  └─ T60 回归头 → t60_pred (B,) 归一化值 [0,1]
      └─ 反归一化 → T60 (秒)
```

### 2.3 模型变体

#### TSCNet_T60Estimator（4 TSCB, ~1.87M 参数）

```python
# generator_t60.py: TSCNet_T60Estimator
DenseEncoder(in_channel=3, channels=64)  # DilatedDenseNet depth=4
TSCB × 4                                  # 每层: time_conformer + freq_conformer
t60_head                                  # 可选 MLP 或 Fourier-KAN
```

#### TSCNet_T60Estimator_2TSCB（2 TSCB, ~1.35M 参数, 当前默认）

```python
# generator_t60.py: TSCNet_T60Estimator_2TSCB
DenseEncoder(in_channel=3, channels=64)
TSCB × 2                                  # 计算量减半，训练速度约 2-3x
t60_head
```

### 2.4 T60 回归头设计

#### T60HeadWithPool (MLP) — ~33K 参数

```
(B, 64, F, T)
  → mean(F)                 → (B, 64, T)
  → MultiStatsPool(avg+max+std, T) → (B, 192)   [64×3=192]
  → Linear(192, 128) + ReLU + Dropout(0.3)
  → Linear(128, 64)  + ReLU + Dropout(0.3)
  → Linear(64, 1)    + Sigmoid              → (B,)
```

#### T60HeadFourierKAN (v2) — ~86K 参数

```
(B, 64, F, T)
  → mean(F)                 → (B, 64, T)
  → MultiStatsPool(avg+max+std, T) → (B, 192)
  → Linear(192, 64) + LayerNorm         (投影层，无 Tanh)
  → FourierKAN(64→32, Ω=16) + LayerNorm + Dropout(0.1)
  → FourierKAN(32→16, Ω=8)  + LayerNorm + Dropout(0.1)
  → Linear(16, 1) + Sigmoid             → (B,)
```

**FourierKANLayer 核心公式**:
```
y_j = Σ_i Σ_w [a_{j,i,w}·cos(w·x_i) + b_{j,i,w}·sin(w·x_i)] + bias_j
```
实现使用 `einsum("bik,oik->bo")` 高效计算。

**v1 → v2 关键区别**:
| 特征 | v1 | v2 |
|------|----|----|
| Omega | 统一 4 | 分层 16/8 (首层大, 隐层小) |
| 投影层激活 | Tanh | 无 (仅 LayerNorm) |
| 理论依据 | - | 对齐 Fourier-ASR 分层 gridsize |

### 2.5 上游组件（原始 CMGAN，未修改）

- **DenseEncoder**: Conv2d(3→64) → DilatedDenseNet(depth=4, dilation=1,2,4,8) → Conv2d(64→64, F stride=2)
- **TSCB**: 双分支 Conformer，reshape (B,C,T,F) 为 (B×F,T,C) 做 time attention，(B×T,F,C) 做 freq attention。每层含 FFN(0.5x) → PreNorm+Attention(4 heads) → ConvModule(kernel=31) → FFN(0.5x) → LayerNorm

---

## 三、数据流程

### 3.1 数据集: T60_Dataset_v7 (23GB)

```
T60_Dataset_v7/
  train/      40,000 samples
  eval/        5,136 samples
  test1/       1,080 samples (simulated RIR + seen noise)
  test2/       1,080 samples (real RIR + seen noise)
  test3/       1,080 samples (simulated RIR + unseen noise)
  test4/       1,080 samples (real RIR + unseen noise)
```

**样本目录结构**:
```
speech000001_reverb_0.214_0dB/
  ├── speech000001_reverb_0.214_0dB.wav          ← 含噪混响 (模型输入)
  ├── speech000001_reverb_0.214_0dB_denoised.wav  ← 纯混响 (本分支忽略)
  └── speech000001_reverb_0.214_0dB_noise.wav     ← 噪声分量 (本分支忽略)
```

**标签提取** (`parse_sample_name`):
- `speech000001_reverb_0.214_0dB` → T60=0.214s, SNR=0dB
- T60 范围: [0.1, 1.5] 秒
- SNR 值: -5, 0, 5, 10, 15, 20 dB

### 3.2 单样本数据处理流程

```python
# dataset.py: CMGANT60Dataset.__getitem__

1. 加载 WAV (torchaudio, 16kHz mono)
2. 验证固定长度 (4s = 64000 samples, 严格校验)
3. 能量归一化: c = sqrt(T / (sum(wav²) + 1e-8)), wav *= c
4. 复数 STFT: torch.stft(n_fft=400, hop=100, window=hamming)
   → (2, T, F) [real, imag]
5. T60 标签归一化: min-max [0.1, 1.5] → [0, 1]
   - 线性: norm = (T60 - 0.1) / 1.4
   - 对数: norm = (log(T60) - log(0.1)) / (log(1.5) - log(0.1))
```

### 3.3 T60Normalizer

```python
class T60Normalizer:
    # 线性模式 (log_scale=False):
    normalize: (T60 - 0.1) / 1.4        → [0, 1]
    denormalize: norm * 1.4 + 0.1       → [0.1, 1.5]

    # 对数模式 (log_scale=True):
    normalize: (log(T60) - log(0.1)) / (log(1.5) - log(0.1))  → [0, 1]
    denormalize: exp(norm * (log(1.5) - log(0.1)) + log(0.1))  → [0.1, 1.5]
```

### 3.4 数据集路径配置

优先级: CLI `--dataset_root` > YAML `data.dataset_root` > 环境变量 `T60_DATASET_ROOT`

当前配置: `/mnt/tidal-sh01/usr/chuan/youling/project/dataset/T60_Dataset_v7_4s`

---

## 四、训练流程

### 4.1 训练脚本入口

```bash
# 仅训练
./train.sh -c configs/t60_single/kan_v2_mae_clip1_log.yaml -g 0,1

# 训练后自动测试 test1-test4
./train-and-test.sh -c configs/t60_single/kan_v2_mae_clip1_log.yaml -g 0,1

# 临时覆盖参数
./train.sh -c configs/t60_single/mlp_mse.yaml -d /path/to/data -- --batch_size 2
```

### 4.2 训练脚本内部流程

```
train.py::train(args)
  │
  ├─ 1. 加载 YAML 配置为默认参数 (load_config_defaults)
  ├─ 2. 解析 CLI 参数 (YAML 作为默认值，CLI 可覆盖)
  ├─ 3. 解析数据集路径 (resolve_dataset_root)
  ├─ 4. 设备检测 & 多 GPU DataParallel
  ├─ 5. SwanLab 实验追踪初始化
  ├─ 6. 创建 DataLoader (train/eval)
  │     create_dataloaders() → CMGANT60Dataset → collate_fn
  ├─ 7. 构建模型 (TSCNet_T60Estimator / _2TSCB)
  │     根据 n_tscb 选择，t60_head_type 选择 MLP/FourierKAN
  ├─ 8. 损失函数: T60Loss (MSE / MAE / Huber)
  ├─ 9. 优化器: AdamW + ReduceLROnPlateau
  ├─ 10. AMP: torch.amp.autocast + GradScaler
  │
  ├─ 训练循环 (每个 epoch):
  │   ├─ model.train()
  │   ├─ for batch in train_loader:
  │   │   ├─ 功率压缩: power_compress(STFT)
  │   │   ├─ AMP 前向: model(noisy_input) → t60_pred
  │   │   ├─ 损失计算 + 梯度累积缩放
  │   │   ├─ 梯度裁剪: clip_grad_norm_(max_norm=clip_grad_norm)
  │   │   └─ 累积完成后: scaler.step + optimizer.zero_grad
  │   ├─ model.eval()
  │   ├─ 验证 (no_grad, AMP)
  │   ├─ 学习率调度: scheduler.step(val_loss)
  │   ├─ 早停检查: EarlyStopping(patience=15)
  │   └─ SwanLab 日志记录
  │
  ├─ 保存训练历史: training_history.json
  ├─ 加载最佳模型: best_model.pth
  └─ (若非 --train_only) 自动调用 test.run_tests()
```

### 4.3 关键训练超参数

| 参数 | 默认值 (H800 配置) | 说明 |
|------|-----|------|
| `batch_size` | 32 | H800 80GB; 旧配置为 4 (显存受限 GPU) |
| `accum_steps` | 1 | H800 配置; 旧配置为 8 (等效 batch=32) |
| `lr` | 5e-4 | AdamW |
| `weight_decay` | 1e-5 | |
| `max_epochs` | 100 | |
| `early_stop_patience` | 15 | |
| `clip_grad_norm` | 5.0 (默认) / 1.0 (最佳实验) | |
| `loss` | MSE (默认) / MAE (最佳实验) | 可选 mse / mae / huber |
| `huber_delta` | 1.0 (默认) / 0.05 (实验) | |
| `log_scale` | False (默认) / True (最佳实验) | T60 标签对数归一化 |
| `n_tscb` | 2 | 可选 2 或 4 |
| `num_workers` | 16 | |
| `n_fft` | 400 | CMGAN 原始参数 |
| `hop_length` | 100 | CMGAN 原始参数 |
| `audio_length` | 4.0 秒 | |
| `target_sr` | 16000 Hz | |

### 4.4 损失函数

```python
class T60Loss(nn.Module):
    # MSE:  F.mse_loss(pred, target)
    # MAE:  F.l1_loss(pred, target)
    # Huber: F.huber_loss(pred, target, delta=huber_delta)
```

### 4.5 早停策略

```python
class EarlyStopping:
    patience = 15        # 15 个 epoch 无改善则停止
    mode = 'min'         # 监控 val_loss (越小越好)
    min_delta = 1e-4     # 最小改善阈值
    save_path = 'best_model.pth'  # 自动保存最佳模型
```

### 4.6 学习率调度

```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='min',
    factor=0.5,     # LR 减半
    patience=5,     # 5 epoch 无改善则降低 LR
    min_lr=1e-6     # 最低学习率
)
```

### 4.7 梯度累积机制

```python
# 有效 batch = batch_size × accum_steps
# H800: batch=32, accum=1 → 有效 batch=32
# 旧配置: batch=4, accum=8 → 有效 batch=32
scaled_loss = loss / accum_steps
scaler.scale(scaled_loss).backward()
# 每 accum_steps 步才更新参数
if (step_idx + 1) % accum_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 五、评估流程

### 5.1 评估脚本入口

```bash
# 训练后自动评估 (train-and-test.sh 内部调用)
# 或手动评估:
./test.sh -c configs/t60_single/kan_v2_mae_clip1_log.yaml \
          -m runs/single_t60_kan_v2_mae_clip1_log/best_model.pth -g 0
```

### 5.2 评估流程

```
test.py::run_tests(args)
  │
  ├─ 1. 加载最佳模型 checkpoint
  ├─ 2. 创建 T60Normalizer (线性或对数模式)
  │
  ├─ for test_split in ['test1', 'test2', 'test3', 'test4']:
  │   ├─ 创建 test DataLoader
  │   ├─ 推理 (no_grad, AMP)
  │   ├─ 反归一化: t60_normalizer.denormalize(pred)
  │   ├─ 计算标准指标
  │   ├─ 计算 T60 分组指标 (7 个区间)
  │   ├─ 计算 SNR 分组指标 (6 个等级)
  │   └─ 保存 JSON: evaluation_results_{split}.json
  │
  ├─ 四测试集平均
  └─ 保存汇总: evaluation_summary.json
```

### 5.3 评估指标

#### 总体指标
| 指标 | 公式 |
|------|------|
| RMSE (ms) | sqrt(mean((pred-true)²)) × 1000 |
| MAE (ms) | mean(|pred-true|) × 1000 |
| Bias (ms) | mean(pred-true) × 1000 |
| Pearson r | scipy.stats.pearsonr(true, pred) |
| R² | 1 - Σ(pred-true)² / Σ(true-mean(true))² |
| Mean Rel Error (%) | mean(|pred-true| / |true|) × 100 |

#### T60 分组分析 (7 个区间)

| 区间标签 | 范围 |
|---------|------|
| T60_0.1-0.3s | [0.1, 0.3) |
| T60_0.3-0.5s | [0.3, 0.5) |
| T60_0.5-0.7s | [0.5, 0.7) |
| T60_0.7-0.9s | [0.7, 0.9) |
| T60_0.9-1.1s | [0.9, 1.1) |
| T60_1.1-1.3s | [1.1, 1.3) |
| T60_1.3-1.5s | [1.3, 1.5) |

每组统计: count, RMSE, MAE, Bias, Pearson r

#### SNR 分组分析 (6 个等级)

| SNR 等级 | 值 |
|---------|-----|
| SNR_-5dB | -5 |
| SNR_0dB | 0 |
| SNR_5dB | 5 |
| SNR_10dB | 10 |
| SNR_15dB | 15 |
| SNR_20dB | 20 |

每组统计: count, RMSE, MAE, Bias, Pearson r

### 5.4 输出格式

JSON 格式兼容 `/compare` skill 的 **Format B**:

```json
{
  "experiment_name": "single_t60_kan_v2_mae_clip1_log",
  "dataset_version": "T60_Dataset_v7",
  "test_split": "test1",
  "standard_metrics": { "rmse_ms": ..., "mae_ms": ..., "bias_ms": ..., ... },
  "bin_analysis": { ... },
  "snr_analysis": { ... },
  "per_sample_results": [
    { "sample_name": "...", "true_t60": ..., "pred_t60": ..., "snr": ..., "error_ms": ... }
  ]
}
```

---

## 六、Shell 脚本说明

### 6.1 train.sh — 仅训练

```bash
./train.sh -c <config.yaml> -g <gpu_ids> [-d <dataset_path>]
```
- 激活 conda 环境 `demucs_xxn`
- 设置 `CUDA_VISIBLE_DEVICES`
- 传递 `--train_only` 给 train.py (训练完不测试)

### 6.2 train-and-test.sh — 训练 + 自动测试

```bash
./train-and-test.sh -c <config.yaml> -g <gpu_ids> [-d <dataset_path>]
```
- 与 train.sh 相同参数
- **不传** `--train_only`，训练完自动跑 test1-test4

### 6.3 test.sh — 仅测试

```bash
./test.sh -c <config.yaml> -m <model.pth> -g <gpu_id>
```
- 必须指定 `-m` 为模型 checkpoint 路径

---

## 七、实验结果汇总

### 7.1 最佳结果: KAN v2 + MAE + clip=1.0 + log-scale

| 测试集 | RMSE (ms) | MAE (ms) | Bias (ms) | Pearson r | R² |
|--------|-----------|----------|-----------|-----------|------|
| test1 (sim RIR + seen noise) | 101.63 | 60.08 | -5.47 | 0.9545 | 0.9109 |
| test2 (real RIR + seen noise) | 110.55 | 64.64 | -12.96 | 0.9468 | 0.8946 |
| test3 (sim RIR + unseen noise) | 107.53 | 61.13 | -4.56 | 0.9492 | 0.9006 |
| test4 (real RIR + unseen noise) | 111.10 | 62.71 | -7.69 | 0.9448 | 0.8921 |
| **四集平均** | **107.70** | **62.14** | **-7.67** | **0.9488** | **0.8996** |

### 7.2 全部实验对比 (四集平均)

| 实验 | Head | Loss | 特殊设置 | RMSE (ms) | MAE (ms) | Bias (ms) |
|------|------|------|---------|-----------|----------|-----------|
| mlp_mse | MLP | MSE | — | 113.1 | 70.0 | +2.5 |
| mlp_mae | MLP | MAE | — | 111.5 | 65.5 | -5.5 |
| mlp_huber | MLP | Huber | δ=1.0 | — | — | — |
| mlp_mae_log | MLP | MAE | log-scale | 121.9 | 74.0 | +8.7 |
| kan_mse | KAN v1 | MSE | Ω=4 | 116.8 | 71.5 | +2.6 |
| kan_mae | KAN v1 | MAE | Ω=4 | 114.8 | 67.6 | -4.8 |
| kan_huber | KAN v1 | Huber | δ=1.0 | — | — | — |
| kan_v2_mse | KAN v2 | MSE | Ω=16/8 | 111.2 | 72.4 | **+26.4** |
| kan_v2_mae | KAN v2 | MAE | Ω=16/8 | 118.2 | 70.6 | -5.8 |
| kan_v2_mse_clip1 | KAN v2 | MSE | clip=1.0 | 118.4 | 73.9 | +13.3 |
| kan_v2_mae_clip1 | KAN v2 | MAE | clip=1.0 | 112.9 | 65.3 | +1.5 |
| kan_v2_huber_δ0.05 | KAN v2 | Huber | δ=0.05 | 114.7 | 67.5 | -0.7 |
| kan_v2_mse_log | KAN v2 | MSE | log-scale | — | — | — |
| **kan_v2_mae_clip1_log** | **KAN v2** | **MAE** | **clip=1.0+log** | **107.7** | **62.1** | **-7.7** |
| kan_v2_mae_clip1_1kan | KAN v2 | MAE | clip=1.0, 1层KAN | — | — | — |
| kan_v2_mae_clip1_log_1kan | KAN v2 | MAE | clip=1.0+log, 1层KAN | — | — | — |

### 7.3 迭代优化链

```
kan_v1_mae → kan_v2_mae → kan_v2_mae_clip1 → kan_v2_mae_clip1_log
  MAE=67.59    MAE=70.57     MAE=65.3           MAE=62.1  🏆
```

每一步解决了上一步引入的问题:
1. **v1→v2**: Omega 从 4 增大到 16/8，增强表达力
2. **v2→v2+clip**: 大 Omega + MSE 梯度不稳定，引入梯度裁剪
3. **v2+clip→+log**: 标签空间分布不均，引入对数归一化

---

## 八、关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 功率压缩 | mag^0.3 (power_compress) | CMGAN 原始设计，Conv2d 自适应数值范围 |
| 输入格式 | 复数 STFT [mag, real, imag] → 3ch | 保留相位信息 |
| 能量归一化 | c = sqrt(T / Σwav²) | 沿用 CMGAN 原始设计 |
| T60 归一化 | min-max [0.1,1.5] → [0,1] 或 log | 匹配 Sigmoid 输出范围 |
| GAN | 去掉 Discriminator | 回归任务不需要对抗训练 |
| Enhancement decoder | 去掉 MaskDecoder/ComplexDecoder | 单任务分支 |
| AMP | 开启 | Conformer attention O(T²) |
| Backbone 冻结 | 不冻结 | 端到端全量训练 |
| TSCB 层数 | 默认 2 层 | 平衡性能与显存 |

### 显存分析

Conformer self-attention 为 O(T²)：
- T=641, F=101 时，每个 TSCB 对 (B×101, 641, 64) 做 attention
- batch=1 时单层 attention 约 330MB
- 4 层 × 2 (time+freq) = 8 层 attention

---

## 九、实验追踪

### 9.1 SwanLab

训练脚本集成了 SwanLab 实验追踪，记录以下指标:
- `train_loss`, `val_loss`
- `learning_rate`
- `train_t60_loss`, `val_t60_loss`

日志保存在 `swanlog/` 目录下，每个实验包含 `config.yaml`、`requirements.txt`、`swanlab-metadata.json`。

### 9.2 输出目录结构

```
runs/{experiment_name}/
  ├── best_model.pth           # 最佳模型 checkpoint
  ├── training_history.json    # 训练历史 (每 epoch 的 loss, lr 等)
  └── test/                    # 评估结果
      ├── evaluation_results_test1.json
      ├── evaluation_results_test2.json
      ├── evaluation_results_test3.json
      ├── evaluation_results_test4.json
      └── evaluation_summary.json
```

---

## 十、配置系统

### 10.1 YAML 配置结构

```yaml
experiment:
  name: single_t60_kan_v2_mae_clip1_log    # 实验名 (决定输出目录)
  output_root: runs                          # 输出根目录

model:
  n_tscb: 2                                  # TSCB 层数
  t60_head_type: fourier_kan                 # 回归头类型
  t60_fourier_proj_dim: 64                   # Fourier-KAN 投影维度
  t60_fourier_hidden_dims: [32, 16]          # KAN 隐层维度序列
  t60_fourier_first_num_frequencies: 16      # 第一层 Omega
  t60_fourier_hidden_num_frequencies: 8      # 隐层 Omega
  t60_fourier_dropout: 0.1                   # KAN Dropout
  t60_out_activation: sigmoid                # 输出激活

data:
  dataset_root: /path/to/T60_Dataset_v7_4s
  n_fft: 400
  hop_length: 100
  audio_length: 4.0
  target_sr: 16000
  t60_min: 0.1
  t60_max: 1.5
  log_scale: true                            # 对数归一化

train:
  batch_size: 32
  accum_steps: 1
  max_epochs: 100
  lr: 0.0005
  weight_decay: 0.00001
  early_stop_patience: 15
  num_workers: 16
  clip_grad_norm: 1.0                        # 梯度裁剪

test:
  batch_size: 8
  num_workers: 4
  gpu_id: 0

loss:
  name: mae

logging:
  swanlab_project: T60_Estimation
```

### 10.2 参数优先级

```
CLI 参数 > YAML 配置 > 代码默认值
```

特殊: `--dataset_root` / `-d` > YAML `data.dataset_root` > 环境变量 `T60_DATASET_ROOT`

---

## 十一、关键发现与实验洞察

### 11.1 核心发现

1. **对数归一化是最有效的改进**: kan_v2_mae_clip1_log 超越所有先前实验
2. **Log-scale 仅对 KAN 有效，对 MLP 有害**: Fourier 基函数与对数空间标签分布存在协同效应
3. **MSE + KAN v2 产生严重正偏差**: 分层大 Omega + MSE 二次梯度导致系统性高估 (Bias=+26.4ms)
4. **MAE 损失更稳定**: 所有实验中 Bias 保持在 ±14ms 以内
5. **梯度裁剪与对数归一化有叠加效果**: clip 解决优化稳定性，log-scale 重新平衡标签空间

### 11.2 各变量的独立效应

| 变量 | 效果 |
|------|------|
| KAN v1 vs MLP | 略差 (v1 Omega=4 太小) |
| KAN v2 vs v1 | 表达力更强但梯度不稳定 |
| clip=1.0 | 稳定 KAN v2 训练 |
| log-scale | 对 KAN 显著提升，对 MLP 反而退化 |
| MAE vs MSE | MAE 更鲁棒 (Bias 更低) |

---

## 十二、已知问题

1. **显存限制**: 4 秒音频 + Conformer attention → batch_size 受限。可缩短音频到 2 秒或减少 TSCB 层数
2. **STFT 参数差异**: CMGAN 用 n_fft=400, hop=100；其他实验多用 n_fft=320, hop=160。结果不直接横向对比
3. **系统性负偏差**: 最佳实验仍有 -7.67ms 的平均负偏差 (低估 T60)

---

## 十三、后续方向

| 方向 | 优先级 | 成本 | 说明 |
|------|--------|------|------|
| 频率感知池化 | 最高 | 最低 | 替换 mean(F) 为可学习频带池化 |
| CCC 损失 | 中 | 低 | 解决系统性偏差 |
| SpecAugment | 中 | 低 | 增强对未见噪声的鲁棒性 |
| 多任务 (T60 + EDC) | 低 | 高 | 更大改动，更高上限 |

---

## 十四、依赖环境

```
# requirements.txt
torch>=2.0
torchaudio
numpy
scipy
pyyaml
swanlab        # 可选，实验追踪
```

Conda 环境: `demucs_xxn`

GPU: H800 80GB (当前配置 batch=32, accum=1) 或普通 GPU (batch=4, accum=8)
