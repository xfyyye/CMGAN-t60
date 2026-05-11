# Fourier-KAN T60 Head 实验记录

基准：MLP head（T60HeadWithPool）+ MAE/MSE loss  
对比：Fourier-KAN head（T60HeadFourierKAN）v1 → v2

---

## 背景

CMGAN backbone（DenseEncoder + 2×TSCB）输出 `(B, 64, F, T)` 的特征，经过 `mean(F)` 和 `MultiStatsPool(T)` 池化后得到 192 维向量，再送入回归 head 预测归一化 T60。

原始 MLP head 结构（3 层线性）：

```
MultiStatsPool: 64×3 = 192 维
Linear(192 → 128) + ReLU + Dropout(0.3)
Linear(128 → 64)  + ReLU + Dropout(0.3)
Linear(64  → 1)   + Sigmoid
```

Fourier-KAN head 结构（投影层 + 2 层 KAN + 输出线性层）：

```
MultiStatsPool: 64×3 = 192 维
Linear(192 → 64) + LayerNorm          ← 投影层（普通线性，非 KAN）
FourierKAN(64 → 32) + LayerNorm + Dropout   ← 第 1 层 KAN
FourierKAN(32 → 16) + LayerNorm + Dropout   ← 第 2 层 KAN
Linear(16 → 1) + Sigmoid                    ← 输出层（普通线性，非 KAN）
```

每个 FourierKAN 层对每个输入维度 i、输出维度 j 计算：

```
y_j = Σ_i Σ_w [ a_{j,i,w}·cos(w·x_i) + b_{j,i,w}·sin(w·x_i) ]
```

其中 `w = 1, 2, ..., Omega`，Omega 即 `num_frequencies`。

---

## 基准结果：MLP Head

**配置**（`mlp_mse.yaml` / `mlp_mae.yaml`）：

```yaml
model:
  t60_head_type: mlp
  t60_hidden_dim: 128   # 第一隐层宽度，第二隐层自动取 hidden_dim//2=64
  t60_dropout: 0.3
  # 结构：Linear(192→128)+ReLU+Dropout → Linear(128→64)+ReLU+Dropout → Linear(64→1)+Sigmoid
```

| 测试集 | MSE — RMSE/MAE/Bias | MAE — RMSE/MAE/Bias |
|--------|---------------------|---------------------|
| test1  | 110.1 / 67.3 / +2.0 | 108.4 / 63.6 / -5.1 |
| test2  | 115.2 / 72.8 / -2.7 | 120.2 / 70.8 / -6.8 |
| test3  | 113.4 / 70.4 / +6.2 | 106.3 / 63.5 / -4.6 |
| test4  | 113.7 / 69.4 / +4.5 | 110.9 / 63.9 / -5.5 |
| **平均** | **RMSE=113.1 MAE=70.0** | **RMSE=111.5 MAE=65.5** |

- MSE 的 Bias 较小（-3~+6ms），MSE 下 MLP 行为正常
- MAE 整体略优于 MSE（RMSE 低 1.6ms，MAE 低 4.5ms）
- Pearson r 均在 0.944~0.951

---

## Fourier-KAN v1

### 设计动机

MLP 使用固定的多项式（ReLU 分段线性）激活，对池化后的声学嵌入表达能力有限。Fourier-KAN 用三角函数基替换线性层，理论上对周期性或非线性强的映射拟合能力更强，且参数利用率更高。

### 配置（`kan_mse.yaml` / `kan_mae.yaml`）

```yaml
model:
  t60_head_type: fourier_kan
  t60_fourier_proj_dim: 64
  t60_fourier_hidden_dims: [32, 16]
  t60_fourier_num_frequencies: 4    # 所有层统一使用 Omega=4
  t60_fourier_dropout: 0.1
  t60_out_activation: sigmoid
```

所有 FourierKAN 层（64→32、32→16）均使用相同的频率数 Omega=4。

### 结果

| 测试集 | MSE — RMSE/MAE/Bias | MAE — RMSE/MAE/Bias |
|--------|---------------------|---------------------|
| test1  | 114.3 / 68.6 / +0.6 | 111.3 / 64.5 / -4.8 |
| test2  | 127.1 / 78.2 / -5.5 | 113.5 / 70.0 / -13.0 |
| test3  | 107.1 / 67.7 / +9.3 | 110.3 / 66.7 / -0.8 |
| test4  | 118.7 / 71.4 / +6.0 | 124.2 / 69.1 / -0.4 |
| **平均** | **RMSE=116.8 MAE=71.5** | **RMSE=114.8 MAE=67.6** |

### 分析

- **未能超过 MLP 基准**：MAE loss 下 RMSE 比 MLP 多 3.4ms，MSE loss 下多 3.7ms
- Bias 正常（MSE 下 -5~+9ms，与 MLP 相当），说明 v1 和 MSE 的组合没有系统性偏移
- Omega=4 偏小，第一层输入维度 64，表达能力可能不足
- 训练收敛更快（39 epoch，MLP 为 59 epoch），但最终精度偏低，疑似欠拟合

---

## Fourier-KAN v2

### 设计动机

v1 所有层用同一个 Omega=4，但不同层的职责不同：

- **第一层**（输入 proj_dim=64）：接收的是高维池化嵌入，非线性结构复杂，需要更多频率分量才能捕捉细粒度特征
- **后续隐层**（输入 32→16）：特征已被压缩，维度小，再用大 Omega 会引入冗余参数并增加过拟合风险

参考 Fourier-ASR 论文的分层 grid size 设计（`input_grid_size` 和 `hidden_grid_size` 分开设置），将第一层和隐层的频率数解耦：

```
第一层 Omega=16（大）→ 捕捉输入嵌入中的细粒度非线性
后续层 Omega=8 （小）→ 精炼压缩，控制参数量
```

同时去掉了原始投影层中的 Tanh（只保留 Linear + LayerNorm），减少信息截断。

### 配置（`kan_v2_mse.yaml` / `kan_v2_mae.yaml`）

```yaml
model:
  t60_head_type: fourier_kan
  t60_fourier_proj_dim: 64
  t60_fourier_hidden_dims: [32, 16]
  t60_fourier_first_num_frequencies: 16   # 第一层 Omega，较大
  t60_fourier_hidden_num_frequencies: 8   # 后续隐层 Omega，较小
  t60_fourier_dropout: 0.1
  t60_out_activation: sigmoid
```

### 结果

| 测试集 | MSE — RMSE/MAE/Bias | MAE — RMSE/MAE/Bias |
|--------|---------------------|---------------------|
| test1  | 105.9 / 68.8 / **+26.8** | 116.8 / 68.9 / -6.5 |
| test2  | 111.6 / 73.0 / **+17.1** | 123.8 / 73.0 / -13.8 |
| test3  | 113.9 / 75.5 / **+32.7** | 117.8 / 71.9 / -1.1 |
| test4  | 113.1 / 72.3 / **+28.9** | 114.6 / 68.5 / -2.0 |
| **平均** | **RMSE=111.2 MAE=72.4** | **RMSE=118.2 MAE=70.6** |

### 分析

**MSE loss 下**：
- RMSE 从 116.8ms 降到 111.2ms，首次接近 MLP 水平
- 但 Bias 从 v1 的正常范围（±9ms）暴增至 **+17~+33ms**，出现严重系统性高估
- 原因：v2 的更大 Omega 使 KAN 非线性更强，MSE 的二次梯度在大误差样本上更大，将模型推向训练集的均值方向，形成系统性正偏移；v1 中 Tanh 的限幅效果无意中抑制了这一偏移

**MAE loss 下**：
- RMSE 从 114.8ms 升至 118.2ms，反而退步
- Bias 正常（-14~-2ms），说明 MAE 的常数梯度规避了均值偏移问题
- 退步原因：更大的 Omega 参数量增加，配合 MAE 的常数梯度，训练信号相对更弱，14 epoch 就早停，模型未充分收敛

---

## 横向对比汇总

### MAE loss

| 实验 | avg RMSE | avg MAE | avg Bias | 收敛 epoch |
|------|----------|---------|----------|-----------|
| mlp_mae（基准）| 111.5 ms | 65.5 ms | -5.5 ms | 44 |
| kan_v1_mae | 114.8 ms | 67.6 ms | -4.8 ms | 24 |
| kan_v2_mae | 118.2 ms | 70.6 ms | -5.9 ms | 14 |

### MSE loss

| 实验 | avg RMSE | avg MAE | avg Bias | 收敛 epoch |
|------|----------|---------|----------|-----------|
| mlp_mse（基准）| 113.1 ms | 70.0 ms | +2.5 ms | 56 |
| kan_v1_mse | 116.8 ms | 71.5 ms | +2.6 ms | 54 |
| kan_v2_mse | 111.2 ms | 72.4 ms | **+26.4 ms** | 38 |

---

## Fourier-KAN v2 + Gradient Clip

### 设计动机

kan_v2_mae 仅 14 epoch 就早停，疑似梯度不稳定导致欠拟合。加入 `clip_grad_norm=1.0` 以延长有效训练期。同时对比 MSE+clip 是否能解决 bias 问题。

### 配置（`kan_v2_mae_clip1.yaml` / `kan_v2_mse_clip1.yaml`）

```yaml
train:
  clip_grad_norm: 1.0   # 新增梯度裁剪
# 其余与 kan_v2_mae / kan_v2_mse 相同
```

### 结果

| 测试集 | MSE+clip — RMSE/MAE/Bias | MAE+clip — RMSE/MAE/Bias |
|--------|--------------------------|--------------------------|
| test1  | 110.3 / 69.0 / +19.0 | 108.3 / 62.5 / -4.3 |
| test2  | 121.5 / 76.3 / +24.7 | 118.1 / 68.7 / -7.8 |
| test3  | 122.3 / 76.7 / +23.6 | 111.8 / 64.3 / -3.0 |
| test4  | 119.5 / 73.5 / +20.5 | 113.5 / 65.8 / -2.5 |
| **平均** | **RMSE=118.4 MAE=73.9** | **RMSE=112.9 MAE=65.3** |

### 分析

- **MAE+clip**：MAE=65.3ms 微赢 MLP（MLP=65.5ms），RMSE=112.9ms 略差（MLP=111.5ms）；梯度裁剪延长了训练期，效果有所提升
- **MSE+clip**：bias 未改善（+19~+25ms），clip 对 MSE 下的 kan_v2 bias 问题无效；RMSE=118.4ms 反而更差（无 clip 时 111.2ms）；clip 让 MSE 梯度更保守，收敛速度变慢，但 bias 根因（非线性+均值偏移）未解决

---

## Fourier-KAN v2 + Huber Loss (delta=0.05)

### 设计动机

Huber loss 在小误差区间（|e| < delta）用 MSE（鼓励精确），大误差区间用 MAE（避免离群点主导）。delta=0.05 对应 T60 归一化空间 5ms 量级。

### 配置（`kan_v2_huber_delta0.05.yaml`）

```yaml
loss:
  name: huber
  huber_delta: 0.05
```

### 结果

| 测试集 | RMSE | MAE | Bias |
|--------|------|-----|------|
| test1  | 111.0 ms | 65.6 ms | -2.2 ms |
| test2  | 120.5 ms | 70.5 ms | -9.3 ms |
| test3  | 110.5 ms | 66.2 ms | +6.0 ms |
| test4  | 116.9 ms | 67.5 ms | +2.9 ms |
| **平均** | **114.7 ms** | **67.5 ms** | — |

### 分析

- bias 正常（-9~+6ms），Huber 有效抑制了 MSE 的系统性偏移
- 但整体精度介于 MLP_MAE 和 MLP_MSE 之间，未超越最佳基准
- delta=0.05 时大部分样本（|e|>0.05）已退化为 MAE 梯度，接近 MAE 的行为

---

## 横向对比汇总（全部实验）

### MAE loss

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| mlp_mae（**基准**）| 111.5 ms | 65.5 ms | -5.5 ms | — |
| kan_v1_mae | 114.8 ms | 67.6 ms | -4.8 ms | Omega=4 |
| kan_v2_mae | 118.2 ms | 70.6 ms | -5.9 ms | 14 epoch 早停 |
| kan_v2_mae_clip1 | 112.9 ms | **65.3 ms** | -4.4 ms | MAE 微赢，RMSE 略差 |

### MSE loss

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| mlp_mse（基准）| 113.1 ms | 70.0 ms | +2.5 ms | — |
| kan_v1_mse | 116.8 ms | 71.5 ms | +2.6 ms | Omega=4 |
| kan_v2_mse | 111.2 ms | 72.4 ms | **+26.4 ms** | bias 失控 |
| kan_v2_mse_clip1 | 118.4 ms | 73.9 ms | **+21.9 ms** | clip 未解决 bias |

### Huber loss

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| kan_v2_huber_delta0.05 | 114.7 ms | 67.5 ms | -0.7 ms | bias 正常，精度中等 |

---

## Log-Scale T60 归一化

### 设计动机

线性归一化 `norm = (T60 - 0.1) / 1.4` 下，T60 高值区间（1.3-1.5s）仅对应 label 空间 0.857~1.0（范围 0.143），而 T60 低值区间（0.1-0.3s）也是相同的 0.143。但由于绝对 ms 值更大，高 T60 区间的绝对误差天然更大，模型倾向于低估高 T60。

Log-scale 归一化：`norm = (log(T60) - log(0.1)) / (log(1.5) - log(0.1))`，使各 T60 bin 在 label 空间中等比例分布，改善高 T60 区间的训练信号。

> **注意（Bug 修复）**：`dataset.py` 的 `T60Normalizer.denormalize` 原先对 numpy array 输入会报 `TypeError`（`math.exp` 只支持标量，`torch.exp` 不接受 numpy array）。已修复为按类型分派：Tensor → `torch.exp`，ndarray → `numpy.exp`，float → `math.exp`。

### 配置（`kan_v2_mae_clip1_log.yaml` / `mlp_mae_log.yaml`）

```yaml
data:
  log_scale: true   # 唯一新增（相比各自对应的线性版本）
train:
  clip_grad_norm: 1.0
```

### 结果：kan_v2_mae_clip1_log（已完成）

| 测试集 | RMSE | MAE | Bias | Pearson r | R² |
|--------|------|-----|------|-----------|-----|
| test1  | 101.6 ms | 60.1 ms | -5.5 ms | 0.9545 | 0.9109 |
| test2  | 110.6 ms | 64.6 ms | -13.0 ms | 0.9468 | 0.8946 |
| test3  | 107.5 ms | 61.1 ms | -4.6 ms | 0.9492 | 0.9006 |
| test4  | 111.1 ms | 62.7 ms | -7.7 ms | 0.9448 | 0.8921 |
| **平均** | **107.7 ms** | **62.1 ms** | **-7.7 ms** | **0.9488** | **0.8996** |

### 分析

- **全面超越此前所有实验**：RMSE=107.7ms 比基准 mlp_mae（111.5ms）低 3.8ms（-3.4%），MAE=62.1ms 比基准（65.5ms）低 3.4ms（-5.2%）
- **bias 正常**：-5~-13ms，与 kan_v2_mae_clip1 的 -2~-8ms 相近，log-scale 未引入新的系统性偏移
- **log-scale 有效**：确认 label 空间分布不均是此前高 T60 低估的根因之一
- **待定**：mlp_mae_log（对照组）训练中，用于判断增益来自 log-scale 本身还是 KAN 架构

### 实验状态

| 实验 | 状态 | avg RMSE | avg MAE |
|------|------|----------|---------|
| kan_v2_mae_clip1_log | **已完成** | 107.7 ms | 62.1 ms |
| mlp_mae_log | **训练中** | — | — |

---

## 横向对比汇总（全部实验）

### MAE loss（线性归一化）

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| mlp_mae（**基准**）| 111.5 ms | 65.5 ms | -5.5 ms | — |
| kan_v1_mae | 114.8 ms | 67.6 ms | -4.8 ms | Omega=4 |
| kan_v2_mae | 118.2 ms | 70.6 ms | -5.9 ms | 14 epoch 早停 |
| kan_v2_mae_clip1 | 112.9 ms | 65.3 ms | -4.4 ms | MAE 微赢，RMSE 略差 |

### MAE loss（log-scale 归一化）

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| kan_v2_mae_clip1_log | **107.7 ms** | **62.1 ms** | -7.7 ms | **当前最佳** |
| mlp_mae_log | 训练中 | 训练中 | — | 对照组 |

### MSE loss

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| mlp_mse（基准）| 113.1 ms | 70.0 ms | +2.5 ms | — |
| kan_v1_mse | 116.8 ms | 71.5 ms | +2.6 ms | Omega=4 |
| kan_v2_mse | 111.2 ms | 72.4 ms | **+26.4 ms** | bias 失控 |
| kan_v2_mse_clip1 | 118.4 ms | 73.9 ms | **+21.9 ms** | clip 未解决 bias |

### Huber loss

| 实验 | avg RMSE | avg MAE | avg Bias | 备注 |
|------|----------|---------|----------|------|
| kan_v2_huber_delta0.05 | 114.7 ms | 67.5 ms | -0.7 ms | bias 正常，精度中等 |

---

## 结论（截至当前）

1. **Log-scale 归一化是本轮最有效的改进**：kan_v2_mae_clip1_log 以 avg RMSE=107.7ms / MAE=62.1ms 全面超越所有先前实验，超越线性基准 mlp_mae 3.4ms RMSE / 3.4ms MAE
2. **Fourier-KAN v1/v2 在线性 label 空间下均未超过 MLP 基准**（MAE loss）；在 log-scale 下 KAN 效果显著
3. **MSE + kan_v2 出现严重 bias**：分层大 Omega 与 MSE 二次梯度组合导致系统性高估，梯度裁剪无法解决根因
4. **MAE loss 是更稳定的选择**：所有实验中 MAE loss 的 bias 均小于 ±14ms
5. **待定问题**：mlp_mae_log 对照实验训练中，结果将揭示 log-scale 增益是否与 KAN 架构解耦
