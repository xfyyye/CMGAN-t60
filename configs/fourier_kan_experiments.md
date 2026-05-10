# Fourier-KAN T60 Head 实验记录

基准：MLP head（T60HeadWithPool）+ MAE/MSE loss  
对比：Fourier-KAN head（T60HeadFourierKAN）v1 → v2

---

## 背景

CMGAN backbone（DenseEncoder + 2×TSCB）输出 `(B, 64, F, T)` 的特征，经过 `mean(F)` 和 `MultiStatsPool(T)` 池化后得到 192 维向量，再送入回归 head 预测归一化 T60。

原始 MLP head 结构：`FC(192→128→64→1) + Sigmoid`

Fourier-KAN head 替换了 MLP 部分，在投影层之后使用 FourierKAN 层：

```
MultiStatsPool(192) → Linear(192→proj_dim) + LayerNorm
    → FourierKANBlock(hidden_dims) → Linear(→1) → Sigmoid
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
  # FC(192→128→64→1) + Sigmoid，Dropout=0.3
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

## 结论

1. **Fourier-KAN v1/v2 在当前配置下均未超过 MLP 基准**（MAE loss 下）
2. **MSE + kan_v2 出现严重 bias**：分层大 Omega 增强了非线性，与 MSE 的二次梯度组合，导致模型系统性高估；v1 和 MLP 的 MSE bias 均正常（±9ms 以内）
3. **MAE loss 是更稳定的选择**：所有实验中 MAE loss 的 bias 均小于 ±14ms，MSE 在 kan_v2 下失控
4. **后续改进方向**：在 kan_v2_mae 基础上加梯度裁剪（clip_grad_norm=1.0）以稳定训练，同时探索 log-scale T60 归一化以改善高 T60 区间的系统性低估
