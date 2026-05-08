# Fourier-KAN T60 头实现说明

本文记录把 MLP T60 回归头替换为 Fourier-KAN 的完整改动历史和设计决策。

---

## Fourier-KAN 是什么

普通 MLP 的每一层是：

```
y = activation(W · x + b)
```

Fourier-KAN 把"权重 × 激活"换成"每条边上的可学习 Fourier 展开"：

```
y_j = Σ_i φ_{j,i}(x_i)

φ_{j,i}(x_i) = Σ_{ω=1}^{Ω} [ a_{j,i,ω} cos(ω·x_i) + b_{j,i,ω} sin(ω·x_i) ]
```

MLP 用一个全局激活函数处理所有输入，KAN 给每条输入→输出的连接分配一组频率可调的基函数，更适合捕捉细粒度的非线性关系。

单层参数量：`out × in × Ω × 2 + out`

---

## 代码结构

```
models/
  fourier_kan.py
    FourierKANLayer     # 单层 Fourier-KAN
    FourierKANBlock     # 堆叠多层 + LayerNorm + Dropout，支持分层 Omega

  generator_t60.py
    MultiStatsPool      # 原有，保留
    T60HeadWithPool     # 原有 MLP head，保留不变
    T60HeadFourierKAN   # KAN head
    build_t60_head()    # 工厂函数，按 head_type 返回对应 head
    TSCNet_T60Estimator      # 支持 t60_head_type 等参数，默认 "mlp"
    TSCNet_T60Estimator_2TSCB
```

---

## 与 Fourier-ASR 参考实现的对比

参考仓库本地路径：`/mnt/tidal-sh01/usr/chuan/youling/project/Fourier-ASR`

### 核心计算完全等价

两者都对每条 `(输入维 i → 输出维 j)` 的连接展开 Ω 个频率的 cos+sin 后求和，只是实现路径不同：原版用 reshape + 广播，我们用 einsum，数值结果相同。

### 逐项差异

| 方面 | Fourier-ASR 原版 | 我们的实现 | 影响 |
|------|-----------------|-----------|------|
| **权重存储** | 一个 `fouriercoeffs (2, O, I, K)`，第0维cos第1维sin | 拆成 `cos_weight` 和 `sin_weight` 两个 Parameter | 无影响，参数量相同 |
| **bias shape** | `(1, outdim)` | `(outdim,)` | 无影响 |
| **初始化** | 默认 norm，另支持 uniform、rand | 固定 norm，公式相同 | 等价 |
| **输入维度** | 支持任意 `(..., inputdim)`，内部 reshape | 严格要求 `(B, in_features)` 2D | 对 T60 head 场景无影响 |
| **层间 LayerNorm / Dropout** | 无 | 有，用于声学 embedding 场景的稳定性 | 有意扩展 |
| **分层 Omega** | 第一层 `input_grid_size=1024`，隐层 `hidden_grid_size=5` | v1 全层统一 Ω；v2 分层设置 | v2 对齐原版设计 |
| **proj 层激活** | 无（输入天然归一化） | v1 加了 Tanh；v2 去掉 Tanh | v2 对齐原版 |

### 原版的输入为什么不需要 Tanh

Fourier-ASR 的任务是音频隐式神经表示（INR）：输入是归一化时间坐标 `t ∈ [-1, 1]`，直接送进 KAN，不需要任何预处理。

我们的输入是 pooled acoustic embedding，需要 `Linear + LayerNorm` 先控制幅度，但 **Tanh 是多余的**——LayerNorm 之后幅度已经受控，再加 Tanh 反而截断了 embedding 中有用的幅度信息（混响越长，能量衰减越慢，这部分差异对 T60 估计有意义）。

---

## 版本历史

### v1：初始实现（`kan_*.yaml`）

**改动文件：**

| 文件 | 操作 |
|------|------|
| `models/fourier_kan.py` | 新建，`FourierKANLayer` + `FourierKANBlock` |
| `models/generator_t60.py` | 新增 `T60HeadFourierKAN`、`build_t60_head`；两个 TSCNet 类加 `t60_head_type` 参数 |
| `train.py` / `test.py` | 新增 KAN 相关 argparse 参数，YAML 读取，打印 head 参数量 |
| `configs/t60_single/kan_mse.yaml` | 新建 |
| `configs/t60_single/kan_mae.yaml` | 新建 |
| `configs/t60_single/kan_huber.yaml` | 新建 |

**v1 的 T60 head 数据流：**

```
out_5: (B, 64, F, T)
  → mean(F) → (B, 64, T)
  → MultiStatsPool → (B, 192)
  → Linear(192→64) + LayerNorm + Tanh   ← Tanh 是后来发现有问题的地方
  → FourierKANBlock [64→32→16]，全层统一 Ω=4
  → Linear(16→1) + Sigmoid → (B,)
```

**v1 参数量（`bottleneck_dim=64`）：**

| Head | 参数量 |
|------|--------|
| MLP（hidden=128） | 33,025 |
| KAN v1（Ω=4） | 33,121 |
| KAN v1（Ω=8） | 53,601 |

**v1 实验结果（4 测试集平均）：**

| 实验 | RMSE (ms) | MAE (ms) | Pearson r | R² |
|------|----------:|----------:|----------:|----:|
| MLP MSE | 113.1 | 70.0 | 0.9439 | 0.8894 |
| MLP MAE | **111.5** | **65.5** | **0.9452** | **0.8923** |
| KAN v1 MSE | 116.8 | 71.5 | 0.9397 | 0.8815 |
| KAN v1 MAE | 114.8 | 67.6 | 0.9414 | 0.8857 |

KAN v1 全面弱于 MLP，分析原因：

1. **Tanh 截断了幅度信息**：原始 Fourier-ASR 没有 Tanh，我们加了等于无故损失信息
2. **Ω=4 太小，全层统一**：原版第一层用 1024，我们用 4，表达能力严重不足；且原版对不同层用不同 Ω
3. **参数量对等时 KAN 没有天然优势**：T60 回归的 pooled embedding 未必有 Fourier 结构，需要给 KAN 更多频率分量才能体现优势

---

### v2：对齐 Fourier-ASR 设计（`kan_v2_*.yaml`）

**问题定位后的改动：**

**1. 去掉 proj 层的 Tanh**

```python
# v1
self.proj = nn.Sequential(
    nn.Linear(fc_input, proj_dim),
    nn.LayerNorm(proj_dim),
    nn.Tanh(),           # ← 删掉
)

# v2
self.proj = nn.Sequential(
    nn.Linear(fc_input, proj_dim),
    nn.LayerNorm(proj_dim),
)
```

**2. FourierKANBlock 支持分层 Omega**

对齐 Fourier-ASR 的 `input_grid_size` / `hidden_grid_size` 设计：第一层用大 Ω 捕捉细粒度非线性，后续隐层用小 Ω 控制参数量。

```python
# v1：全层统一 num_frequencies
FourierKANBlock(in_dim, hidden_dims, num_frequencies=4)

# v2：分层设置
FourierKANBlock(
    in_dim, hidden_dims,
    first_num_frequencies=16,    # 第一层，对应 Fourier-ASR input_grid_size
    hidden_num_frequencies=8,    # 后续隐层，对应 Fourier-ASR hidden_grid_size
)
```

**参数名变更（train.py / test.py / YAML）：**

| 旧参数名 | 新参数名 |
|---------|---------|
| `t60_fourier_num_frequencies` | 拆分为 `t60_fourier_first_num_frequencies` 和 `t60_fourier_hidden_num_frequencies` |

**v2 的 T60 head 数据流：**

```
out_5: (B, 64, F, T)
  → mean(F) → (B, 64, T)
  → MultiStatsPool → (B, 192)
  → Linear(192→64) + LayerNorm         ← 去掉了 Tanh
  → FourierKANBlock [64→32→16]
      第一层 Ω=16，后续层 Ω=8          ← 分层 Omega
  → Linear(16→1) + Sigmoid → (B,)
```

**新增配置文件：**

| 文件 | 说明 |
|------|------|
| `configs/t60_single/kan_v2_mse.yaml` | KAN v2 + MSE loss |
| `configs/t60_single/kan_v2_mae.yaml` | KAN v2 + MAE loss |
| `configs/t60_single/kan_v2_huber_delta0.05.yaml` | KAN v2 + Huber loss（delta=0.05） |

旧的 `kan_*.yaml` 保留不动，供历史对比。

**运行命令：**

```bash
./train-and-test.sh -c configs/t60_single/kan_v2_mse.yaml   -g 0,1
./train-and-test.sh -c configs/t60_single/kan_v2_mae.yaml   -g 2,3
./train-and-test.sh -c configs/t60_single/kan_v2_huber_delta0.05.yaml -g 4,5
```

**v2 实验结果（4 测试集平均）：**

| 实验 | RMSE (ms) | MAE (ms) | Pearson r | R² | 平均 bias |
|------|----------:|----------:|----------:|----:|----------:|
| MLP MSE | 113.1 | 70.0 | 0.9439 | 0.8894 | +2.5 ms |
| MLP MAE | 111.5 | **65.5** | 0.9452 | 0.8923 | -5.5 ms |
| KAN v2 MSE | **111.2** | 72.4 | **0.9484** | **0.8931** | **+26.4 ms** ⚠️ |
| KAN v2 MAE | 118.2 | 70.6 | 0.9379 | 0.8790 | -5.8 ms |
| KAN v2 Huber δ=0.05 | 114.7 | 67.5 | 0.9420 | 0.8860 | -0.7 ms |

**v2 关键发现：**

1. **KAN v2 MSE 的 RMSE 最低（111.2ms），但 bias 异常大（+26.4ms）**：MSE 损失在 KAN 的非凸 Fourier 参数空间里容易陷入系统性偏高的局部极值；MLP+MSE 只有 +2.5ms bias，说明这是 KAN 架构特有的问题。

2. **KAN v2 MAE 的 RMSE 最差（118.2ms），且 bias 正常（-5.8ms）**：训练不稳定是根本原因——val loss 在全程剧烈振荡（振幅 ±0.01），最优点出现在 ep14，此后从未突破，早停在 ep29 触发。若 MAE 能稳定收敛，bias 本可与 MLP+MAE 接近。

3. **MAE × KAN 梯度不稳定的根因**：MAE 梯度幅度恒定（±1），不随误差缩小而衰减，无法给 Fourier 权重的多模态 loss landscape 提供"软着陆"；而代码里的梯度裁剪 `max_norm=5.0` 太松，几乎不触发。

4. **Huber δ=0.05 最接近无偏（-0.7ms）**，但在归一化 [0,1] 空间中 δ=0.05 等价于 70ms T60 误差，大多数样本仍落在 MAE 的线性区，本质上仍偏 MAE 特性，训练稳定性改善有限。

---

### v2.1：训练稳定性修正（`kan_v2_mae_clip1.yaml`）

**问题：** KAN v2 + MAE 训练振荡，早停过早，潜力未充分发挥。

**改动（仅 `train.py` 一处 + 新增 YAML）：**

```python
# 之前（硬编码）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 之后（从 args/YAML 读取）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
```

同时在 `parse_args()` 和 YAML 读取中增加了 `clip_grad_norm` 参数，默认值仍为 5.0 以保持向后兼容。

**新配置：** `configs/t60_single/kan_v2_mae_clip1.yaml`

关键字段：`loss: mae` + `clip_grad_norm: 1.0`（从 5.0 收紧到 1.0）。

**运行命令：**

```bash
./train-and-test.sh -c configs/t60_single/kan_v2_mae_clip1.yaml -g 0,1
```

**预期：** 收紧梯度裁剪后，每步 Fourier 权重更新幅度受限，振荡减小，val loss 能持续下降到更低值；若成功，RMSE 和 bias 应同时优于 KAN v2 MSE。

---

## 当前有效的 YAML 配置一览

| 文件 | head | loss | Omega 配置 | 状态 |
|------|------|------|-----------|------|
| `mlp_mse.yaml` | MLP | MSE | — | 已完成 |
| `mlp_mae.yaml` | MLP | MAE | — | 已完成 |
| `mlp_huber.yaml` | MLP | Huber | — | 已完成 |
| `kan_mse.yaml` | KAN v1 | MSE | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_mae.yaml` | KAN v1 | MAE | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_huber.yaml` | KAN v1 | Huber | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_v2_mse.yaml` | KAN v2 | MSE | 第一层 Ω=16，隐层 Ω=8，无 Tanh | 已完成 |
| `kan_v2_mae.yaml` | KAN v2 | MAE | 同上 | 已完成 |
| `kan_v2_huber_delta0.05.yaml` | KAN v2 | Huber δ=0.05 | 同上 | 已完成 |
| `kan_v2_mae_clip1.yaml` | KAN v2 | MAE | 同上 + clip_norm=1.0 | 待运行 |

---

## 不要做的事

- 不要删除 `T60HeadWithPool`（MLP 保留用于对照）
- 不要删除旧的 `kan_*.yaml`（存档，用于历史对比）
- 不要在同一轮实验里同时改 loss、Omega、pooling 方式等多个变量
