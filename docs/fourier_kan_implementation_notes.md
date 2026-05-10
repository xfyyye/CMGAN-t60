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

## Fourier-ASR 原版论文与实现

### 论文信息

**全名**：*Representing Sounds as Neural Amplitude Fields: A Benchmark of Coordinate-MLPs and A Fourier Kolmogorov-Arnold Framework*  
**发表**：AAAI 2025（Proceedings of the AAAI Conference on Artificial Intelligence, Vol.39, No.23, pp.24458-24466）  
**作者**：Linfei Li, Lin Zhang, Zhong Wang, Fengyi Zhang, Zelin Li, Ying Shen  
**arXiv**：[2601.06406](https://arxiv.org/abs/2601.06406)  
**本地仓库**：`/mnt/tidal-sh01/usr/chuan/youling/project/Fourier-ASR/`  
**注意**："ASR" = **Audio Signal Representation**（音频信号表示），不是 Automatic Speech Recognition。

### 原版任务：音频隐式神经表示（INR）

任务：将一段音频波形用一个连续函数表示，输入归一化时间坐标 `t ∈ [-1, 1]`，输出对应振幅值，即学习 `f: t → amplitude`。类比 NeRF 做图像/3D，NeAF（Neural Amplitude Fields）做音频。

原版 `FourierKAN` 结构（默认参数）：

```python
FourierKAN(
    in_features=1,           # 输入：1 维时间坐标
    hidden_features=64,      # 隐层宽度
    hidden_layers=3,         # 隐层数量
    out_features=1,          # 输出：1 维振幅
    input_grid_size=1024,    # 第一层 Omega：极大，捕捉音频全频谱
    hidden_grid_size=5,      # 后续隐层 Omega：小，精炼压缩
    output_grid_size=3,      # 输出层 Omega（若不用 outermost_linear）
    outermost_linear=False,  # False=输出也用 KAN；True=输出改 Linear
    init_type="norm",        # 初始化方式
)
```

网络结构：`KAN(1→64, Ω=1024) → KAN(64→64, Ω=5)×3 → KAN(64→1, Ω=3)`，总参数 ~254K。

输入 `t` 天然在 `[-1, 1]` 内（归一化时间坐标），**不需要任何投影层或 Tanh**，直接送入 FourierKANLayer。

### 原版 FourierKANLayer 实现细节

**权重存储**（`models/fourier_kan.py`）：

```python
# 一个 Parameter 存 cos 和 sin 系数，第 0 维区分 cos/sin
self.fouriercoeffs = torch.nn.Parameter(
    torch.randn(2, outdim, inputdim, gridsize) * std_dev
)
# fouriercoeffs[0]: cos 系数，shape (O, I, K)
# fouriercoeffs[1]: sin 系数，shape (O, I, K)
```

**初始化**（norm 模式）：

```python
std_dev = np.sqrt(1.0 / (inputdim * gridsize))
# 含义：如果输入各维独立同分布且方差为 1，则输出各维方差也约为 1
# 与层宽度 outdim 无关（因为 outdim 个输出相互独立）
```

**前向计算**（reshape + 广播实现，与 einsum 数值等价）：

```python
k = torch.reshape(torch.arange(1, gridsize+1), (1, 1, 1, gridsize))
xrshp = torch.reshape(x, (B, 1, I, 1))       # (B, 1, I, 1)
c = torch.cos(k * xrshp)                      # (B, 1, I, K) 广播 → (B, K_grid, I, K)
s = torch.sin(k * xrshp)                      # 同上
# fouriercoeffs[0:1]: (1, O, I, K)  广播到 (B, O, I, K)
y = torch.sum(c * fouriercoeffs[0:1], (-2,-1))  # (B, O)
y += torch.sum(s * fouriercoeffs[1:2], (-2,-1)) # (B, O)
```

**bias**：`shape = (1, outdim)`（可广播），我们改为 `(outdim,)`，效果相同。

**支持多种初始化**：`norm`（默认）/ `uniform` / `rand`，我们只保留了 `norm`。

**输入维度**：原版内部 `reshape(x, (-1, inputdim))`，支持任意 batch 维度；我们严格要求 2D `(B, in_features)`，对 T60 head 场景无影响。

---

## 与 Fourier-ASR 原版的逐项差异

| 方面 | Fourier-ASR 原版 | 我们的实现 | 影响 |
|------|-----------------|-----------|------|
| **权重存储** | `fouriercoeffs (2, O, I, K)` 一个 Parameter，[0]=cos，[1]=sin | 拆成 `cos_weight (O,I,K)` 和 `sin_weight (O,I,K)` 两个 Parameter | 无影响，参数量完全相同 |
| **bias shape** | `(1, outdim)` | `(outdim,)` | 无影响，广播等价 |
| **初始化** | 支持 norm / uniform / rand；norm 公式 `std = sqrt(1/(I×K))` | 固定 norm，公式相同 | 完全等价 |
| **前向实现** | reshape + 广播，注释说 einsum 可减少内存但更慢 | einsum `"bik,oik->bo"` | 数值完全相同 |
| **输入维度** | 支持任意 `(..., inputdim)`，内部 reshape | 严格要求 `(B, in_features)` 2D | 对 T60 head 无影响 |
| **proj 层** | 无（输入是 1D 时间坐标，天然归一化） | `Linear(192→64) + LayerNorm` | 有意添加，处理高维声学嵌入 |
| **proj 层激活** | 无 | v1 有 Tanh；v2 去掉 Tanh | v2 对齐原版（无 Tanh） |
| **层间 LN / Dropout** | 无（INR 任务过拟合无关紧要） | 有 | 有意添加，防止声学 embedding 过拟合 |
| **分层 Omega** | 第一层 Ω=1024，隐层 Ω=5，输出层 Ω=3 | v1 全层统一 Ω=4；v2 第一层 Ω=16，隐层 Ω=8 | v2 学习分层思路；绝对数值不同（原版输入 1D 需要极大 Ω，我们输入 64D 嵌入不需要） |
| **输出层** | 支持 KAN 或 Linear（`outermost_linear`） | 固定用 `Linear(16→1)` | 相当于原版 `outermost_linear=True` |

### 为什么原版 Omega=1024 而我们只用 16

原版任务输入是 **1 维时间坐标**，直接用 KAN 拟合音频波形（采样率 16kHz，4 秒 = 64000 个时间点）。音频信号频谱丰富（0~8kHz），需要 KAN 在第一层就捕捉极高频的分量，所以 `input_grid_size=1024`。

我们的输入是 **192 维 pooled acoustic embedding**，已经是声学模型中间层提取的高阶特征，不再是原始波形，不需要在 T60 head 里重建完整频谱。64 维投影后 Ω=16 已经能覆盖嵌入空间的主要非线性结构。

### 为什么原版不需要 Tanh

原版输入 `t ∈ [-1, 1]`，`cos(k·t)` 和 `sin(k·t)` 的输出自然有界（`[-1, 1]`）；各层 KAN 的输出由初始化保证方差约为 1，不会爆炸。

我们的投影输出经过 LayerNorm 后方差约为 1、均值为 0，同样有界，v1 额外加 Tanh 是多余的，反而截断了 embedding 中有用的幅度差异（混响时间越长，RIR 尾部能量越大，pooled embedding 的幅度差异携带了 T60 信息）。

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

**v1 参数量（`bottleneck_dim=64`，含各子模块明细）：**

MLP head（对照组，hidden=128）：

| 子模块 | 参数量 |
|--------|--------|
| Linear(192→128) + bias | 24,704 |
| Linear(128→64) + bias | 8,256 |
| Linear(64→1) + bias | 65 |
| **合计** | **33,025** |

KAN v1 head（Ω=4 全层统一）：

| 子模块 | 参数量 |
|--------|--------|
| Linear(192→64) + LayerNorm(64) | 12,480 |
| KAN(64→32, Ω=4) + LayerNorm(32) | 16,480 |
| KAN(32→16, Ω=4) + LayerNorm(16) | 4,144 |
| Linear(16→1) + bias | 17 |
| **合计** | **33,121** |

（若 Ω=8 全层统一：53,601；KAN 层单层参数 = out × in × Ω × 2 + out）

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

**v2 的 T60 head 数据流（对照代码 `T60HeadFourierKAN.forward`）：**

```
out_5: (B, 64, F, T)               ← TSCB 输出，4D 张量
  → mean(dim=2) → (B, 64, T)       ← 频率维平均（T60 与频率无关，对所有 bin 做平均）
  → MultiStatsPool(dim=-1)          ← 时间维统计: concat[avg, max, std] → (B, 64×3) = (B, 192)
  → Linear(192→64) + LayerNorm(64) ← 投影压缩，无 Tanh（v2 改动点）
  → FourierKANBlock:
      KAN(64→32, Ω=16) + LayerNorm(32) + Dropout(0.1)   ← 第一层，大 Ω
      KAN(32→16, Ω=8)  + LayerNorm(16) + Dropout(0.1)   ← 第二层，小 Ω
  → Linear(16→1)                   ← 输出层（普通线性，非 KAN）
  → Sigmoid                        ← 限制输出到 [0,1]，对应归一化 T60
  → squeeze(-1) → (B,)
```

**v2 head 参数量明细（`bottleneck_dim=64`）：**

| 子模块 | 参数量 |
|--------|--------|
| Linear(192→64) + bias | 12,352 |
| LayerNorm(64) | 128 |
| KAN(64→32, Ω=16) + bias | 65,568 |
| LayerNorm(32) | 64 |
| KAN(32→16, Ω=8) + bias | 8,208 |
| LayerNorm(16) | 32 |
| Linear(16→1) + bias | 17 |
| **合计** | **86,369** |

KAN 层参数公式：`out × in × Ω × 2 + out`  
KAN(64→32, Ω=16)：`32 × 64 × 16 × 2 + 32 = 65,568`  
KAN(32→16, Ω=8)：`16 × 32 × 8 × 2 + 16 = 8,208`

**v2 比 v1 参数量大 2.6×**（86,369 vs 33,121），主因是第一层 Ω 从 4 增大到 16（64×32×16×2 vs 64×32×4×2）。

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
| `mlp_mae.yaml` | MLP | MAE | — | 已完成（**当前基准**） |
| `mlp_huber.yaml` | MLP | Huber | — | 已完成 |
| `mlp_mae_log.yaml` | MLP | MAE | — | 待运行（log-scale 对照组） |
| `kan_mse.yaml` | KAN v1 | MSE | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_mae.yaml` | KAN v1 | MAE | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_huber.yaml` | KAN v1 | Huber | 全层 Ω=4 + Tanh | 已完成，存档 |
| `kan_v2_mse.yaml` | KAN v2 | MSE | 第一层 Ω=16，隐层 Ω=8，无 Tanh | 已完成 |
| `kan_v2_mae.yaml` | KAN v2 | MAE | 同上 | 已完成 |
| `kan_v2_huber_delta0.05.yaml` | KAN v2 | Huber δ=0.05 | 同上 | 已完成 |
| `kan_v2_mse_clip1.yaml` | KAN v2 | MSE | 同上 + clip_norm=1.0 | 已完成 |
| `kan_v2_mae_clip1.yaml` | KAN v2 | MAE | 同上 + clip_norm=1.0 | 已完成 |
| `kan_v2_mae_clip1_log.yaml` | KAN v2 | MAE | 同上 + log-scale 归一化 | **训练中** |

---

## 不要做的事

- 不要删除 `T60HeadWithPool`（MLP 保留用于对照）
- 不要删除旧的 `kan_*.yaml`（存档，用于历史对比）
- 不要在同一轮实验里同时改 loss、Omega、pooling 方式等多个变量
