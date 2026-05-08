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
| `configs/t60_single/kan_v2_huber.yaml` | KAN v2 + Huber loss |

旧的 `kan_*.yaml` 保留不动，供历史对比。

**运行命令：**

```bash
mkdir -p logs
nohup bash train.sh -c configs/t60_single/kan_v2_mse.yaml   -g 0 > logs/kan_v2_mse.log   2>&1 &
nohup bash train.sh -c configs/t60_single/kan_v2_mae.yaml   -g 1 > logs/kan_v2_mae.log   2>&1 &
nohup bash train.sh -c configs/t60_single/kan_v2_huber.yaml -g 2 > logs/kan_v2_huber.log 2>&1 &
```

---

## 当前有效的 YAML 配置一览

| 文件 | head | loss | Omega 配置 | 备注 |
|------|------|------|-----------|------|
| `mlp_mse.yaml` | MLP | MSE | — | baseline |
| `mlp_mae.yaml` | MLP | MAE | — | baseline |
| `mlp_huber.yaml` | MLP | Huber | — | baseline |
| `kan_mse.yaml` | KAN v1 | MSE | 全层 Ω=4 + Tanh | 已跑，存档对比 |
| `kan_mae.yaml` | KAN v1 | MAE | 全层 Ω=4 + Tanh | 已跑，存档对比 |
| `kan_huber.yaml` | KAN v1 | Huber | 全层 Ω=4 + Tanh | 已跑，存档对比 |
| `kan_v2_mse.yaml` | KAN v2 | MSE | 第一层 Ω=16，隐层 Ω=8，无 Tanh | 当前实验 |
| `kan_v2_mae.yaml` | KAN v2 | MAE | 同上 | 当前实验 |
| `kan_v2_huber.yaml` | KAN v2 | Huber | 同上 | 当前实验 |

---

## 不要做的事

- 不要删除 `T60HeadWithPool`（MLP 保留用于对照）
- 不要删除旧的 `kan_*.yaml`（存档，用于历史对比）
- 不要在同一轮实验里同时改 loss、Omega、pooling 方式等多个变量
