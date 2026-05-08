# T60 估计头从 MLP 替换为 Fourier-KAN 的代码修改方案

本文档用于指导 coding agent 在 `xfyyye/CMGAN-t60` 仓库中进行最小侵入式修改。目标不是改动 CMGAN 主干，也不是改变语音增强输出路径，而是在当前 T60 估计头基础上新增一个可切换的 Fourier-KAN 回归头，用于比较其是否能提升 T60 估计能力。

---

## 1. 修改目标

当前仓库中的 T60 估计路径大致是：

```text
out_5: CMGAN / TSCNet bottleneck feature
→ if 4D: mean over frequency dimension F
→ MultiStatsPool over time dimension T: mean + max + std
→ MLP regression head
→ Sigmoid
→ normalized t60_pred: (B,)
```

本次修改目标是新增：

```text
out_5
→ frequency compression, initially keep existing mean(F)
→ MultiStatsPool
→ Projection + LayerNorm + bounded mapping
→ Fourier-KAN regression block
→ Linear output layer
→ Sigmoid / Softplus
→ t60_pred: (B,)
```

核心要求：

1. 保留当前 MLP T60 head，不能删除。
2. 新增 Fourier-KAN T60 head，并通过参数选择。
3. 默认行为不破坏现有训练脚本，默认仍可使用 MLP。
4. `forward()` 返回格式保持不变：`final_real, final_imag, t60_pred`。
5. 第一阶段只替换 T60 head，不改 mask decoder、complex decoder、loss、dataset。

---

## 2. 当前需要重点修改的文件

优先修改：

```text
models/generator_t60.py
```

建议新增：

```text
models/fourier_kan.py
```

如训练脚本中有模型构造参数，需要进一步修改对应 train/config 文件，例如：

```text
train*.py
config*.py
options*.py
```

coding agent 需要先在仓库中搜索：

```bash
grep -R "TSCNet_MultiTask" -n .
grep -R "T60HeadWithPool" -n .
grep -R "generator_t60" -n .
```

根据搜索结果决定是否需要在训练入口暴露 `t60_head_type` 等参数。

---

## 3. 设计原则

### 3.1 不要直接把 ReLU 改成 sin

Fourier-KAN 不是简单地把 MLP 中的激活函数从 `ReLU` 换成 `sin`。它的核心区别是：

```text
MLP layer:
    y = activation(Wx + b)

Fourier-KAN layer:
    y_j = Σ_i φ_{j,i}(x_i)
```

其中每条边上的 `φ_{j,i}` 不是单个标量权重，而是一组可学习 Fourier 基函数：

```text
φ_{j,i}(x_i) = Σ_{ω=1}^{Ω} [a_{j,i,ω} cos(ω x_i) + b_{j,i,ω} sin(ω x_i)] + c_{j,i}
```

因此代码里应该新增一个真正的 `FourierKANLayer`，而不是写一个 `Linear + Sin`。

### 3.2 T60 head 的输入不是原始波形，而是深层声学 embedding

当前 T60 head 的输入来自 `out_5`，它已经是 CMGAN / TSCNet 编码后的瓶颈特征，不是原始 waveform，也不是原始 STFT。因此 Fourier-KAN 前面建议增加：

```text
Linear projection + LayerNorm + Tanh
```

理由：

1. Fourier basis 对输入尺度敏感。
2. pooled acoustic embedding 的数值范围不一定稳定。
3. `LayerNorm + Tanh` 可以把输入约束到更适合 Fourier 展开的有界区间。
4. Projection 可以降低计算量，避免 Fourier-KAN 参数量过大。

---

## 4. 新增文件：`models/fourier_kan.py`

新增一个独立文件，避免把 Fourier-KAN 逻辑堆进 `generator_t60.py`。

### 4.1 文件内容建议

```python
"""Fourier-KAN modules for T60 regression head.

This file implements a lightweight Fourier-KAN layer for pooled acoustic
embeddings. It is designed to replace the MLP part of the T60 regression head,
without changing the CMGAN/TSCNet enhancement backbone.
"""

import math
import torch
import torch.nn as nn


class FourierKANLayer(nn.Module):
    """A lightweight Fourier-KAN layer.

    Input:
        x: Tensor of shape (B, in_features)

    Output:
        y: Tensor of shape (B, out_features)

    For each output dimension j:
        y_j = sum_i phi_{j,i}(x_i)

    where:
        phi_{j,i}(x_i) = sum_w [a_{j,i,w} cos(w x_i)
                                + b_{j,i,w} sin(w x_i)] + c_j
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # Shape: (out_features, in_features, num_frequencies)
        self.cos_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_frequencies)
        )
        self.sin_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_frequencies)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        frequencies = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Following the intuition from Fourier-KAN initialization:
        # coefficient variance should shrink with both input dimension and frequency count.
        std = math.sqrt(1.0 / (self.in_features * self.num_frequencies))
        nn.init.normal_(self.cos_weight, mean=0.0, std=std)
        nn.init.normal_(self.sin_weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"FourierKANLayer expects 2D input (B, in_features), got {tuple(x.shape)}"
            )
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input dim {self.in_features}, got {x.size(-1)}"
            )

        # x: (B, I)
        # angles: (B, I, K)
        angles = x.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        cos_feat = torch.cos(angles)
        sin_feat = torch.sin(angles)

        # cos_feat: (B, I, K)
        # cos_weight: (O, I, K)
        # output: (B, O)
        y = torch.einsum("bik,oik->bo", cos_feat, self.cos_weight)
        y = y + torch.einsum("bik,oik->bo", sin_feat, self.sin_weight)

        if self.bias is not None:
            y = y + self.bias
        return y


class FourierKANBlock(nn.Module):
    """Stacked Fourier-KAN regression block.

    This block intentionally uses LayerNorm and Dropout between Fourier-KAN layers
    for stability on pooled acoustic embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims=(32, 16),
        num_frequencies: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []

        for idx in range(len(dims) - 1):
            layers.append(
                FourierKANLayer(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    num_frequencies=num_frequencies,
                    bias=True,
                )
            )
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[idx + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### 4.2 参数量提醒

单层 Fourier-KAN 的参数量大致为：

```text
out_features × in_features × num_frequencies × 2 + out_features
```

因此不要一开始把 `num_frequencies` 设太大。建议第一轮实验：

```text
proj_dim = 64
hidden_dims = (32, 16)
num_frequencies = 4 或 8
```

---

## 5. 修改 `models/generator_t60.py`

### 5.1 保留原始 `MultiStatsPool`

当前 `MultiStatsPool` 可以保留：

```python
class MultiStatsPool(nn.Module):
    """聚合 avg + max + std 三种统计量"""
    def forward(self, x, dim=-1):
        avg = x.mean(dim=dim)
        max_val = x.amax(dim=dim)
        std = x.std(dim=dim)
        return torch.cat([avg, max_val, std], dim=-1)
```

后续可以再扩展 `FrequencyAwarePool`，但第一阶段不要引入太多变量。

### 5.2 新增 `T60HeadFourierKAN`

在 `generator_t60.py` 顶部新增 import：

```python
from models.fourier_kan import FourierKANBlock
```

然后在 `T60HeadWithPool` 后新增：

```python
class T60HeadFourierKAN(nn.Module):
    """Fourier-KAN T60 回归头。

    输入:
        x: (B, C, F, T) or (B, C, T)

    流程:
        if 4D: mean over F
        MultiStatsPool over T
        Projection + LayerNorm + Tanh
        FourierKANBlock
        Linear output
        Sigmoid / Softplus
    """

    def __init__(
        self,
        bottleneck_dim: int,
        proj_dim: int = 64,
        hidden_dims=(32, 16),
        num_frequencies: int = 8,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
    ):
        super().__init__()
        self.multi_pool = MultiStatsPool()
        fc_input = bottleneck_dim * 3

        self.proj = nn.Sequential(
            nn.Linear(fc_input, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Tanh(),
        )

        self.kan = FourierKANBlock(
            in_dim=proj_dim,
            hidden_dims=hidden_dims,
            num_frequencies=num_frequencies,
            dropout=dropout,
            use_layer_norm=True,
        )

        self.out = nn.Linear(self.kan.out_dim, 1)

        if out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_activation == "softplus":
            self.out_act = nn.Softplus()
        elif out_activation in (None, "none", "identity"):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported out_activation: {out_activation}")

    def forward(self, x):
        # x: (B, C, F, T) or (B, C, T)
        if x.dim() == 4:
            # 第一阶段保持当前行为：频率维做平均。
            # 后续 frequency-aware pooling 可以在这里替换。
            x = x.mean(dim=2)  # (B, C, T)
        elif x.dim() != 3:
            raise ValueError(f"Expected 3D or 4D input, got {tuple(x.shape)}")

        x = self.multi_pool(x, dim=-1)  # (B, C*3)
        x = self.proj(x)                # (B, proj_dim)
        x = self.kan(x)                 # (B, hidden_dims[-1])
        x = self.out(x)                 # (B, 1)
        x = self.out_act(x)
        return x.squeeze(-1)            # (B,)
```

---

## 6. 修改模型构造逻辑

当前 `TSCNet_MultiTask` 中类似：

```python
self.t60_head = T60HeadWithPool(bottleneck_dim=num_channel, hidden_dim=128)
```

需要改成可配置，但默认保持原始 MLP：

```python
def build_t60_head(
    head_type: str,
    num_channel: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    fourier_proj_dim: int = 64,
    fourier_hidden_dims=(32, 16),
    fourier_num_frequencies: int = 8,
    fourier_dropout: float = 0.1,
    out_activation: str = "sigmoid",
):
    if head_type == "mlp":
        return T60HeadWithPool(
            bottleneck_dim=num_channel,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif head_type == "fourier_kan":
        return T60HeadFourierKAN(
            bottleneck_dim=num_channel,
            proj_dim=fourier_proj_dim,
            hidden_dims=fourier_hidden_dims,
            num_frequencies=fourier_num_frequencies,
            dropout=fourier_dropout,
            out_activation=out_activation,
        )
    else:
        raise ValueError(f"Unsupported t60 head type: {head_type}")
```

然后修改 `TSCNet_MultiTask.__init__`：

```python
class TSCNet_MultiTask(nn.Module):
    def __init__(
        self,
        num_channel=64,
        num_features=201,
        t60_head_type="mlp",
        t60_hidden_dim=128,
        t60_dropout=0.3,
        t60_fourier_proj_dim=64,
        t60_fourier_hidden_dims=(32, 16),
        t60_fourier_num_frequencies=8,
        t60_fourier_dropout=0.1,
        t60_out_activation="sigmoid",
    ):
        super().__init__()
        ...
        self.t60_head = build_t60_head(
            head_type=t60_head_type,
            num_channel=num_channel,
            hidden_dim=t60_hidden_dim,
            dropout=t60_dropout,
            fourier_proj_dim=t60_fourier_proj_dim,
            fourier_hidden_dims=t60_fourier_hidden_dims,
            fourier_num_frequencies=t60_fourier_num_frequencies,
            fourier_dropout=t60_fourier_dropout,
            out_activation=t60_out_activation,
        )
```

同样修改 `TSCNet_MultiTask_2TSCB.__init__`，保持参数名一致。

---

## 7. 训练脚本参数建议

coding agent 需要搜索当前训练脚本中模型初始化位置。如果训练脚本支持 argparse，建议新增：

```python
parser.add_argument(
    "--t60-head-type",
    type=str,
    default="mlp",
    choices=["mlp", "fourier_kan"],
)
parser.add_argument("--t60-fourier-proj-dim", type=int, default=64)
parser.add_argument("--t60-fourier-num-frequencies", type=int, default=8)
parser.add_argument("--t60-fourier-dropout", type=float, default=0.1)
parser.add_argument(
    "--t60-out-activation",
    type=str,
    default="sigmoid",
    choices=["sigmoid", "softplus", "identity", "none"],
)
```

如果训练代码没有 argparse，而是直接写死模型初始化，则第一阶段可以手动改：

```python
model = TSCNet_MultiTask(
    num_channel=64,
    num_features=201,
    t60_head_type="fourier_kan",
    t60_fourier_proj_dim=64,
    t60_fourier_num_frequencies=8,
)
```

---

## 8. 推荐实验配置

### 8.1 Baseline

```text
t60_head_type = mlp
hidden_dim = 128
dropout = 0.3
```

### 8.2 Fourier-KAN 第一版

```text
t60_head_type = fourier_kan
t60_fourier_proj_dim = 64
t60_fourier_hidden_dims = (32, 16)
t60_fourier_num_frequencies = 4
t60_fourier_dropout = 0.1
t60_out_activation = sigmoid
```

### 8.3 Fourier-KAN 稍强版

```text
t60_head_type = fourier_kan
t60_fourier_proj_dim = 64
t60_fourier_hidden_dims = (32, 16)
t60_fourier_num_frequencies = 8
t60_fourier_dropout = 0.1
t60_out_activation = sigmoid
```

### 8.4 不建议第一轮使用

```text
num_frequencies >= 16
proj_dim >= 128
hidden_dims >= (128, 64)
```

原因：参数量与计算量上升明显，容易把实验变成“参数量增加导致的提升”。

---

## 9. 必须添加的快速测试

新增一个最小测试文件，例如：

```text
tests/test_t60_fourier_kan_head.py
```

如果仓库没有 tests 目录，可以先写一个临时脚本：

```text
scripts/smoke_test_t60_fourier_kan.py
```

测试内容：

```python
import torch

from models.generator_t60 import T60HeadWithPool, T60HeadFourierKAN
from models.fourier_kan import FourierKANLayer


def test_fourier_kan_layer_shape():
    layer = FourierKANLayer(in_features=64, out_features=32, num_frequencies=8)
    x = torch.randn(4, 64)
    y = layer(x)
    assert y.shape == (4, 32)
    assert torch.isfinite(y).all()


def test_t60_fourier_head_4d_shape():
    head = T60HeadFourierKAN(
        bottleneck_dim=64,
        proj_dim=64,
        hidden_dims=(32, 16),
        num_frequencies=8,
    )
    x = torch.randn(4, 64, 50, 100)
    y = head(x)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_mlp_head_unchanged_shape():
    head = T60HeadWithPool(bottleneck_dim=64, hidden_dim=128)
    x = torch.randn(4, 64, 50, 100)
    y = head(x)
    assert y.shape == (4,)
    assert torch.isfinite(y).all()
```

也可以直接用命令做 smoke test：

```bash
python - <<'PY'
import torch
from models.generator_t60 import T60HeadWithPool, T60HeadFourierKAN

for Head in [T60HeadWithPool, T60HeadFourierKAN]:
    head = Head(bottleneck_dim=64)
    x = torch.randn(2, 64, 50, 100)
    y = head(x)
    print(Head.__name__, y.shape, y.min().item(), y.max().item())
PY
```

---

## 10. 运行检查

完成修改后，至少执行：

```bash
python -m py_compile models/fourier_kan.py models/generator_t60.py
```

如果有测试框架：

```bash
pytest tests/test_t60_fourier_kan_head.py -q
```

如果无测试框架：

```bash
python scripts/smoke_test_t60_fourier_kan.py
```

然后启动一个极小训练步数验证：

```bash
# 示例，具体命令按仓库实际训练脚本调整
python train.py --t60-head-type fourier_kan --t60-fourier-num-frequencies 4
```

确认：

1. 模型可以正常 forward。
2. `t60_pred` shape 为 `(B,)`。
3. loss 不出现 NaN。
4. 显存没有明显爆炸。
5. 保存 checkpoint 不报错。

---

## 11. 参数量统计建议

为避免实验结论被质疑，需要统计 MLP head 和 Fourier-KAN head 的参数量。

可以新增工具函数：

```python
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
```

训练开始时打印：

```python
print(f"T60 head type: {t60_head_type}")
print(f"T60 head params: {count_parameters(model.t60_head):,}")
```

核心比较不应该只看最终指标，还要记录：

```text
T60 head type
T60 head params
Total model params
T60 MAE
T60 RMSE
T60 Pearson correlation
Enhancement metrics, if available
```

---

## 12. 评价指标建议

T60 估计能力优先看：

```text
MAE
RMSE
Pearson correlation
Spearman correlation, optional
分桶误差：short / medium / long reverb
```

如果 T60 label 是归一化值，建议同时记录：

```text
normalized MAE / RMSE
real-scale MAE / RMSE, after inverse normalization
```

不要只看总 loss，因为多任务 loss 中可能混有语音增强 loss。

---

## 13. 可选第二阶段：Frequency-aware pooling

第一阶段先保持当前：

```python
x = x.mean(dim=2)
```

等 Fourier-KAN head 能稳定训练后，再考虑替换为频带统计。原因是 T60 具有频带依赖性，直接对 `F` 做全局平均可能损失频率相关混响信息。

第二阶段可以新增：

```text
FrequencyBandPool:
    input:  (B, C, F, T)
    split F into K bands
    pool each band over frequency
    output: (B, C*K, T)
```

示意代码：

```python
class FrequencyBandPool(nn.Module):
    def __init__(self, num_bands=4):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x):
        # x: (B, C, F, T)
        bands = torch.chunk(x, chunks=self.num_bands, dim=2)
        band_feats = [b.mean(dim=2) for b in bands]  # each: (B, C, T)
        return torch.cat(band_feats, dim=1)          # (B, C*K, T)
```

如果启用这个模块，`T60HeadFourierKAN` 中的 `fc_input` 不能再写死为 `bottleneck_dim * 3`，而应该是：

```text
bottleneck_dim * num_bands * 3
```

第二阶段再做，不要和第一阶段混在一起，否则难以判断提升来自 Fourier-KAN 还是 frequency-aware pooling。

---

## 14. 最终期望代码结构

```text
models/
  generator_t60.py
    MultiStatsPool
    T60HeadWithPool                # 原始 MLP head，保留
    T60HeadFourierKAN              # 新增 Fourier-KAN head
    build_t60_head                 # 新增 head factory
    TSCNet_MultiTask               # 增加 t60_head_type 参数
    TSCNet_MultiTask_2TSCB         # 增加 t60_head_type 参数

  fourier_kan.py                   # 新增
    FourierKANLayer
    FourierKANBlock
```

---

## 15. 不要做的事情

1. 不要删除原始 `T60HeadWithPool`。
2. 不要改变 `TSCNet_MultiTask.forward()` 的返回值数量和顺序。
3. 不要在第一阶段改动去噪 decoder。
4. 不要在第一阶段同时引入 B-spline KAN、Fourier-KAN、frequency-aware pooling、loss 修改等多个变量。
5. 不要默认把所有训练都切到 Fourier-KAN；应通过参数控制。
6. 不要让 Fourier-KAN 输入未经归一化的高幅值 embedding，建议至少使用 `LayerNorm + Tanh`。

---

## 16. 推荐提交顺序

建议拆成 3 个 commit：

### Commit 1: add Fourier-KAN modules

```text
add models/fourier_kan.py
add FourierKANLayer and FourierKANBlock
add smoke shape test
```

### Commit 2: add Fourier-KAN T60 head

```text
modify models/generator_t60.py
add T60HeadFourierKAN
add build_t60_head
make TSCNet_MultiTask configurable
make TSCNet_MultiTask_2TSCB configurable
keep default t60_head_type="mlp"
```

### Commit 3: expose training config

```text
add --t60-head-type
add --t60-fourier-proj-dim
add --t60-fourier-num-frequencies
add params logging
```

---

## 17. 第一轮实验建议

按以下顺序跑：

```text
Exp 1: MLP Head baseline
Exp 2: Projection + Fourier-KAN, Ω=4
Exp 3: Projection + Fourier-KAN, Ω=8
Exp 4: Projection + Fourier-KAN, Ω=12, only if Exp 2/3 stable
```

如果 `Ω=8` 比 `Ω=4` 没有提升，不要继续增大 Ω，优先检查：

```text
input feature 是否过度平均 F
T60 label 是否归一化正确
T60 loss 权重是否太小
T60 分桶误差是否只在长混响场景变好
```

---

## 18. 验收标准

修改完成后，coding agent 应输出：

1. 修改了哪些文件。
2. 新增了哪些类。
3. MLP head 是否仍能正常使用。
4. Fourier-KAN head 的 dummy forward 是否通过。
5. `TSCNet_MultiTask(..., t60_head_type="fourier_kan")` 是否能实例化。
6. `t60_pred` shape 是否仍为 `(B,)`。
7. Fourier-KAN head 的参数量。
8. 与原始 MLP head 的参数量对比。

