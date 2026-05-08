"""
CMGAN backbone for single-task T60 estimation.

This branch keeps the DenseEncoder + TSCB acoustic backbone and removes the
enhancement decoders entirely. The model only predicts normalized T60.

T60 head is configurable via `t60_head_type`:
  - "mlp"         : original MLP head (T60HeadWithPool) — default, unchanged
  - "fourier_kan" : Fourier-KAN head (T60HeadFourierKAN)
"""

import torch
import torch.nn as nn
from models.generator import DenseEncoder, TSCB
from models.fourier_kan import FourierKANBlock


# ─────────────────────────────────────────────────────────────────
# Shared pooling utility
# ─────────────────────────────────────────────────────────────────

class MultiStatsPool(nn.Module):
    """聚合 avg + max + std 三种统计量"""

    def forward(self, x, dim=-1):
        avg = x.mean(dim=dim)
        max_val = x.amax(dim=dim)
        std = x.std(dim=dim)
        return torch.cat([avg, max_val, std], dim=-1)


# ─────────────────────────────────────────────────────────────────
# T60 heads
# ─────────────────────────────────────────────────────────────────

class T60HeadWithPool(nn.Module):
    """T60 回归头: 从瓶颈层特征预测归一化T60 (原始 MLP 版本，保留不变)

    自动处理 4D/3D 输入:
      (B, C, F, T) → mean(F) → MultiStatsPool(T) → FC → Sigmoid → (B,)
    """

    def __init__(self, bottleneck_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.multi_pool = MultiStatsPool()
        fc_input = bottleneck_dim * 3  # avg + max + std
        self.fc = nn.Sequential(
            nn.Linear(fc_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, F, T)
        if x.dim() == 4:
            x = x.mean(dim=2)  # (B, C, T) 频率维度平均
        # MultiStatsPool on time dim
        x = self.multi_pool(x, dim=-1)  # (B, C*3)
        return self.fc(x).squeeze(-1)  # (B,)


class T60HeadFourierKAN(nn.Module):
    """Fourier-KAN T60 回归头。

    输入:
        x: (B, C, F, T) 或 (B, C, T)

    流程:
        if 4D: mean over F         → (B, C, T)
        MultiStatsPool over T      → (B, C*3)
        Projection + LayerNorm + Tanh  → (B, proj_dim)
        FourierKANBlock            → (B, hidden_dims[-1])
        Linear output              → (B, 1)
        Sigmoid / Softplus / Identity  → (B,)

    说明:
        - 投影层只用 Linear + LayerNorm，不加 Tanh。
          LayerNorm 已足够控制幅度，Tanh 会截断有用的幅度信息。
        - 对齐 Fourier-ASR 分层 gridsize 设计：
          第一层用较大的 first_num_frequencies，后续隐层用较小的 hidden_num_frequencies。

    Parameters
    ----------
    bottleneck_dim : int
        TSCB 输出的通道数（默认 64）。
    proj_dim : int
        投影维度（Fourier-KAN 的输入维度）。
    hidden_dims : tuple of int
        Fourier-KAN 内部隐层维度序列。
    first_num_frequencies : int
        第一层 KAN 的 Omega，建议用较大值（如 16）。
    hidden_num_frequencies : int
        后续隐层 KAN 的 Omega，建议用较小值（如 8）。
    dropout : float
        KAN 层间 Dropout 概率。
    out_activation : str
        输出激活函数，支持 "sigmoid" / "softplus" / "identity" / "none"。
    """

    def __init__(
        self,
        bottleneck_dim: int,
        proj_dim: int = 64,
        hidden_dims=(32, 16),
        first_num_frequencies: int = 16,
        hidden_num_frequencies: int = 8,
        dropout: float = 0.1,
        out_activation: str = "sigmoid",
    ):
        super().__init__()
        self.multi_pool = MultiStatsPool()
        fc_input = bottleneck_dim * 3  # avg + max + std

        # 去掉 Tanh：LayerNorm 已经控制幅度，Tanh 会截断幅度信息
        self.proj = nn.Sequential(
            nn.Linear(fc_input, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.kan = FourierKANBlock(
            in_dim=proj_dim,
            hidden_dims=hidden_dims,
            first_num_frequencies=first_num_frequencies,
            hidden_num_frequencies=hidden_num_frequencies,
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
            raise ValueError(f"Unsupported out_activation: {out_activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) or (B, C, T)
        if x.dim() == 4:
            x = x.mean(dim=2)  # (B, C, T)
        elif x.dim() != 3:
            raise ValueError(f"Expected 3D or 4D input, got {tuple(x.shape)}")

        x = self.multi_pool(x, dim=-1)  # (B, C*3)
        x = self.proj(x)                # (B, proj_dim)
        x = self.kan(x)                 # (B, hidden_dims[-1])
        x = self.out(x)                 # (B, 1)
        x = self.out_act(x)
        return x.squeeze(-1)            # (B,)


# ─────────────────────────────────────────────────────────────────
# Head factory
# ─────────────────────────────────────────────────────────────────

def build_t60_head(
    head_type: str,
    num_channel: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    fourier_proj_dim: int = 64,
    fourier_hidden_dims=(32, 16),
    fourier_first_num_frequencies: int = 16,
    fourier_hidden_num_frequencies: int = 8,
    fourier_dropout: float = 0.1,
    out_activation: str = "sigmoid",
) -> nn.Module:
    """T60 head 工厂函数。

    Parameters
    ----------
    head_type : {"mlp", "fourier_kan"}
        选择使用的 T60 head 实现。
    num_channel : int
        TSCB 输出通道数（bottleneck_dim）。
    hidden_dim : int
        MLP head 的隐层宽度（仅 head_type="mlp" 使用）。
    dropout : float
        MLP head 的 Dropout 概率。
    fourier_proj_dim : int
        Fourier-KAN head 的投影维度。
    fourier_hidden_dims : tuple of int
        Fourier-KAN head 的内部隐层维度序列。
    fourier_first_num_frequencies : int
        第一层 KAN 的 Omega（较大值）。
    fourier_hidden_num_frequencies : int
        后续隐层 KAN 的 Omega（较小值）。
    fourier_dropout : float
        Fourier-KAN head 的 Dropout 概率。
    out_activation : str
        Fourier-KAN head 输出激活函数。

    Returns
    -------
    nn.Module
    """
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
            first_num_frequencies=fourier_first_num_frequencies,
            hidden_num_frequencies=fourier_hidden_num_frequencies,
            dropout=fourier_dropout,
            out_activation=out_activation,
        )
    else:
        raise ValueError(f"Unsupported t60_head_type: {head_type!r}. Choose 'mlp' or 'fourier_kan'.")


# ─────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────

class TSCNet_T60Estimator(nn.Module):
    """CMGAN encoder/TSCB backbone for single-task T60 estimation (4 TSCB blocks).

    Parameters
    ----------
    num_channel : int
        Channel width throughout the backbone.
    num_features : int
        Kept for CLI/checkpoint compatibility; not used internally.
    t60_head_type : str
        "mlp" (default) or "fourier_kan".
    t60_hidden_dim : int
        MLP head hidden width (only when t60_head_type="mlp").
    t60_dropout : float
        MLP head dropout (only when t60_head_type="mlp").
    t60_fourier_proj_dim : int
        Fourier-KAN projection dim (only when t60_head_type="fourier_kan").
    t60_fourier_hidden_dims : tuple of int
        Fourier-KAN hidden dims (only when t60_head_type="fourier_kan").
    t60_fourier_first_num_frequencies : int
        第一层 KAN 的 Omega（较大值，仅 fourier_kan 有效）。
    t60_fourier_hidden_num_frequencies : int
        后续隐层 KAN 的 Omega（较小值，仅 fourier_kan 有效）。
    t60_fourier_dropout : float
        Fourier-KAN dropout (only when t60_head_type="fourier_kan").
    t60_out_activation : str
        Output activation for Fourier-KAN head.
    """

    def __init__(
        self,
        num_channel=64,
        num_features=201,
        t60_head_type="mlp",
        t60_hidden_dim=128,
        t60_dropout=0.3,
        t60_fourier_proj_dim=64,
        t60_fourier_hidden_dims=(32, 16),
        t60_fourier_first_num_frequencies=16,
        t60_fourier_hidden_num_frequencies=8,
        t60_fourier_dropout=0.1,
        t60_out_activation="sigmoid",
    ):
        super().__init__()
        del num_features  # Kept for CLI/checkpoint compatibility with old constructors.
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.t60_head = build_t60_head(
            head_type=t60_head_type,
            num_channel=num_channel,
            hidden_dim=t60_hidden_dim,
            dropout=t60_dropout,
            fourier_proj_dim=t60_fourier_proj_dim,
            fourier_hidden_dims=t60_fourier_hidden_dims,
            fourier_first_num_frequencies=t60_fourier_first_num_frequencies,
            fourier_hidden_num_frequencies=t60_fourier_hidden_num_frequencies,
            fourier_dropout=t60_fourier_dropout,
            out_activation=t60_out_activation,
        )

    def forward(self, x):
        """
        参数:
            x: (B, 2, T, F) 复数STFT [real, imag]，已 permute
        返回:
            t60_pred:   (B,) 归一化T60预测 [0, 1]
        """
        # 提取幅度，与原始复数谱一起作为 3 通道输入。
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        # 编码器 + Conformer
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        # T60 估计
        return self.t60_head(out_5)


class TSCNet_T60Estimator_2TSCB(nn.Module):
    """CMGAN single-task T60 estimator with 2 TSCB blocks.

    与 TSCNet_T60Estimator 的区别:
      - 只用 2 层 TSCB（砍掉 TSCB_3 和 TSCB_4）
      - 计算量减半，训练速度约 2-3x
      - 参数: ~1.87M → ~1.35M

    其余参数含义与 TSCNet_T60Estimator 相同。
    """

    def __init__(
        self,
        num_channel=64,
        num_features=201,
        t60_head_type="mlp",
        t60_hidden_dim=128,
        t60_dropout=0.3,
        t60_fourier_proj_dim=64,
        t60_fourier_hidden_dims=(32, 16),
        t60_fourier_first_num_frequencies=16,
        t60_fourier_hidden_num_frequencies=8,
        t60_fourier_dropout=0.1,
        t60_out_activation="sigmoid",
    ):
        super().__init__()
        del num_features  # Kept for CLI/checkpoint compatibility with old constructors.
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)

        self.t60_head = build_t60_head(
            head_type=t60_head_type,
            num_channel=num_channel,
            hidden_dim=t60_hidden_dim,
            dropout=t60_dropout,
            fourier_proj_dim=t60_fourier_proj_dim,
            fourier_hidden_dims=t60_fourier_hidden_dims,
            fourier_first_num_frequencies=t60_fourier_first_num_frequencies,
            fourier_hidden_num_frequencies=t60_fourier_hidden_num_frequencies,
            fourier_dropout=t60_fourier_dropout,
            out_activation=t60_out_activation,
        )

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)

        return self.t60_head(out_3)
