"""Fourier-KAN modules for T60 regression head.

This file implements a lightweight Fourier-KAN layer for pooled acoustic
embeddings. It is designed to replace the MLP part of the T60 regression head,
without changing the CMGAN/TSCNet enhancement backbone.

Reference: Fourier-KAN — Kolmogorov-Arnold Networks with Fourier basis functions.
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
        y_j = sum_i phi_{j,i}(x_i) + bias_j

    where:
        phi_{j,i}(x_i) = sum_w [a_{j,i,w} cos(w * x_i)
                                + b_{j,i,w} sin(w * x_i)]

    Parameters
    ----------
    in_features : int
        Input dimensionality.
    out_features : int
        Output dimensionality.
    num_frequencies : int
        Number of Fourier frequency components (Omega). Default 8.
    bias : bool
        Whether to include a per-output bias term. Default True.
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

        # Non-trainable integer frequency indices [1, 2, ..., num_frequencies]
        frequencies = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights.

        Variance is scaled by 1 / (in_features * num_frequencies) so that the
        initial output magnitude is roughly O(1) regardless of input dimension
        and number of frequency components.
        """
        std = math.sqrt(1.0 / (self.in_features * self.num_frequencies))
        nn.init.normal_(self.cos_weight, mean=0.0, std=std)
        nn.init.normal_(self.sin_weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, in_features)

        Returns
        -------
        Tensor, shape (B, out_features)
        """
        if x.dim() != 2:
            raise ValueError(
                f"FourierKANLayer expects 2D input (B, in_features), got {tuple(x.shape)}"
            )
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input dim {self.in_features}, got {x.size(-1)}"
            )

        # angles: (B, I, K)   where I = in_features, K = num_frequencies
        angles = x.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        cos_feat = torch.cos(angles)  # (B, I, K)
        sin_feat = torch.sin(angles)  # (B, I, K)

        # Einsum: sum over in_features and frequencies
        # cos_weight: (O, I, K)  →  y: (B, O)
        y = torch.einsum("bik,oik->bo", cos_feat, self.cos_weight)
        y = y + torch.einsum("bik,oik->bo", sin_feat, self.sin_weight)

        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_frequencies={self.num_frequencies}, "
            f"params={self.out_features * self.in_features * self.num_frequencies * 2 + (self.out_features if self.bias is not None else 0)}"
        )


class FourierKANBlock(nn.Module):
    """Stacked Fourier-KAN regression block.

    对齐 Fourier-ASR 的分层 gridsize 设计：
      - 第一层用较大的 Omega（first_num_frequencies）捕捉 embedding 的细粒度非线性
      - 后续隐层用较小的 Omega（hidden_num_frequencies）控制参数量

    Parameters
    ----------
    in_dim : int
        输入维度（投影后）。
    hidden_dims : tuple of int
        各层输出维度序列，最后一个值即 block 输出维度。
    first_num_frequencies : int
        第一层的 Fourier 频率分量数 Omega。对应 Fourier-ASR 的 input_grid_size。
    hidden_num_frequencies : int
        后续隐层的 Fourier 频率分量数。对应 Fourier-ASR 的 hidden_grid_size。
    dropout : float
        层间 Dropout 概率。
    use_layer_norm : bool
        是否在每层后插入 LayerNorm。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims=(32, 16),
        first_num_frequencies: int = 16,
        hidden_num_frequencies: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []

        for idx in range(len(dims) - 1):
            # 第一层用大 Omega，后续层用小 Omega
            num_freq = first_num_frequencies if idx == 0 else hidden_num_frequencies
            layers.append(
                FourierKANLayer(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    num_frequencies=num_freq,
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
        """
        Parameters
        ----------
        x : Tensor, shape (B, in_dim)

        Returns
        -------
        Tensor, shape (B, hidden_dims[-1])
        """
        return self.net(x)
