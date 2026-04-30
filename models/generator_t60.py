"""
CMGAN backbone for single-task T60 estimation.

This branch keeps the DenseEncoder + TSCB acoustic backbone and removes the
enhancement decoders entirely. The model only predicts normalized T60.
"""

import torch
import torch.nn as nn
from models.generator import DenseEncoder, TSCB


class MultiStatsPool(nn.Module):
    """聚合 avg + max + std 三种统计量"""

    def forward(self, x, dim=-1):
        avg = x.mean(dim=dim)
        max_val = x.amax(dim=dim)
        std = x.std(dim=dim)
        return torch.cat([avg, max_val, std], dim=-1)


class T60HeadWithPool(nn.Module):
    """T60 回归头: 从瓶颈层特征预测归一化T60

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


class TSCNet_T60Estimator(nn.Module):
    """CMGAN encoder/TSCB backbone for single-task T60 estimation."""

    def __init__(self, num_channel=64, num_features=201):
        super().__init__()
        del num_features  # Kept for CLI/checkpoint compatibility with old constructors.
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.t60_head = T60HeadWithPool(bottleneck_dim=num_channel, hidden_dim=128)

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

        # T60 估计 (新增)
        return self.t60_head(out_5)


class TSCNet_T60Estimator_2TSCB(nn.Module):
    """CMGAN single-task T60 estimator with 2 TSCB blocks.

    与 TSCNet_T60Estimator 的区别:
      - 只用 2 层 TSCB（砍掉 TSCB_3 和 TSCB_4）
      - 计算量减半，训练速度约 2-3x
      - 参数: 1.87M → ~1.35M
    """

    def __init__(self, num_channel=64, num_features=201):
        super().__init__()
        del num_features  # Kept for CLI/checkpoint compatibility with old constructors.
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)

        self.t60_head = T60HeadWithPool(bottleneck_dim=num_channel, hidden_dim=128)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)

        return self.t60_head(out_3)
