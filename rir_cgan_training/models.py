import torch
from torch import nn


class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(),
                                 nn.Linear(cond_dim, cond_dim), nn.SiLU())

    def forward(self, t60):
        return self.net(t60)


class Generator(nn.Module):
    """WaveGAN-style conditional 1D generator."""

    def __init__(self, z_dim=128, cond_dim=32, base_channels=64, rir_length=32768):
        super().__init__()
        if rir_length % 32 != 0:
            raise ValueError("rir_length must be divisible by 32")
        self.cond = ConditionEmbedding(cond_dim)
        self.start = rir_length // 32
        self.fc = nn.Linear(z_dim + cond_dim, base_channels * 16 * self.start)
        widths = [base_channels * 16, base_channels * 8, base_channels * 4,
                  base_channels * 2, base_channels]
        blocks = []
        for a, b in zip(widths[:-1], widths[1:]):
            blocks += [nn.ConvTranspose1d(a, b, 4, 2, 1), nn.BatchNorm1d(b), nn.GELU()]
        blocks += [nn.ConvTranspose1d(widths[-1], 1, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*blocks)

    def forward(self, z, t60):
        h = torch.cat([z, self.cond(t60)], dim=1)
        h = self.fc(h).view(h.size(0), -1, self.start)
        return self.net(h)


class Critic(nn.Module):
    def __init__(self, cond_dim=32, base_channels=64, rir_length=32768):
        super().__init__()
        self.cond = ConditionEmbedding(cond_dim)
        widths = [1, base_channels, base_channels * 2, base_channels * 4,
                  base_channels * 8, base_channels * 16]
        blocks = []
        for a, b in zip(widths[:-1], widths[1:]):
            blocks += [nn.Conv1d(a, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.features = nn.Sequential(*blocks)
        final_length = rir_length // (2 ** (len(widths) - 1))
        self.head = nn.Sequential(nn.Linear(widths[-1] * final_length + cond_dim, 256),
                                  nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, rir, t60):
        h = self.features(rir).flatten(1)
        return self.head(torch.cat([h, self.cond(t60)], dim=1))
