from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DarknetConvUNet2D"]


def _num_groups(channels: int) -> int:
    for groups in (32, 24, 16, 12, 8, 6, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    freq = torch.exp(-np.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    emb = timesteps[:, None] * freq[None]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DarknetConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.act(self.norm(self.conv(x))))


class DarknetBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden_channels = max(channels // 2, 16)
        self.reduce = DarknetConv(channels, hidden_channels, 1, dropout=dropout)
        self.time_proj = nn.Linear(time_dim, hidden_channels)
        self.expand = DarknetConv(hidden_channels, channels, 3, dropout=dropout)

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        residual = x
        x = self.reduce(x)
        x = x + self.time_proj(time_emb).to(x).reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.expand(x)
        return x + residual


class DarknetConvUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        ch_mults: Tuple[int, ...],
        layers_per_block: int,
        time_dim: int,
        dropout: float,
    ):
        super().__init__()
        if len(ch_mults) < 2:
            raise ValueError("DarknetConvUNet2D expects at least two levels, e.g. ch_mults=(1, 2).")

        channels = [base_channels * mult for mult in ch_mults]
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = DarknetConv(in_channels, channels[0], 3, dropout=dropout)
        self.enc_blocks = nn.ModuleList(
            nn.ModuleList(DarknetBlock(channels[level], time_dim, dropout=dropout) for _ in range(layers_per_block)) for level in range(len(channels))
        )
        self.downs = nn.ModuleList(DarknetConv(channels[level], channels[level + 1], 3, stride=2, dropout=dropout) for level in range(len(channels) - 1))
        self.mid = nn.ModuleList(DarknetBlock(channels[-1], time_dim, dropout=dropout) for _ in range(layers_per_block))
        self.up_projs = nn.ModuleList(DarknetConv(channels[level + 1], channels[level], 1, dropout=dropout) for level in range(len(channels) - 2, -1, -1))
        self.skip_projs = nn.ModuleList(DarknetConv(channels[level] * 2, channels[level], 1, dropout=dropout) for level in range(len(channels) - 2, -1, -1))
        self.dec_blocks = nn.ModuleList(
            nn.ModuleList(DarknetBlock(channels[level], time_dim, dropout=dropout) for _ in range(layers_per_block))
            for level in range(len(channels) - 2, -1, -1)
        )
        self.out_norm = nn.GroupNorm(_num_groups(channels[0]), channels[0])
        self.out_act = nn.LeakyReLU(0.1, inplace=True)
        self.out = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        self.conv_out = self.out

    def forward(self, x: Tensor, timesteps: Tensor, return_dict: bool = False) -> Tuple[Tensor] | dict[str, Tensor]:
        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(x.shape[0])
        elif timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.reshape(-1)[0].repeat(x.shape[0])
        time_emb = self.time_mlp(_timestep_embedding(timesteps, self.time_dim).to(x))

        x = self.stem(x)
        skips = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                x = block(x, time_emb)
            if level < len(self.downs):
                skips.append(x)
                x = self.downs[level](x)
        for block in self.mid:
            x = block(x, time_emb)

        for skip, up_proj, skip_proj, blocks in zip(reversed(skips), self.up_projs, self.skip_projs, self.dec_blocks):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = up_proj(x)
            x = skip_proj(torch.cat([x, skip], dim=1))
            for block in blocks:
                x = block(x, time_emb)
        x = self.out(self.out_act(self.out_norm(x)))
        if return_dict:
            return {"sample": x}
        return (x,)
