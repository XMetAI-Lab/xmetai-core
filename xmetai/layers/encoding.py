import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["StepEncoding", "CyclicTimesteps"]


class StepEncoding(nn.Module):
    def __init__(self, dim, max_period=1e4):
        super().__init__()
        self.dim_half = dim // 2
        self.max_period = torch.tensor(max_period, dtype=torch.float32)

    def forward(self, x):
        emb = torch.log(self.max_period) / (self.dim_half - 1)
        emb = torch.exp(torch.arange(self.dim_half, dtype=torch.float32) * -emb)
        emb = rearrange(x.float(), "i -> i 1") * rearrange(emb.to(x.device), "j -> 1 j")
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.to(x)


class TimeEncoding(nn.Module):
    def __init__(self, freq=6):
        super().__init__()
        self.freq = freq

    def forward(self, i, times):
        bs = times.shape[0]
        hours = np.array([pd.Timedelta(hours=t * self.freq) for t in [i - 1, i, i + 1]])
        times = times[:, None] + hours[None]
        times = [pd.Period(t, "H") for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        emb = torch.from_numpy(np.array(times, dtype=np.float32))
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.reshape(bs, -1)


class CyclicTimesteps(nn.Module):
    """Sinusoidal harmonic embeddings for periodic variables (e.g. day-of-year, hour-of-day)."""

    def __init__(
        self,
        num_channels: int,
        period: float,
        max_harmonic: int | None = None,
        flip_sin_to_cos: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(f"num_channels must be even, got {num_channels}.")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}.")

        self.num_channels = num_channels
        self.period = float(period)
        self.flip_sin_to_cos = flip_sin_to_cos

        half = num_channels // 2
        if max_harmonic is None:
            max_harmonic = half
        if max_harmonic > half:
            raise ValueError(f"max_harmonic ({max_harmonic}) cannot exceed num_channels//2 ({half}).")

        self.max_harmonic = max_harmonic
        k = torch.arange(1, max_harmonic + 1, dtype=dtype)
        self.register_buffer("k", k, persistent=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() != 1:
            raise ValueError(f"Expected 1-D input (B,), got shape {s.shape}.")
        s = s.float()
        k = self.k.to(s.device)
        phase = (2.0 * math.pi / self.period) * s[:, None] * k[None, :]
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        half = self.num_channels // 2
        if self.max_harmonic < half:
            pad = torch.zeros(s.shape[0], half - self.max_harmonic, device=s.device, dtype=s.dtype)
            sin_part = torch.cat([sin_part, pad], dim=-1)
            cos_part = torch.cat([cos_part, pad], dim=-1)
        if self.flip_sin_to_cos:
            return torch.cat([cos_part, sin_part], dim=-1)
        return torch.cat([sin_part, cos_part], dim=-1)
