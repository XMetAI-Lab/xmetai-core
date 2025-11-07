import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["StepEncoding", "TimeEncoding"]


class StepEncoding(nn.Module):
    def __init__(self, dim, max_period=1e4):
        super().__init__()
        self.dim_half = dim // 2
        self.max_period = torch.tensor(max_period, dtype=torch.float32)

    def forward(self, x):
        emb = torch.log(self.max_period) / (self.dim_half - 1)
        emb = torch.exp(torch.arange(self.dim_half, dtype=torch.float32) * - emb)
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
