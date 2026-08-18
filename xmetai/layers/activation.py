from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = [
    "GEGLU",
    "SquaredReLU",
    "SwiGLU",
    "trunc_exp",
]


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class TruncExp(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = TruncExp.apply
