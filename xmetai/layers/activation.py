import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["GEGLU", "trunc_exp"]


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class TruncExp(Function):
    @staticmethod
    def forward(ctx, x):
        x = x.float()
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = TruncExp.apply
