import torch
from torch.autograd import Function

__all__ = ["trunc_exp"]


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
