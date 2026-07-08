from __future__ import annotations

from typing import Callable, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = [
    "ACTIVATIONS",
    "GEGLU",
    "SquaredReLU",
    "SwiGLU",
    "get_activation",
    "trunc_exp",
]

ActivationType = Union[str, Type[nn.Module], Callable[[], nn.Module]]


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
    def forward(ctx: Function, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx: Function, g: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = TruncExp.apply

ACTIVATIONS: dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "mish": nn.Mish,
    "leaky_relu": nn.LeakyReLU,
    "geglu": GEGLU,
    "swiglu": SwiGLU,
    "relu2": SquaredReLU,
    "squared_relu": SquaredReLU,
}


def get_activation(name: ActivationType) -> nn.Module:
    """Instantiate an activation module from a name, class, or factory."""
    if isinstance(name, str):
        key = name.lower()
        if key not in ACTIVATIONS:
            supported = ", ".join(sorted(ACTIVATIONS))
            raise KeyError(f"Unknown activation {name!r}. Supported values: {supported}")
        return ACTIVATIONS[key]()
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name()
    if callable(name):
        module = name()
        if not isinstance(module, nn.Module):
            raise TypeError(f"Activation factory must return nn.Module, got {type(module)!r}")
        return module
    raise TypeError(f"Unsupported activation spec: {type(name)!r}")
