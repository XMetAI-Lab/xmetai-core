"""PuYun LKA-FCN building blocks."""

from typing import Optional

import torch
import torch.nn as nn

from .drop import DropPath

__all__ = [
    "ConvLayerNorm",
    "LargeKernelAttention",
    "PuYunBlock",
]


class ConvLayerNorm(nn.Module):
    """LayerNorm over channels for BCHW feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class LargeKernelAttention(nn.Module):
    """Large kernel attention used by PuYun LKA-FCN blocks."""

    def __init__(self, dim: int, kernel_size: int = 9, dilation: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        dilated_kernel = 2 * dilation - 1
        padding = kernel_size // 2
        dilated_padding = dilation * (dilated_kernel // 2)

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            kernel_size=dilated_kernel,
            padding=dilated_padding,
            dilation=dilation,
            groups=dim,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class PuYunBlock(nn.Module):
    """ConvNeXt-style block with PuYun large-kernel attention."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 9,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ConvLayerNorm(dim)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act1 = nn.GELU()
        self.attn = LargeKernelAttention(dim, kernel_size=kernel_size)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

        self.norm2 = ConvLayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def _scale(self, x: torch.Tensor, gamma: Optional[torch.Tensor]) -> torch.Tensor:
        if gamma is None:
            return x
        return x * gamma.view(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = self.proj1(h)
        h = self.act1(h)
        h = self.attn(h)
        h = self.proj2(h)
        h = self.drop(h)
        x = residual + self.drop_path(self._scale(h, self.gamma1))

        residual = x
        h = self.norm2(x)
        h = self.mlp(h)
        x = residual + self.drop_path(self._scale(h, self.gamma2))
        return x
