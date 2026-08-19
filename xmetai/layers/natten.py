"""Neighborhood Attention (Hourglass-NATTEN) with 3D RoPE and temporal remapping.

Requires the ``Hourglass-NATTEN`` package (CUDA kernel). Conditionally imported.
"""

from __future__ import annotations

from functools import reduce
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    import natten  # Hourglass-NATTEN
except ImportError:
    natten = None

from .norm import SpatialAdaLN, _zero_init
from .rotary_attention import apply_rotary_3d_batched, freqs_cis_3d_per_sample

__all__ = [
    "NattenSelfAttention3D",
    "GlobalSelfAttention3D",
    "TransformerLayer3D",
]


class NattenSelfAttention3D(nn.Module):
    """3D neighborhood attention (NATTEN) with adaptive RMS norm and 3D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        kernel_size_thw: Tuple[int, int, int] = (3, 7, 7),
        dropout: float = 0.0,
        theta: int = 10000,
        qkv_bias: bool = True,
        cosine_sim_scale: bool = True,
    ):
        super().__init__()
        if natten is None:
            raise ModuleNotFoundError("Hourglass-NATTEN is required. `pip install Hourglass-NATTEN`")
        if dim % num_heads != 0:
            raise ValueError(f"{dim=} must be divisible by {num_heads=}")
        if (dim // num_heads) % 8 != 0:
            raise ValueError(f"head_dim={(dim // num_heads)} must be divisible by 8")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size_thw = kernel_size_thw
        self.cosine_sim_scale = cosine_sim_scale
        self.theta = theta

        self.norm = SpatialAdaLN(dim, cond_dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if cosine_sim_scale:
            self.scale = nn.Parameter(torch.full([num_heads], 10.0))
        self.drop = nn.Dropout(dropout)
        self.out = _zero_init(nn.Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, cond: Tensor, time_coord: Tensor) -> Tensor:
        skip = x
        x = self.norm(x, cond)
        B, T, H, W, _ = x.shape

        q = rearrange(self.q(x), "b t h w (nh hd) -> b t h w nh hd", nh=self.num_heads)
        kv = rearrange(self.kv(x), "b t h w (two nh hd) -> b t h w two nh hd", two=2, nh=self.num_heads)
        k, v = kv.unbind(dim=4)

        if self.cosine_sim_scale:
            dtype = reduce(torch.promote_types, (q.dtype, k.dtype, self.scale.dtype, torch.float32))
            qn = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
            kn = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
            s = rearrange(torch.sqrt(self.scale.to(dtype)), "nh -> 1 1 1 1 nh 1")
            q = q * (s * torch.rsqrt(qn + 1e-6)).to(q.dtype)
            k = k * (s * torch.rsqrt(kn + 1e-6)).to(k.dtype)

        cos, sin = freqs_cis_3d_per_sample(self.head_dim, time_coord, H, W, self.theta, q.dtype)

        q_flat = rearrange(q, "b t h w nh hd -> b nh (t h w) hd")
        k_flat = rearrange(k, "b t h w nh hd -> b nh (t h w) hd")
        q_flat = apply_rotary_3d_batched(q_flat, cos, sin)
        k_flat = apply_rotary_3d_batched(k_flat, cos, sin)
        q = rearrange(q_flat, "b nh (t h w) hd -> b t h w nh hd", t=T, h=H, w=W).contiguous()
        k = rearrange(k_flat, "b nh (t h w) hd -> b t h w nh hd", t=T, h=H, w=W).contiguous()
        v = v.contiguous()

        attn_scale = 1.0 if self.cosine_sim_scale else self.head_dim**-0.5
        out = natten.functional.neighborhood_attention_generic(
            q, k, v, kernel_size=self.kernel_size_thw, stride=1, dilation=1, is_causal=False, scale=attn_scale
        )
        out = rearrange(out, "b t h w nh hd -> b t h w (nh hd)")
        return self.out(self.drop(out)) + skip


class GlobalSelfAttention3D(nn.Module):
    """Global 3D self-attention with adaptive RMS norm and 3D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
        theta: int = 10000,
        qkv_bias: bool = True,
        cosine_sim_scale: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"{dim=} must be divisible by {num_heads=}")
        if (dim // num_heads) % 8 != 0:
            raise ValueError(f"head_dim={(dim // num_heads)} must be divisible by 8")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.cosine_sim_scale = cosine_sim_scale
        self.theta = theta

        self.norm = SpatialAdaLN(dim, cond_dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if cosine_sim_scale:
            self.scale = nn.Parameter(torch.full([num_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out = _zero_init(nn.Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, cond: Tensor, time_coord: Tensor) -> Tensor:
        skip = x
        x = self.norm(x, cond)
        B, T, H, W, _ = x.shape

        q = rearrange(self.q(x), "b t h w (nh hd) -> b t h w nh hd", nh=self.num_heads)
        kv = rearrange(self.kv(x), "b t h w (two nh hd) -> b t h w two nh hd", two=2, nh=self.num_heads)
        k, v = kv.unbind(dim=4)

        if self.cosine_sim_scale:
            dtype = reduce(torch.promote_types, (q.dtype, k.dtype, self.scale.dtype, torch.float32))
            qn = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
            kn = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
            s = rearrange(torch.sqrt(self.scale.to(dtype)), "nh -> 1 1 1 1 nh 1")
            q = q * (s * torch.rsqrt(qn + 1e-6)).to(q.dtype)
            k = k * (s * torch.rsqrt(kn + 1e-6)).to(k.dtype)

        cos, sin = freqs_cis_3d_per_sample(self.head_dim, time_coord, H, W, self.theta, q.dtype)

        q = rearrange(q, "b t h w nh hd -> b nh (t h w) hd")
        k = rearrange(k, "b t h w nh hd -> b nh (t h w) hd")
        v = rearrange(v, "b t h w nh hd -> b nh (t h w) hd")
        q = apply_rotary_3d_batched(q, cos, sin)
        k = apply_rotary_3d_batched(k, cos, sin)

        attn_scale = 1.0 if self.cosine_sim_scale else None
        out = F.scaled_dot_product_attention(q, k, v, scale=attn_scale, is_causal=False)
        out = rearrange(out, "b nh (t h w) hd -> b t h w (nh hd)", t=T, h=H, w=W)
        return self.out(self.dropout(out)) + skip


class TransformerLayer3D(nn.Module):
    """Transformer layer: local NATTEN attention + optional global attention + conditioned FFN."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        num_heads: int,
        cond_dim: int,
        kernel_size_thw: Tuple[int, int, int],
        dropout: float = 0.0,
        theta: int = 10000,
        cosine_sim_scale: bool = True,
        use_checkpoint: bool = False,
        use_global_attn: bool = False,
        global_attn_heads: Optional[int] = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_global_attn = use_global_attn

        self.local_attn = NattenSelfAttention3D(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            kernel_size_thw=kernel_size_thw,
            dropout=dropout,
            theta=theta,
            cosine_sim_scale=cosine_sim_scale,
        )
        if use_global_attn:
            self.global_attn = GlobalSelfAttention3D(
                dim=dim,
                num_heads=global_attn_heads or num_heads,
                cond_dim=cond_dim,
                dropout=dropout,
                theta=theta,
                cosine_sim_scale=cosine_sim_scale,
            )
        self.ff_norm = SpatialAdaLN(dim, cond_dim)
        from .mlp import GeGLUFFN

        self.ff = GeGLUFFN(dim, ff_dim, dim, drop=dropout)

    def _run(self, x: Tensor, cond: Tensor, time_coord: Tensor) -> Tensor:
        x = self.local_attn(x, cond, time_coord)
        if self.use_global_attn:
            x = self.global_attn(x, cond, time_coord)
        x = x + self.ff(self.ff_norm(x, cond))
        return x

    def forward(self, x: Tensor, cond: Tensor, time_coord: Tensor) -> Tensor:
        if self.use_checkpoint:
            return checkpoint(self._run, x, cond, time_coord, use_reentrant=False)
        return self._run(x, cond, time_coord)
