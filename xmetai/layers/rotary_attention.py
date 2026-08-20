from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "precompute_freqs_cis",
    "precompute_freqs_cis_3d",
    "precompute_freqs_cis_3d_remapped",
    "freqs_cis_3d_per_sample",
    "apply_rotary_3d",
    "apply_rotary_3d_batched",
    "apply_rotary_emb",
]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def precompute_freqs_cis_2d(dim: int, size: tuple[int, int], theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    y = torch.arange(size[0], device=freqs.device)
    x = torch.arange(size[1], device=freqs.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    freqs_yy = torch.outer(yy.flatten(), freqs).float()  # type: ignore
    freqs_xx = torch.outer(xx.flatten(), freqs).float()  # type: ignore
    freqs = torch.cat([freqs_yy, freqs_xx], dim=-1)
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


@lru_cache(maxsize=None)
@torch.no_grad()
def precompute_freqs_cis_3d(
    dim: int, T: int, H: int, W: int, theta: float = 10000.0, device_str: str = "cpu", dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE frequencies for (T, H, W) spatial-temporal tokens.

    Frequency allocation matches diffusers: dim_t = dim//4, dim_h = dim_w = dim//8*3.
    Each axis uses its own sub-dimension as denominator so frequency ranges are
    independent and non-overlapping.

    Returns cos, sin of shape (T * H * W, dim // 2).
    """
    # Sub-dimensions (full, not halved) per axis — matches diffusers
    dim_t = dim // 4
    dim_h = dim // 8 * 3
    dim_w = dim - dim_t - dim_h  # should equal dim // 8 * 3

    # Each axis: standard RoPE formula with its own sub-dim as denominator
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))

    t = torch.arange(T, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    freqs_t = torch.outer(tt.flatten(), freqs_t).float()
    freqs_y = torch.outer(yy.flatten(), freqs_h).float()
    freqs_x = torch.outer(xx.flatten(), freqs_w).float()
    freqs = torch.cat([freqs_t, freqs_y, freqs_x], dim=-1)
    return torch.cos(freqs).to(dtype).to(device_str), torch.sin(freqs).to(dtype).to(device_str)


@lru_cache(maxsize=None)
@torch.no_grad()
def precompute_freqs_cis_3d_remapped(
    dim: int,
    T_seq: int,
    H: int,
    W: int,
    temporal_mapping: Tuple[int, ...],
    theta: float = 10000.0,
    device_str: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE with temporal position remapping.

    Allows multiple sequence positions to share the same temporal RoPE
    (e.g. a COARSE token and a NOISE token at t=3 both use temporal index 3).

    Returns cos, sin of shape (T_seq * H * W, dim).
    """
    T_real = max(temporal_mapping) + 1
    cos, sin = precompute_freqs_cis_3d(dim, T_real, H, W, theta, device_str, dtype)
    hw = H * W
    indices = torch.empty(T_seq * hw, dtype=torch.long, device=torch.device(device_str))
    for t_seq in range(T_seq):
        t_real = temporal_mapping[t_seq]
        indices[t_seq * hw : (t_seq + 1) * hw] = torch.arange(t_real * hw, (t_real + 1) * hw, device=indices.device)
    return cos[indices], sin[indices]


# Author: zhoushuang — 以下 per-sample 动态时间 RoPE 原创支持任意实数时间/异步模态/任意 t0;
# 上方静态 precompute_* 为原 xmetai 既有实现。
def _rope_freqs_3d(dim: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3D RoPE 每轴的频率向量(t/h/w),子维分配与 ``precompute_freqs_cis_3d`` 一致。"""
    dim_t = dim // 4
    dim_h = dim // 8 * 3
    dim_w = dim - dim_t - dim_h
    freq_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
    freq_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freq_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))
    return freq_t, freq_h, freq_w


def freqs_cis_3d_per_sample(
    dim: int,
    time_coords: torch.Tensor,
    H: int,
    W: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """per-sample 3D RoPE:任意实数时间坐标 → cos, sin。

    Args:
        dim:         head_dim(= RoPE 作用维度)。
        time_coords: ``(B, T)`` float —— 每 slot 的真实物理时间(per-sample,任意实数)。
        H, W:        当前层空间尺寸(多级下采样时逐层不同)。
    Returns:
        cos, sin of shape ``(B, T * H * W, dim // 2)``。
    """
    freq_t, freq_h, freq_w = _rope_freqs_3d(dim, theta)
    freq_t = freq_t.to(time_coords.device)
    freq_h = freq_h.to(time_coords.device)
    freq_w = freq_w.to(time_coords.device)
    B, T = time_coords.shape

    # 时间轴:per-sample 任意实数
    args_t = time_coords.float().unsqueeze(-1) * freq_t.view(1, 1, -1)  # (B, T, dim_t//2)
    # 空间轴:整数网格(0..H-1, 0..W-1)
    y = torch.arange(H, device=time_coords.device, dtype=torch.float32)
    x = torch.arange(W, device=time_coords.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)
    args_h = yy.unsqueeze(-1) * freq_h.view(1, 1, -1)  # (H, W, dim_h//2)
    args_w = xx.unsqueeze(-1) * freq_w.view(1, 1, -1)  # (H, W, dim_w//2)

    cos_t = torch.cos(args_t)[:, :, None, None, :].expand(B, T, H, W, args_t.shape[-1])
    sin_t = torch.sin(args_t)[:, :, None, None, :].expand(B, T, H, W, args_t.shape[-1])
    cos_h = torch.cos(args_h)[None, None, :, :, :].expand(B, T, H, W, args_h.shape[-1])
    sin_h = torch.sin(args_h)[None, None, :, :, :].expand(B, T, H, W, args_h.shape[-1])
    cos_w = torch.cos(args_w)[None, None, :, :, :].expand(B, T, H, W, args_w.shape[-1])
    sin_w = torch.sin(args_w)[None, None, :, :, :].expand(B, T, H, W, args_w.shape[-1])

    cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).reshape(B, T * H * W, -1).to(dtype)
    sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).reshape(B, T * H * W, -1).to(dtype)
    return cos, sin


def apply_rotary_3d_batched(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """per-sample 3D RoPE。

    Args:
        x:   ``(B, num_heads, seq, head_dim)``。
        cos, sin: ``(B, seq, head_dim // 2)`` —— 来自 :func:`freqs_cis_3d_per_sample`。
    """
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)  # (B, nh, seq, head_dim//2)
    cos = cos[:, None, :, :]  # (B, 1, seq, head_dim//2)
    sin = sin[:, None, :, :]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    return torch.stack([out_r, out_i], dim=-1).flatten(-2).to(x.dtype)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_3d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 3D rotary position embedding to a single tensor (NATTEN-style).

    Assumes x has shape (..., seq, head_dim) where the last two dims are the
    flattened spatiotemporal sequence and head dimension.  This is the layout
    used after rearranging 3D tokens (B, T, H, W) into a flat sequence.

    Args:
        x: (..., seq, head_dim) — e.g. (B, num_heads, seq, head_dim).
        cos, sin: (seq, head_dim // 2) — from precompute_freqs_cis_3d.
    """
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    n_leading = x_r.ndim - 2
    cos = cos.reshape((1,) * n_leading + cos.shape)
    sin = sin.reshape((1,) * n_leading + sin.shape)
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    return torch.stack([out_r, out_i], dim=-1).flatten(-2).to(x.dtype)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r).float()
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r).float()

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.to(xq), xk_out.to(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = False,
        flash: bool = False,
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.flash = flash

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.max_seq_len is None:
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        else:
            ns = self.max_seq_len // seq_len
            xq = xq.view(-1, ns * seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(-1, ns * seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if self.max_seq_len is not None:
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seq_len, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seq_len, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if mask is not None:
            mask = mask.to(xq).unsqueeze(1).contiguous()
            if mask.shape[0] != bsz:
                assert bsz % mask.shape[0] == 0, f"Attention mask batch {mask.shape[0]} must divide input batch {bsz}"
                mask = mask.repeat([bsz // mask.shape[0], 1, 1, 1])

        # flash implementation
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / np.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).to(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
