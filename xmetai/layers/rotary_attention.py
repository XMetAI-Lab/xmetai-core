from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "precompute_freqs_cis",
    "precompute_freqs_cis_2d",
    "reshape_for_broadcast",
    "apply_rotary_emb",
    "repeat_kv",
    "RotaryAttention",
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


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
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
