"""
Author: XMetAI
Date: 2026-08-19
Description: FuXi-2.1 building blocks.

Flat single-trunk Transformer used by the FuXi-2.1 architecture: patch/const
embedding -> Swin windowed attention (1-D RoPE) + adaLN time conditioning
(lead step / time-of-day / day-of-year) + SwiGLU FFN -> variable-aware
multi-head decoder (pressure-level deconv head + surface/derived pixel-shuffle
heads scattered back to canonical channel order).

Reference: https://huggingface.co/tpys/fuxi-2.1
The module layout mirrors the released .pt2 state_dict so that, if official
weights are ever adopted, key names stay aligned:
  decoder.patch_embed.{input_proj,norm}   decoder.const_embed.{proj,norm}
  decoder.layers.N.{norm1,attn,norm2,mlp,adaln}
  decoder.norm_layer.scale_shift           decoder.head.*
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .drop import DropPath
from .embed import sincos_embedding
from .helpers import to_2tuple
from .norm import LayerNorm
from .rotary_attention import apply_rotary_emb, precompute_freqs_cis
from .swin import window_partition, window_reverse

__all__ = [
    "SwiGLUFFN",
    "FuxiWindowAttention",
    "FuxiSwinBlock",
    "FuxiSwinLayer",
    "FuxiConstEmbed",
    "FuxiTimeEmbed",
    "FuxiPatchEmbed",
    "FuxiSubpixelHead",
    "FuxiVarHead",
]


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: silu(w1(x)) * w3(x) -> w2, no bias (matches .pt2 mlp.w1/w2/w3)."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return self.drop(x)


class FuxiWindowAttention(nn.Module):
    """Windowed multi-head attention with 1-D RoPE, evaluated over one window's tokens.

    The rotary frequencies are precomputed for the flattened window sequence
    (``win_h * win_w``) once and shared across all windows/blocks, mirroring
    FuXi-2.1's use of RoPE inside Swin windows instead of learned relative bias.
    """

    def __init__(self, dim, num_heads, window_size, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = list(to_2tuple(window_size))

        win_seq = self.window_size[0] * self.window_size[1]
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, win_seq)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # x: (nW * B, win_seq, C), mask: (nW, win_seq, win_seq) additive (0 / -100)
        B, N, C = x.shape
        q = self.wq(x).view(B, N, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.num_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.num_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, self.freqs_cos, self.freqs_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.to(x).unsqueeze(1).contiguous()  # (nW, 1, N, N)
            if mask.shape[0] != B:
                assert B % mask.shape[0] == 0, f"Attention mask batch {mask.shape[0]} must divide {B}"
                mask = mask.repeat([B // mask.shape[0], 1, 1, 1])

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.wo(x))


class FuxiSwinBlock(nn.Module):
    """Swin block with window attention (RoPE) + SwiGLU FFN, conditioned via adaLN."""

    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        mlp_ratio=8 / 3,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_size = list(to_2tuple(input_size))
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]
        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = FuxiWindowAttention(dim, num_heads, self.window_size, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio), drop=drop)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        if max(self.shift_size) > 0:
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, win_h, win_w, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, time_embedding=None):
        H, W = self.input_size
        B, L, C = x.shape
        shortcut = x

        gamma = self.adaln(time_embedding).unsqueeze(1)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = gamma.chunk(6, dim=-1)
        x = self.norm1(x) * (1 + scale_msa) + shift_msa

        x = x.view(B, H, W, C)

        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, win_h, win_w, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, win_seq, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, win_seq, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(gate_msa * x)
        x = x + self.drop_path(gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_size={self.input_size}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class FuxiSwinLayer(nn.Module):
    """Stack of FuxiSwinBlock with alternating (shifted / non-shifted) windows."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        input_size,
        window_size,
        mlp_ratio=8 / 3,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        if isinstance(drop_path, (float, int)):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList(
            [
                FuxiSwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    input_size=input_size,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, time_embedding=None):
        for blk in self.blocks:
            x = blk(x, time_embedding=time_embedding)
        return x


class FuxiTimeEmbed(nn.Sequential):
    """step / hour / doy -> per-scalar sincos(128) -> concat(384) -> MLP -> embed_dim.

    Subclasses ``nn.Sequential`` so the state_dict keys read ``joint_embed_layer.0/2``,
    matching the official FuXi-2.1 ``decoder.joint_embed_layer.*``.
    """

    def __init__(self, dim, freq=128):
        super().__init__(
            nn.Linear(freq * 3, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.freq = freq

    def forward(self, timestamp):
        embeds = [sincos_embedding(timestamp[k].float(), self.freq) for k in ("step", "hour", "doy")]
        x = torch.cat(embeds, dim=-1)
        return super().forward(x)


class FuxiPatchEmbed(nn.Module):
    """Patchify the time-varying input field. Mirrors ``decoder.patch_embed.{input_proj,norm}``."""

    def __init__(self, in_chans, in_frames, embed_dim, patch_size):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.input_proj = nn.Conv2d(in_chans * in_frames, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x):
        if x.ndim == 5:
            x = rearrange(x, "n t c h w -> n (t c) h w")
        x = self.input_proj(x).flatten(2).transpose(1, 2)  # B N C
        x = self.norm(x)
        return x


class FuxiConstEmbed(nn.Module):
    """Patchify the constant/static field. Mirrors ``decoder.const_embed.{proj,norm}``."""

    def __init__(self, const_chans, embed_dim, patch_size):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(const_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = LayerNorm(embed_dim)

    def forward(self, const):
        if const.ndim == 3:
            const = const.unsqueeze(0)
        c = self.proj(const).flatten(2).transpose(1, 2)
        return self.norm(c)


class FuxiSubpixelHead(nn.Module):
    """surface/derived head: conv -> GELU -> conv -> pixel_shuffle, ``patch_size``x upsample."""

    def __init__(self, dim, out_chans, patch_size):
        super().__init__()
        self.dim = dim
        self.out_chans = out_chans
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, out_chans * patch_size * patch_size, kernel_size=3, padding=1),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        return self.net(x)


class FuxiPressureHead(nn.Module):
    """pressure head (transposed conv), nested as ``pressure.head.deconv``.

    The extra ``head`` wrapper reproduces the official
    ``decoder.pred_layer.heads.pressure.head.deconv`` key so weights stay aligned.
    """

    def __init__(self, dim, out_chans, patch_size):
        super().__init__()
        self.head = _FuxiPressureDeconv(dim, out_chans, patch_size)

    def forward(self, x):
        return self.head(x)


class _FuxiPressureDeconv(nn.Module):
    def __init__(self, dim, out_chans, patch_size):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            dim, out_chans, kernel_size=2 * patch_size, stride=patch_size, padding=patch_size // 2
        )

    def forward(self, x):
        return self.deconv(x)


class FuxiVarHead(nn.Module):
    """Variable-aware decoder: three heads (pressure deconv / surface / derived) + scatter.

    ``group_indices`` is a list of three channel-index lists ``[pl, sfc, der]``
    into the model's ``in_names``. Group 0 (pressure levels) is decoded with a
    transposed convolution, groups 1/2 (surface, derived) with pixel-shuffle
    convs; outputs are concatenated in group order then scattered back into the
    canonical ``in_names`` ordering.
    """

    def __init__(self, dim, group_indices, patch_size, feat_size):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.feat_size = tuple(feat_size)
        pl, sfc, der = [len(g) for g in group_indices]

        # Mirrors the official ``decoder.pred_layer.heads.{pressure,surface,derived}``
        # layout (with ``pressure.head.deconv``); key names aligned for weight loading.
        self.heads = nn.ModuleDict()
        if pl > 0:
            self.heads["pressure"] = FuxiPressureHead(dim, pl, patch_size)
        if sfc > 0:
            self.heads["surface"] = FuxiSubpixelHead(dim, sfc, patch_size)
        if der > 0:
            self.heads["derived"] = FuxiSubpixelHead(dim, der, patch_size)

        self.register_buffer("scatter_idx", self._build_scatter(group_indices), persistent=False)

    @staticmethod
    def _build_scatter(group_indices):
        total = sum(len(g) for g in group_indices)
        scatter = torch.zeros(total, dtype=torch.long)
        offset = 0
        for g in group_indices:
            for i, ch in enumerate(g):
                scatter[ch] = offset + i
            offset += len(g)
        return scatter

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.feat_size
        x = x.transpose(1, 2).reshape(B, C, H, W)
        outs = [head(x) for head in self.heads.values()]
        out = torch.cat(outs, dim=1)
        return out[:, self.scatter_idx]
