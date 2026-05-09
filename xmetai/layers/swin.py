from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from xmetai.layers import DropPath, LayerNorm, to_2tuple

from .attention import WindowAttention, WindowAttentionV2
from .mlp import GeGLUFFN
from .rotary_attention import RotaryAttention

__all__ = ["SwinLayer"]


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: (win_h, win_w)

    Returns:
        windows: (num_windows*B, win_h, win_w, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: (win_h, win_w)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = windows.shape[0] // (H * W // (window_size[0] * window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_size (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        attn_type="v1",
        mask_type="hw",
        norm_type="ln",
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        ffn_type="gegluffn",
    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio
        self.norm_type = norm_type
        self.attn_type = attn_type
        self.ffn_type = ffn_type

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        if norm_type == "ln":
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        elif norm_type == "adaln":
            self.norm1 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True),
            )

        if attn_type == "v1":
            self.attn = WindowAttention(
                dim,
                window_size=self.window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attn_type == "v2":
            self.attn = WindowAttentionV2(
                dim,
                window_size=self.window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif "rotary" in attn_type:
            max_seq_len = np.prod(input_size)
            self.attn = RotaryAttention(
                dim,
                num_heads,
                n_kv_heads=num_heads,
                dropout=attn_drop,
                max_seq_len=max_seq_len,
                flash=attn_type.endswith("flash"),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if max(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
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
                    if mask_type == "h":
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == "w":
                        img_mask[:, :, w, :] = cnt
                    elif mask_type == "hw":
                        img_mask[:, h, w, :] = cnt

                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H, W = self.input_size
        B, L, C = x.shape
        if torch.jit.is_scripting():
            assert L == H * W, "input feature has wrong size"
        shortcut = x

        if self.norm_type == "adaln":
            gamma = self.adaln(time_embedding).unsqueeze(1)
            scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = gamma.chunk(6, dim=-1)
            x = self.norm1(x) * (1 + scale_msa) + shift_msa
        else:
            x = self.norm1(x)

        x = x.view(B, H, W, C)

        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if "rotary" in self.attn_type:
            attn_windows = self.attn(x_windows, freqs_cos, freqs_sin, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        if self.norm_type == "adaln":
            x = shortcut + gate_msa * x
            x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_size={self.input_size}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_size
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinLayer(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        input_size,
        window_size,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.0,
        attn_type="v1",
        mask_type="hw",
        norm_type="ln",
        ffn_type="gegluffn",
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                attn_type=attn_type,
                mask_type=mask_type,
                norm_type=norm_type,
                drop_path=drop_path,
                ffn_type=ffn_type,
            )
            self.blocks.append(blk)

    def forward(self, h, time_embedding=None, freqs_cos=None, freqs_sin=None):
        for blk in self.blocks:
            h = blk(
                h,
                time_embedding=time_embedding,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            )
        return h
