"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-06 18:02:10
Description: Mamba layer for predictor in UTransformer.
Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn, selective_scan_fn
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp

from xmetai.layers.rotary_attention import RotaryAttention

__all__ = ["MambaVisionLayer"]


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size (int or tuple)
    Returns:
        local window features (num_windows*B, window_size[0]*window_size[1], C)
    """
    B, C, H, W = x.shape
    if isinstance(window_size, int):
        win_h = win_w = window_size
    else:
        win_h, win_w = window_size
    x = x.view(B, C, H // win_h, win_h, W // win_w, win_w)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, win_h * win_w, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size[0]*window_size[1], C)
        window_size: Window size (int or tuple)
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    if isinstance(window_size, int):
        win_h = win_w = window_size
    else:
        win_h, win_w = window_size
    B = int(windows.shape[0] / (H * W / win_h / win_w))
    x = windows.reshape(B, H // win_h, W // win_w, win_h, win_w, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


# define symbolic function for SelectiveScanFn
def symbolic_selective_scan_fn(
    g,
    u,
    delta,
    a,
    b,
    c,
    d=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):  # pylint: disable=too-many-arguments, invalid-name
    """
    ONNX export symbolic for Mamba::SelectiveScanFn custom op; constructs the node with optional inputs and flags.
    """
    node = g.op(
        "ai.onnx.contrib::SelectiveScanFn",
        u,
        delta,
        a,
        b,
        c,
        d if d is not None else g.op("Constant", value_t=torch.tensor([])),
        z if z is not None else g.op("Constant", value_t=torch.tensor([])),
        (delta_bias if delta_bias is not None else g.op("Constant", value_t=torch.tensor([]))),
        delta_softplus_i=int(delta_softplus),
        return_last_state_i=int(return_last_state),
    )
    # shape inference: output shape is same as input u
    node.setType(u.type())
    return node


setattr(SelectiveScanFn, "symbolic", symbolic_selective_scan_fn)


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        self.dt_proj.bias._no_weight_decay = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding="same", groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding="same", groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        if torch.onnx.is_in_onnx_export():
            # custom op only support float32
            x = x.float()
            dt = dt.float()
            B = B.float()
            C = C.float()
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False)
        if torch.onnx.is_in_onnx_export():
            y = y.to(z)
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q).to(v), self.k_norm(k).to(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        layer_scale=None,
        with_embedding=False,
        attn_type="standard",
        window_size=None,
        input_size=None,
        mamba_type="mamba",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_type = attn_type
        self.window_size = window_size
        self.input_size = input_size
        if counter in transformer_blocks:
            if attn_type.find("rotary") != -1:
                max_seq_len = None
                if input_size is not None and window_size is not None:
                    # For windowed attention, max_seq_len is the window size
                    max_seq_len = input_size[0] * input_size[1] if isinstance(input_size, (tuple, list)) else input_size * input_size
                self.mixer = RotaryAttention(
                    dim=dim,
                    n_heads=num_heads,
                    dropout=attn_drop,
                    max_seq_len=max_seq_len,
                    flash=attn_type.find("flash") != -1,
                )
            else:  # default 'standard'
                self.mixer = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    norm_layer=norm_layer,
                )
        else:
            if mamba_type in ["mamba3", "mamba2"]:
                d_inner = dim * 2
                headdim = 64 if d_inner % 64 == 0 else (32 if d_inner % 32 == 0 else 16)
                if mamba_type == "mamba3":
                    try:
                        from mamba_ssm.modules.mamba3 import Mamba3
                    except ImportError:
                        raise ImportError("Mamba3 requires latest mamba_ssm. Please install mamba-ssm.")
                    self.mixer = Mamba3(d_model=dim, d_state=64, d_conv=3, expand=2, headdim=headdim)
                else:
                    try:
                        from mamba_ssm.modules.mamba2 import Mamba2
                    except ImportError:
                        raise ImportError("Mamba2 requires mamba_ssm>=2.0.0. Please install mamba-ssm.")
                    self.mixer = Mamba2(d_model=dim, d_state=64, d_conv=3, expand=2, headdim=headdim)
            else:
                self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.with_embedding = with_embedding
        if with_embedding:
            self.time_embed = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, dim * 4, bias=True),
            )
            # init time embedding to small values to stabilize early training
            nn.init.uniform_(self.time_embed[1].weight, -0.01, 0.01)
            nn.init.constant_(self.time_embed[1].bias, 0.0)
            self.time_embed[1].weight._no_reinit = True
            self.time_embed[1].bias._no_reinit = True
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, time_embedding=None, freqs_cos=None, freqs_sin=None, **kwargs):
        if self.with_embedding and time_embedding is not None:
            gamma = self.time_embed(time_embedding).unsqueeze(1)
            shift_msa, gate_msa, shift_mlp, gate_mlp = gamma.chunk(4, dim=-1)
            gate_msa = torch.sigmoid(gate_msa)
            gate_mlp = torch.sigmoid(gate_mlp)
            norm1_x = self.norm1(x) + shift_msa
            if isinstance(self.mixer, RotaryAttention) and freqs_cos is not None and freqs_sin is not None:
                attn_out = self.mixer(norm1_x, freqs_cos, freqs_sin)
            else:
                attn_out = self.mixer(norm1_x)
            x = x + gate_msa * self.drop_path(self.gamma_1 * attn_out)
            norm2_x = self.norm2(x) + shift_mlp
            x = x + gate_mlp * self.drop_path(self.gamma_2 * self.mlp(norm2_x))
        else:
            norm1_x = self.norm1(x)
            if isinstance(self.mixer, RotaryAttention) and freqs_cos is not None and freqs_sin is not None:
                attn_out = self.mixer(norm1_x, freqs_cos, freqs_sin)
            else:
                attn_out = self.mixer(norm1_x)
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        conv=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        transformer_blocks=[],
        with_embedding=False,
        attn_type="standard",
        window_size=None,
        input_size=None,
        mamba_type="mamba",
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            input_size: input spatial size (H, W) for window partitioning.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
            attn_type: attention type ('standard' or 'rotary').
        """

        super().__init__()
        self.conv = conv
        self.window_size = window_size
        self.input_size = input_size
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    layer_scale=layer_scale,
                    with_embedding=with_embedding,
                    attn_type=attn_type,
                    window_size=window_size,
                    input_size=input_size,
                    mamba_type=mamba_type,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, time_embedding=None, freqs_cos=None, freqs_sin=None, **kwargs):
        if self.window_size is not None and self.input_size is not None:
            B, L, C = x.shape
            H, W = self.input_size

            if isinstance(self.window_size, int):
                win_h = win_w = self.window_size
            else:
                win_h, win_w = self.window_size

            x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

            pad_r = (win_w - W % win_w) % win_w
            pad_b = (win_h - H % win_h) % win_h
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                Hp, Wp = H + pad_b, W + pad_r
            else:
                Hp, Wp = H, W

            x_windows = window_partition(x, (win_h, win_w))
            nW = x_windows.shape[0] // B

            if time_embedding is not None:
                time_embedding_blk = time_embedding.repeat_interleave(nW, dim=0)
            else:
                time_embedding_blk = None

            for _, blk in enumerate(self.blocks):
                x_windows = blk(x_windows, time_embedding=time_embedding_blk, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

            x = window_reverse(x_windows, (win_h, win_w), Hp, Wp)

            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

            x = x.permute(0, 2, 3, 1).view(B, L, C)
        else:
            for _, blk in enumerate(self.blocks):
                x = blk(x, time_embedding=time_embedding, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        return x
