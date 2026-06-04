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

from .attention import Attention, WindowAttention, WindowAttentionV2
from .drop import DropPath
from .mlp import Mlp
from .rotary_attention import RotaryAttention
from .swin import window_partition, window_reverse

__all__ = ["MambaVisionLayer"]


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
):
    """
    ONNX export symbolic for Mamba::SelectiveScanFn custom op; constructs the node with optional inputs and flags.
    """
    node = g.op(
        "xmetai_plugins::SelectiveScanFn",
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


class Mamba3SisoOnnxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size, has_d, has_z):
        return torch.empty_like(v)

    @staticmethod
    def symbolic(g, q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, chunk_size, has_d, has_z):
        node = g.op(
            "xmetai_plugins::Mamba3SisoFn",
            q,
            k,
            v,
            adt,
            dt,
            trap,
            q_bias,
            k_bias,
            angles,
            d if has_d else g.op("Constant", value_t=torch.tensor([])),
            z if has_z else g.op("Constant", value_t=torch.tensor([])),
            chunk_size_i=int(chunk_size),
            has_d_i=int(has_d),
            has_z_i=int(has_z),
            return_final_states_i=0,
            has_input_states_i=0,
            has_cu_seqlens_i=0,
        )
        node.setType(v.type())
        return node


def _register_mamba3_siso_symbolic():
    try:
        from mamba_ssm.modules import mamba3 as mamba3_module
    except ImportError:
        return

    rmsnorm_cls = getattr(mamba3_module, "RMSNormGated", None)
    if rmsnorm_cls is not None and not getattr(rmsnorm_cls, "_xmetai_onnx_export_registered", False):
        original_rmsnorm_forward = rmsnorm_cls.forward

        def rmsnorm_export_forward(self, x, z=None):
            if torch.onnx.is_in_onnx_export():
                hidden_size = x.shape[-1]
                group_size = self.group_size if self.group_size is not None else hidden_size
                y = x.float()
                z_float = z.float() if z is not None else None
                if z_float is not None and not self.norm_before_gate:
                    y = y * F.silu(z_float)
                if group_size != hidden_size:
                    y = y.reshape(*x.shape[:-1], hidden_size // group_size, group_size)
                    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                    y = y.reshape_as(x)
                else:
                    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                y = y * self.weight.float()
                if self.bias is not None:
                    y = y + self.bias.float()
                if z_float is not None and self.norm_before_gate:
                    y = y * F.silu(z_float)
                return y.to(x.dtype)
            return original_rmsnorm_forward(self, x, z=z)

        rmsnorm_cls._xmetai_original_forward = original_rmsnorm_forward
        rmsnorm_cls.forward = rmsnorm_export_forward
        rmsnorm_cls._xmetai_onnx_export_registered = True

    if getattr(mamba3_module, "_xmetai_mamba3_siso_onnx_registered", False):
        return
    original = getattr(mamba3_module, "mamba3_siso_combined", None)
    if original is None:
        return

    def mamba3_siso_onnx(
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        Angles,
        D=None,
        Z=None,
        Input_States=None,
        chunk_size=64,
        return_final_states=False,
        cu_seqlens=None,
    ):
        if torch.onnx.is_in_onnx_export():
            if Input_States is not None:
                raise RuntimeError("Mamba3SisoFn ONNX export does not support Input_States")
            if return_final_states:
                raise RuntimeError("Mamba3SisoFn ONNX export does not support return_final_states")
            if cu_seqlens is not None:
                raise RuntimeError("Mamba3SisoFn ONNX export does not support cu_seqlens")
            d_input = D if D is not None else Q.new_empty((0,), dtype=torch.float32)
            z_input = Z if Z is not None else Q.new_empty((0,), dtype=torch.float32)
            return Mamba3SisoOnnxFn.apply(
                Q.float(),
                K.float(),
                V.float(),
                ADT.float(),
                DT.float(),
                Trap.float(),
                Q_bias.float(),
                K_bias.float(),
                Angles.float(),
                d_input.float(),
                z_input.float(),
                int(chunk_size),
                D is not None,
                Z is not None,
            )
        return original(
            Q,
            K,
            V,
            ADT,
            DT,
            Trap,
            Q_bias,
            K_bias,
            Angles,
            D=D,
            Z=Z,
            Input_States=Input_States,
            chunk_size=chunk_size,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )

    mamba3_module._xmetai_original_mamba3_siso_combined = original
    mamba3_module.mamba3_siso_combined = mamba3_siso_onnx
    mamba3_module._xmetai_mamba3_siso_onnx_registered = True


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
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
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


class MambaMetMixer(MambaVisionMixer):
    """
    Bidirectional Mamba Mixer tailored for Meteorological Modeling.
    Extends MambaVisionMixer to use dual-directional causal scanning, overcoming
    the 1D single-direction bias which destroys 2D spatial physical properties.
    """

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        # Use existing 1D convolutions (could be replaced with Conv2D if geometry allows in future)
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding="same", groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding="same", groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B_proj = rearrange(B_proj, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_proj = rearrange(C_proj, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        if torch.onnx.is_in_onnx_export():
            x = x.float()
            dt = dt.float()
            B_proj = B_proj.float()
            C_proj = C_proj.float()

        # Forward scan (Top-Left to Bottom-Right)
        y_fwd = selective_scan_fn(
            x, dt, A, B_proj, C_proj, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
        )

        # Backward scan (Bottom-Right to Top-Left)
        x_bwd = torch.flip(x, dims=[-1])
        dt_bwd = torch.flip(dt, dims=[-1])
        B_bwd = torch.flip(B_proj, dims=[-1])
        C_bwd = torch.flip(C_proj, dims=[-1])

        y_bwd = selective_scan_fn(
            x_bwd, dt_bwd, A, B_bwd, C_bwd, self.D.float(), z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
        )
        y_bwd = torch.flip(y_bwd, dims=[-1])

        # Bidirectional fusion (averaging preserves scale and maintains numerical stability)
        y = (y_fwd + y_bwd) / 2.0

        if torch.onnx.is_in_onnx_export():
            y = y.to(z)
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class MambaMetWrapper(nn.Module):
    """
    A general wrapper for Mamba-2/3 to enable tied-weight bidirectional scanning
    by reversing the sequence length dimension (B, L, D).
    """

    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def forward(self, hidden_states, **kwargs):
        # Forward scan
        y_fwd = self.mixer(hidden_states, **kwargs)

        # Backward scan (flip along sequence dimension L)
        hidden_states_bwd = torch.flip(hidden_states, dims=[1])
        y_bwd = self.mixer(hidden_states_bwd, **kwargs)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Bidirectional fusion
        return (y_fwd + y_bwd) / 2.0


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
        layer_scale=None,
        with_embedding=False,
        attn_type="standard",
        window_size=None,
        input_size=None,
        mamba_type="mamba",
    ):
        super().__init__()
        self.with_embedding = with_embedding
        self.norm1 = norm_layer(dim, elementwise_affine=not self.with_embedding)
        self.attn_type = attn_type
        self.window_size = window_size
        self.input_size = input_size

        # Determine shift size for Swin capabilities
        if self.window_size is not None:
            if isinstance(self.window_size, int):
                win_h = win_w = self.window_size
            else:
                win_h, win_w = self.window_size
            self.shift_size = (win_h // 2, win_w // 2) if counter % 2 != 0 else (0, 0)
        else:
            self.shift_size = (0, 0)

        if counter in transformer_blocks:
            if attn_type == "v1":
                self.mixer = WindowAttention(
                    dim,
                    window_size=(win_h, win_w),
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale if not isinstance(qk_scale, bool) else None,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
            elif attn_type == "v2":
                self.mixer = WindowAttentionV2(
                    dim,
                    window_size=(win_h, win_w),
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
            elif "rotary" in attn_type:
                max_seq_len = None
                if input_size is not None and window_size is not None:
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
            if mamba_type in ["mamba3", "mamba2", "mamba3_met", "mamba2_met"]:
                d_inner = dim * 2
                headdim = 64 if d_inner % 64 == 0 else (32 if d_inner % 32 == 0 else 16)
                if mamba_type.startswith("mamba3"):
                    try:
                        from mamba_ssm.modules.mamba3 import Mamba3
                    except ImportError:
                        raise ImportError("Mamba3 requires latest mamba_ssm. Please install mamba-ssm.")
                    _register_mamba3_siso_symbolic()
                    mixer_inst = Mamba3(d_model=dim, d_state=64, d_conv=3, expand=2, headdim=headdim)
                else:
                    try:
                        from mamba_ssm.modules.mamba2 import Mamba2
                    except ImportError:
                        raise ImportError("Mamba2 requires mamba_ssm>=2.0.0. Please install mamba-ssm.")
                    mixer_inst = Mamba2(d_model=dim, d_state=64, d_conv=3, expand=2, headdim=headdim)

                if mamba_type.endswith("_met"):
                    self.mixer = MambaMetWrapper(mixer_inst)
                else:
                    self.mixer = mixer_inst
            elif mamba_type == "mamba_met":
                self.mixer = MambaMetMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
            else:
                self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        if with_embedding:
            # AdaLN-Single: global modulation outputs 6*dim, block adds per-module shift
            self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
            self.scale_shift_table._no_weight_decay = True
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=not self.with_embedding)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, time_embedding=None, freqs_cos=None, freqs_sin=None, **kwargs):
        B, L, C = x.shape
        if self.with_embedding and time_embedding is not None:
            # Add per-block scaling/shifting to the globally modulated time_embedding
            gamma = time_embedding.unsqueeze(1) + self.scale_shift_table.view(6 * C).unsqueeze(0).unsqueeze(1)
            scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = gamma.chunk(6, dim=-1)
            norm1_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        else:
            norm1_x = self.norm1(x)

        if self.window_size is not None and self.input_size is not None:
            H, W = self.input_size
            if isinstance(self.window_size, int):
                win_h = win_w = self.window_size
            else:
                win_h, win_w = self.window_size

            norm1_x = norm1_x.view(B, H, W, C)  # (B, H, W, C)
            pad_r = (win_w - W % win_w) % win_w
            pad_b = (win_h - H % win_h) % win_h
            if pad_r > 0 or pad_b > 0:
                norm1_x = torch.nn.functional.pad(norm1_x, (0, 0, 0, pad_r, 0, pad_b))
                Hp, Wp = H + pad_b, W + pad_r
            else:
                Hp, Wp = H, W

            if self.shift_size[0] > 0 or self.shift_size[1] > 0:
                # Cyclic shift
                shifted_x = torch.roll(norm1_x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                # Calculate attention mask for SW-MSA
                img_mask = torch.zeros((1, Hp, Wp, 1), device=norm1_x.device)
                h_slices = (slice(0, -win_h), slice(-win_h, -self.shift_size[0]), slice(-self.shift_size[0], None))
                # 只对 h 方向（纬度方向）进行 mask 隔离，w 方向（经度方向）保持环绕连续
                cnt = 0
                for h in h_slices:
                    img_mask[:, h, :, :] = cnt
                    cnt += 1
                mask_windows = window_partition(img_mask, (win_h, win_w))
                mask_windows = mask_windows.view(-1, win_h * win_w)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

                nW = attn_mask.shape[0]
                # If using Swin's native attention, it automatically broadcasts over B inside, so do not repeat.
                if not isinstance(self.mixer, (WindowAttention, WindowAttentionV2, RotaryAttention)):
                    attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1).view(B * nW, 1, win_h * win_w, win_h * win_w)
            else:
                shifted_x = norm1_x
                attn_mask = None

            x_windows = window_partition(shifted_x, (win_h, win_w))
            x_windows = x_windows.view(-1, win_h * win_w, C)

            if isinstance(self.mixer, RotaryAttention) and freqs_cos is not None and freqs_sin is not None:
                attn_out = self.mixer(x_windows, freqs_cos, freqs_sin, mask=attn_mask)
            else:
                if isinstance(self.mixer, (Attention, WindowAttention, WindowAttentionV2)) and attn_mask is not None:
                    attn_out = self.mixer(x_windows, mask=attn_mask)
                else:
                    attn_out = self.mixer(x_windows)

            attn_out = attn_out.view(-1, win_h, win_w, C)
            attn_out = window_reverse(attn_out, (win_h, win_w), Hp, Wp)

            if self.shift_size[0] > 0 or self.shift_size[1] > 0:
                attn_out = torch.roll(attn_out, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

            if pad_r > 0 or pad_b > 0:
                attn_out = attn_out[:, :H, :W, :].contiguous()
            attn_out = attn_out.view(B, H * W, C)
        else:
            if isinstance(self.mixer, RotaryAttention) and freqs_cos is not None and freqs_sin is not None:
                attn_out = self.mixer(norm1_x, freqs_cos, freqs_sin)
            else:
                attn_out = self.mixer(norm1_x)

        if self.with_embedding and time_embedding is not None:
            x = x + gate_msa * self.drop_path(self.gamma_1 * attn_out)
            norm2_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = x + gate_mlp * self.drop_path(self.gamma_2 * self.mlp(norm2_x))
        else:
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
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        transformer_blocks=None,
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
        transformer_blocks = [] if transformer_blocks is None else transformer_blocks
        self.with_embedding = with_embedding
        if self.with_embedding:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
            nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
            nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)
            self.adaLN_modulation[1].weight._no_reinit = True
            self.adaLN_modulation[1].bias._no_reinit = True

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
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if getattr(param, "_no_reinit", False) or getattr(param, "_no_weight_decay", False):
                continue
            if "norm" in name.lower() or "ln" in name.lower() or "layernorm" in name.lower():
                if name.endswith("weight") and param.dim() == 1:
                    nn.init.ones_(param)
                elif name.endswith("bias") and param.dim() == 1:
                    nn.init.zeros_(param)
            else:
                if param.dim() > 1:
                    nn.init.trunc_normal_(param, std=0.02)
                elif name.endswith("bias") and param.dim() == 1:
                    nn.init.zeros_(param)

    def forward(self, x, time_embedding=None, freqs_cos=None, freqs_sin=None, **kwargs):
        if self.with_embedding and time_embedding is not None:
            time_embedding = self.adaLN_modulation(time_embedding)

        for _, blk in enumerate(self.blocks):
            x = blk(x, time_embedding=time_embedding, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        return x
