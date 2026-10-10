"""
Gated DeltaNet-2 sequence layers for UTransformer token mixers.

This module keeps the public layer contract aligned with ``MambaVisionLayer``:
inputs are ``(B, L, C)`` token tensors, optional AdaLN time embeddings are
accepted, and each block is mixer + MLP with residual connections.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmetai.utils.env import ensure_torchvision_nms_schema

from .drop import DropPath
from .mlp import Mlp
from .swin import SwinBlock

__all__ = ["GatedDeltaNet2Block", "GatedDeltaNet2HybridLayer", "GatedDeltaNet2Layer", "GatedDeltaNet2Mixer"]


def _load_external_gdn2() -> Any:
    try:
        from fla.layers import GatedDeltaNet2
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
        ensure_torchvision_nms_schema()
        from fla.layers import GatedDeltaNet2
    except ImportError as exc:
        raise ImportError("GatedDeltaNet-2 layer_type requires flash-linear-attention with " "`from fla.layers import GatedDeltaNet2` available.") from exc
    return GatedDeltaNet2


class GatedDeltaNet2OnnxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, initial_state, scale, use_qk_l2norm_in_kernel):
        if torch.compiler.is_compiling():
            return torch.onnx.ops.symbolic(
                "xmetai_plugins::GatedDeltaNet2Fn",
                (q, k, v, g, b, w, initial_state),
                {
                    "scale": float(scale),
                    "use_qk_l2norm_in_kernel": int(use_qk_l2norm_in_kernel),
                    "output_final_state": 0,
                    "use_gate_in_kernel": 0,
                    "has_cu_seqlens": 0,
                    "state_v_first": 0,
                },
                dtype=v.dtype,
                shape=v.shape,
                version=1,
            ).to(device=v.device)
        return torch.empty_like(v)

    @staticmethod
    def symbolic(g, q, k, v, decay, beta, w, initial_state, scale, use_qk_l2norm_in_kernel):
        node = g.op(
            "xmetai_plugins::GatedDeltaNet2Fn",
            q,
            k,
            v,
            decay,
            beta,
            w,
            initial_state,
            scale_f=float(scale),
            use_qk_l2norm_in_kernel_i=int(use_qk_l2norm_in_kernel),
            output_final_state_i=0,
            use_gate_in_kernel_i=0,
            has_cu_seqlens_i=0,
            state_v_first_i=0,
        )
        node.setType(v.type())
        return node


def _reshape_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], num_heads, head_dim)


def _short_conv_onnx(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    kernel_size = conv.kernel_size[0]
    y = F.conv1d(
        x.transpose(1, 2),
        conv.weight,
        bias=conv.bias,
        padding=kernel_size - 1,
        groups=conv.groups,
    )
    y = y[..., : x.shape[1]].transpose(1, 2)
    if getattr(conv, "activation", None) in ("silu", "swish"):
        y = F.silu(y)
    return y


def _rms_norm_gated_onnx(norm: nn.Module, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    if norm.weight is not None:
        y = y * norm.weight.float()
    if getattr(norm, "bias", None) is not None:
        y = y + norm.bias.float()
    activation = getattr(norm, "activation", "swish")
    gate = gate.float()
    if activation in ("silu", "swish"):
        y = y * F.silu(gate)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gate)
    else:
        raise RuntimeError(f"GatedDeltaNet2 ONNX export does not support norm activation {activation!r}")
    return y.to(dtype)


def _gdn2_mixer_onnx_forward(mixer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if getattr(mixer, "use_short_conv", False):
        q = _short_conv_onnx(mixer.q_conv1d, mixer.q_proj(hidden_states))
        k = _short_conv_onnx(mixer.k_conv1d, mixer.k_proj(hidden_states))
        v = _short_conv_onnx(mixer.v_conv1d, mixer.v_proj(hidden_states))
    else:
        q = F.silu(mixer.q_proj(hidden_states))
        k = F.silu(mixer.k_proj(hidden_states))
        v = F.silu(mixer.v_proj(hidden_states))

    decay = F.softplus(mixer.f_proj(hidden_states).float() + mixer.dt_bias)
    beta = torch.sigmoid(mixer.b_proj(hidden_states))
    w = torch.sigmoid(mixer.w_proj(hidden_states))

    q = _reshape_heads(q, mixer.num_heads, mixer.head_k_dim)
    k = _reshape_heads(k, mixer.num_heads, mixer.head_k_dim)
    decay = _reshape_heads(decay, mixer.num_heads, mixer.head_k_dim)
    beta = _reshape_heads(beta, mixer.num_heads, mixer.head_k_dim)
    v = _reshape_heads(v, mixer.num_v_heads, mixer.head_v_dim)
    w = _reshape_heads(w, mixer.num_v_heads, mixer.head_v_dim)

    decay = -mixer.A_log.float().exp().view(1, 1, mixer.num_heads, 1) * decay
    if mixer.num_v_heads > mixer.num_heads:
        repeats = mixer.num_v_heads // mixer.num_heads
        q = q.repeat_interleave(repeats, dim=-2)
        k = k.repeat_interleave(repeats, dim=-2)
        decay = decay.repeat_interleave(repeats, dim=-2)
        beta = beta.repeat_interleave(repeats, dim=-2)
    if mixer.allow_neg_eigval:
        beta = beta * 2.0

    initial_state = q.new_empty((0,), dtype=torch.float32)
    scale = mixer.head_k_dim**-0.5
    o = GatedDeltaNet2OnnxFn.apply(q.float(), k.float(), v.float(), decay.float(), beta.float(), w.float(), initial_state, float(scale), True)
    gate = _reshape_heads(mixer.g_proj(hidden_states), mixer.num_v_heads, mixer.head_v_dim)
    o = _rms_norm_gated_onnx(mixer.o_norm, o, gate)
    o = o.reshape(*o.shape[:-2], mixer.value_dim)
    return mixer.o_proj(o.to(hidden_states.dtype))


class GatedDeltaNet2Mixer(nn.Module):
    """Adapter around the external ``fla.layers.GatedDeltaNet2`` module."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_v: float = 1.0,
        head_dim: Optional[int] = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        mode: str = "chunk",
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            head_dim = hidden_size // num_heads

        external_cls = _load_external_gdn2()
        self.mixer = external_cls(
            mode=mode,
            hidden_size=hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            conv_bias=conv_bias,
            allow_neg_eigval=allow_neg_eigval,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return _gdn2_mixer_onnx_forward(self.mixer, hidden_states)
        output = self.mixer(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
        return output


class GatedDeltaNet2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        with_embedding: bool = False,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        allow_neg_eigval: bool = False,
    ) -> None:
        super().__init__()
        self.with_embedding = with_embedding
        self.norm1 = norm_layer(dim, elementwise_affine=not self.with_embedding)
        mixer = GatedDeltaNet2Mixer(
            hidden_size=dim,
            num_heads=num_heads,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            allow_neg_eigval=allow_neg_eigval,
        )
        self.mixer = mixer
        if with_embedding:
            self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
            self.scale_shift_table._no_weight_decay = True

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=not self.with_embedding)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del freqs_cos, freqs_sin, kwargs
        _, _, channels = x.shape
        if self.with_embedding and time_embedding is not None:
            gamma = time_embedding.unsqueeze(1) + self.scale_shift_table.view(6 * channels).unsqueeze(0).unsqueeze(1)
            scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = gamma.chunk(6, dim=-1)
            norm1_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        else:
            norm1_x = self.norm1(x)

        mixed = self.mixer(norm1_x)
        if self.with_embedding and time_embedding is not None:
            x = x + gate_msa * self.drop_path(self.gamma_1 * mixed)
            norm2_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = x + gate_mlp * self.drop_path(self.gamma_2 * self.mlp(norm2_x))
        else:
            x = x + self.drop_path(self.gamma_1 * mixed)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class GatedDeltaNet2Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        layer_scale: Optional[float] = None,
        with_embedding: bool = False,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        allow_neg_eigval: bool = False,
    ) -> None:
        super().__init__()
        self.with_embedding = with_embedding
        if self.with_embedding:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
            nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
            nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)
            self.adaLN_modulation[1].weight._no_reinit = True
            self.adaLN_modulation[1].bias._no_reinit = True

        self.blocks = nn.ModuleList(
            [
                GatedDeltaNet2Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    layer_scale=layer_scale,
                    with_embedding=with_embedding,
                    expand_v=expand_v,
                    use_short_conv=use_short_conv,
                    conv_size=conv_size,
                    allow_neg_eigval=allow_neg_eigval,
                )
                for i in range(depth)
            ]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if getattr(param, "_no_reinit", False) or getattr(param, "_no_weight_decay", False):
                continue
            if ".mixer.mixer." in name:
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

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        if self.with_embedding and time_embedding is not None:
            time_embedding = self.adaLN_modulation(time_embedding)

        for blk in self.blocks:
            x = blk(x, time_embedding=time_embedding, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        return x


def _half_window_size(window_size: Union[int, Tuple[int, int], List[int]]) -> Union[int, Tuple[int, int]]:
    if isinstance(window_size, int):
        return window_size // 2
    return (window_size[0] // 2, window_size[1] // 2)


class GatedDeltaNet2HybridLayer(nn.Module):
    """
    Hybrid GDN2/Swin layer following the GDN2+MLP, Swin-window-attention+MLP pattern.

    Even-indexed blocks use GatedDeltaNet-2. Odd-indexed blocks use the
    existing SwinBlock, which partitions the full 2D token grid into windows.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        input_size: Tuple[int, int],
        window_size: Union[int, Tuple[int, int]],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        layer_scale: Optional[float] = None,
        with_embedding: bool = False,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        allow_neg_eigval: bool = False,
        attn_type: str = "v1",
        mask_type: str = "hw",
        ffn_type: str = "gegluffn",
    ) -> None:
        super().__init__()
        self.with_embedding = with_embedding
        if self.with_embedding:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
            nn.init.constant_(self.adaLN_modulation[1].weight, 0.0)
            nn.init.constant_(self.adaLN_modulation[1].bias, 0.0)
            self.adaLN_modulation[1].weight._no_reinit = True
            self.adaLN_modulation[1].bias._no_reinit = True

        blocks = []
        swin_block_idx = 0
        for i in range(depth):
            current_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            if i % 2 == 0:
                blocks.append(
                    GatedDeltaNet2Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop,
                        drop_path=current_drop_path,
                        layer_scale=layer_scale,
                        with_embedding=with_embedding,
                        expand_v=expand_v,
                        use_short_conv=use_short_conv,
                        conv_size=conv_size,
                        allow_neg_eigval=allow_neg_eigval,
                    )
                )
            else:
                blocks.append(
                    SwinBlock(
                        dim=dim,
                        input_size=input_size,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if swin_block_idx % 2 == 0 else _half_window_size(window_size),
                        mlp_ratio=mlp_ratio,
                        attn_type=attn_type,
                        mask_type=mask_type,
                        norm_type="adaln" if with_embedding else "ln",
                        drop=drop,
                        drop_path=current_drop_path,
                        ffn_type=ffn_type,
                    )
                )
                swin_block_idx += 1
        self.blocks = nn.ModuleList(blocks)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if getattr(param, "_no_reinit", False) or getattr(param, "_no_weight_decay", False):
                continue
            if ".mixer.mixer." in name:
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

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        gdn2_time_embedding = self.adaLN_modulation(time_embedding) if self.with_embedding and time_embedding is not None else None

        # Construction alternates GDN/Swin; AC/FSDP wrappers do not change that order.
        for index, block in enumerate(self.blocks):
            embedding = gdn2_time_embedding if index % 2 == 0 else time_embedding
            x = block(x, time_embedding=embedding, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        return x
