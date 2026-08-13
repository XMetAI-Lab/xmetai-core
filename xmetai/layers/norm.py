from functools import reduce

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from .weight_init import ddpm_fill
from .wrappers import BatchNorm2d

__all__ = [
    "LayerNorm",
    "AdaptiveLN",
    "RMSNorm",
    "AdaRMSNorm",
    "SpatialAdaLN",
]


def _zero_init(layer: nn.Module) -> nn.Module:
    if getattr(layer, "weight", None) is not None:
        nn.init.zeros_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _rms_norm(x: Tensor, scale: Tensor, eps: float) -> Tensor:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    s = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * s.to(x.dtype)


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=self.dim, keepdim=True)
        return x / (norm + self.eps)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dim = dim

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        u = x.mean(-1, keepdim=True)
        s = x.var(-1, keepdim=True)
        out = (x - u) / torch.sqrt(s + self.eps)
        return out

    def forward(self, x):
        output = self._norm(x.float()).to(x)
        if self.elementwise_affine:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class AdaptiveLN(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        in_dim = cond_dim if cond_dim else dim
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim or dim, 2 * dim, bias=True),
        )

    def forward(self, x, embed):
        scale, shift = self.scale_shift(embed).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, out_channels, style_dim):
        super().__init__()

        num_groups = min(out_channels // 4, 32)

        self.norm = nn.GroupNorm(num_groups, out_channels, affine=False, eps=1e-6)
        self.style = nn.Linear(style_dim, out_channels * 2)
        ddpm_fill(self.style)

        self.style.bias.data[:out_channels] = 1
        self.style.bias.data[out_channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class AdaptiveGroupNorm2D(nn.Module):
    def __init__(self, out_channels, style_dim):
        super().__init__()

        num_groups = min(out_channels // 4, 32)

        self.norm = nn.GroupNorm(num_groups, out_channels, affine=False, eps=1e-6)
        self.style = nn.Conv2d(style_dim, out_channels * 2, 1)
        # ddpm_fill(self.style)
        nn.init.constant_(self.style.weight, 0)
        self.style.bias.data[:out_channels] = 1
        self.style.bias.data[out_channels:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, out_channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(out_channels)
        self.style = nn.Linear(style_dim, out_channels * 2)
        ddpm_fill(self.style)

        self.style.bias.data[:out_channels] = 1
        self.style.bias.data[out_channels:] = 0

    def forward(self, input, style):
        input = rearrange(input, "n l c -> n c l")
        style = self.style(style).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        out = rearrange(out, "n c l -> n l c")
        return out


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as
            # well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return _rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.to_scale = _zero_init(nn.Linear(cond_dim, dim, bias=False))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        s = self.to_scale(cond)
        while s.dim() < x.dim():
            s = s.unsqueeze(1)
        return _rms_norm(x, s + 1, self.eps)


# Author: zhoushuang — SpatialAdaLN(空间变化 AdaLN:不池化地形 + 多级 interpolate)原创。
class SpatialAdaLN(nn.Module):
    """Spatially-varying AdaLN(条件在地形等空间场上做 FiLM,不池化)。

    复合空间条件 ``cond_2d (B, cond_dim, H, W)``(如 timestep/周期向量 broadcast + static
    patch 后 concat)→ 1×1 Conv 出 γ, β ``(B, dim, H, W)`` → ``Norm(x) * (1+γ) + β``。
    γ,β 与 x 同 H,W(空间变化),沿 T 广播。zero-init → 初始 identity。

    Args:
        x: (B, T, H, W, dim) — token 序列(channels-last)。
        cond_2d: (B, cond_dim, H, W) — 复合空间条件(channels-first,H,W 与 x 一致)。
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Conv2d(cond_dim, dim * 2, kernel_size=1)
        _zero_init(self.proj)  # γ,β zero-init → 初始 identity

    def forward(self, x: Tensor, cond_2d: Tensor) -> Tensor:
        gb = self.proj(cond_2d)  # (B, dim*2, Hc, Wc)
        gamma, beta = gb.chunk(2, dim=1)  # (B, dim, Hc, Wc)
        # backbone 多级下采样时,x 的 H,W 可能和 cond_2d 不同 → 插值对齐(保留空间细节)
        H, W = x.shape[2], x.shape[3]
        if gamma.shape[-2:] != (H, W):
            gamma = F.interpolate(gamma, size=(H, W), mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=(H, W), mode="bilinear", align_corners=False)
        x = self.norm(x)  # LN over C
        gamma = gamma.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, dim) 沿 T 广播
        beta = beta.permute(0, 2, 3, 1).unsqueeze(1)
        return x * (1.0 + gamma) + beta


def get_norm(norm, out_channels, groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": LayerNorm,
            "BN": BatchNorm2d,
            "IN": nn.InstanceNorm2d,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(groups, channels),
            "nnSyncBN": nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels)
