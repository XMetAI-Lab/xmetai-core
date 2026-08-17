#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""天权 S2S 天气预报模型使用的可复用网络模块。

本模块包含 Perceiver 跨注意力、三维气象编码器/解码器、U 形 Vision Transformer，
以及异常场与原始场融合所需的注意力和卷积模块。"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.init import trunc_normal_

from xmetai.layers.attention import Attention as FrameworkAttention
from xmetai.layers.helpers import to_2tuple
from xmetai.layers.mlp import Mlp
from xmetai.layers.weight_init import vit_fill

__all__ = [
    # Perceiver 跨注意力基础模块
    "PerceiverAttention",
    # Perceiver 重采样器（多层跨注意力 + MLP）
    "PerceiverResampler",
    # 3D 编码器：将多源气象输入编码为统一 latent tokens
    "Perceiver3DEncoder",
    # 3D 解码器：将 latent tokens 还原为地表/高空变量
    "Perceiver3DDecoder",
    # TianQuan 主干 U 形 ViT 模块
    "PEDadd_UViT",
    # 空间-通道联合增强模块
    "SCA",
    # 去噪卷积模块
    "DEConv",
    # 融合引导注意力模块
    "FGA",
]


class PerceiverAttention(nn.Module):
    """Perceiver 架构中的跨注意力模块。

    使用 latent 作为查询、context 作为键和值，将长上下文压缩到较短的 latent 序列。
    输入形状分别为 ``(B, L_latent, D_latent)`` 和 ``(B, L_context, D_context)``，
    输出形状为 ``(B, L_latent, D_latent)``。"""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """使用 latent 查询对上下文键值执行跨注意力。"""
        h = self.num_heads

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """由多层跨注意力和前馈网络组成的 Perceiver 重采样器。

    通过固定长度的 latent 序列反复聚合上下文信息，并采用残差后归一化结构。"""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                        ),
                        Mlp(
                            in_features=latent_dim,
                            hidden_features=mlp_hidden_dim,
                            drop=drop,
                        ),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """通过多层跨注意力迭代更新 latent 表示。"""
        for attn, ff, ln1, ln2 in self.layers:
            attn_out = ln1(attn(latents, x))
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents


radius_earth = 6378137 / 1000


class FourierExpansion(nn.Module):
    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        if self.assert_range:
            abs_x = x.abs()
            in_range = (abs_x >= self.lower) & (abs_x <= self.upper)
            in_range_or_zero = torch.all(in_range | (x == 0))
            if not in_range_or_zero:
                raise AssertionError(f"Input values must lie within [{self.lower}, {self.upper}].")
        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}.")

        x = x.double()
        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * math.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return encoding.float()


def patch_root_area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = 6371**2 * torch.pi * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min))) * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))
    assert (area > 0.0).all()
    return torch.sqrt(area)


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    assert lat.dim() == 1, f"lat must be 1-D, got {lat.dim()}D."
    assert lon.dim() == 1, f"lon must be 1-D, got {lon.dim()}D."
    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)
    return grid


def pos_scale_enc_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: Tuple[int, int],
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert encode_dim % 4 == 0, f"encode_dim must be a multiple of 4, got {encode_dim}."
    assert grid.dim() == 4, f"grid must be 4-D (B, 2, H, W), got {grid.dim()}D."

    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)
    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)
    return pos_encode, scale_encode


def pos_scale_enc(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: Union[int, Tuple[int, int]],
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if lat.dim() == 1 and lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == 2 and lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(f"lat and lon must both be 1-D or both 2-D, got {lat.dim()}D and {lon.dim()}D.")

    grid = grid[None]
    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )
    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)


def area(polygon: torch.Tensor) -> torch.Tensor:
    """计算由经纬度顶点定义的球面多边形面积，单位为平方千米。

    ``polygon`` 的形状为 ``(*b, n, 2)``，最后一维依次表示纬度和经度。"""
    # 补上首个顶点以闭合多边形。
    polygon = torch.cat((polygon, polygon[..., :1, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]  # 多边形顶点数量

    rad = torch.deg2rad  # 将角度转换为弧度。

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))

    area = area * radius_earth * radius_earth / 2

    return torch.abs(area)


# 根据最小经纬度间隔估计尺度编码可使用的合理下限。
_delta = 0.01  # 合理的最小经纬度间隔
_min_patch_area: float = area(
    torch.tensor(
        [
            # 最小网格面积位于极点附近，这里使用北极估算。
            [90, 0],
            [90, _delta],
            [90 - _delta, _delta],
            [90 - _delta, 0],
        ],
        dtype=torch.float64,
    )
).item()
_area_earth = 4 * np.pi * radius_earth * radius_earth

pos_expansion = FourierExpansion(_delta, 720)
"""用于经纬度角度编码的傅里叶展开。"""

scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
"""用于网格面积（平方千米）编码的傅里叶展开。"""

lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
"""用于预报时效（小时）编码的傅里叶展开。"""

levels_expansion = FourierExpansion(0.01, 1e5)
"""用于气压层（hPa）编码的傅里叶展开。"""

absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
"""用于绝对时间（小时）编码的傅里叶展开。"""


class LevelPatchEmbed(nn.Module):
    """将地表或单个气压层上的多个变量映射到统一 patch 表示。"""

    def __init__(
        self,
        var_names: Tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        history_size: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """初始化逐变量 patch 嵌入层。"""
        super().__init__()

        self.var_names = var_names
        self.kernel_size = (history_size,) + to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict(
            {
                # 权重形状为 (C_out, C_in, T, H, W)，每个变量独立嵌入，因此 C_in = 1。
                name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            }
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """初始化逐变量卷积权重和共享偏置。"""
        # kaiming_uniform 中设置 a=sqrt(5)，等价于使用 uniform(-1/sqrt(k), 1/sqrt(k)) 初始化。
        # 具体说明见
        #
        #   https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        #
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # 以下偏置初始化方式参考
        #
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
        #
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: Tuple[str,]) -> torch.Tensor:
        """将 ``(B, V, T, H, W)`` 输入编码为 patch token。

        启用 ``flatten`` 时输出形状为 ``(B, L, D)``，否则保留空间维度。"""
        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[1]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[2]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        # 按当前批次中的变量和历史长度选择对应权重。
        weight = torch.cat(
            [
                # (C_out, C_in, T, H, W)
                self.weights[name][:, :, ...]
                for name in var_names
            ],
            dim=1,
        )
        # 根据实际历史长度调整时间维步长。
        stride = (T,) + self.kernel_size[1:]

        # 卷积将 (B, V, T, H, W) 映射为 (B, D, 1, H/P, W/P)。
        proj = F.conv3d(x, weight, self.bias, stride=stride)
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)  # (B, D, L)
            proj = proj.transpose(1, 2)  # (B, L, D)

        x = self.norm(proj)
        return x


class Perceiver3DEncoder(nn.Module):
    """基于 Perceiver 的多尺度、多源、多变量三维气象编码器。"""

    def __init__(
        self,
        surf_vars: Tuple[str, ...],
        static_vars: Union[Tuple[str, str], None],
        atmos_vars: Tuple[str, str],
        atmos_levels: Tuple[int,],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
    ) -> None:
        """初始化地表、高空气象变量的 patch 嵌入和气压层聚合模块。"""
        super().__init__()

        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 模型中将静态变量与地表变量一起处理。
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        # Latent token
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        # 其中一个 latent 层用于表示地表层。
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # 用于编码地表层的可学习表示。
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # 位置、尺度和时间编码
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch 嵌入
        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(
            surf_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )
        self.atmos_token_embeds = LevelPatchEmbed(
            atmos_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )

        # 可学习的气压层聚合
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
        )

        # 编码完成后对 patch token 应用 dropout。
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # 参考 HuggingFace Perceiver 实现初始化 latent 参数：
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """将物理气压层信息聚合为固定数量的 latent 气压层。"""
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x

    def forward(
        self,
        x_surf: torch.Tensor,
        x_static: Optional[torch.Tensor],
        x_atmos: torch.Tensor,
        lead_time: timedelta,
        lat: torch.Tensor,
        lon: torch.Tensor,
        hours: torch.Tensor,
    ) -> torch.Tensor:
        """编码地表、静态和高空变量，输出形状为 ``(B, L, D)`` 的 token。"""

        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (
            B,
            T,
        ), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."
        surf_vars = self.surf_vars

        if self.static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            # 接受三种静态场形状，并统一广播为 (B, T, C, H, W)。
            if x_static.dim() == 3:
                x_static = x_static.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            elif x_static.dim() == 4:
                x_static = x_static.unsqueeze(1).expand(B, T, -1, -1, -1)
            elif x_static.dim() == 5:
                if x_static.size(0) not in (1, B):
                    raise ValueError(f"x_static batch dim must be 1 or {B}, got {x_static.size(0)}.")
                if x_static.size(1) not in (1, T):
                    raise ValueError(f"x_static time dim must be 1 or {T}, got {x_static.size(1)}.")
                x_static = x_static.expand(B, T, -1, -1, -1)
            else:
                raise ValueError(f"x_static must be 3D, 4D, or 5D, got {x_static.dim()}D.")
            x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
            surf_vars = self.surf_vars + self.static_vars

        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # 对地表变量执行 patch 嵌入。
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # 混合精度下保持后续编码的数据类型一致。

        # 对各高空气压层执行 patch 嵌入。
        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = self.atmos_token_embeds(x_atmos, self.atmos_vars)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        # 加入地表层编码，使模型能够区分地表层和高空气压层。
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # 地表层无需跨层聚合，仅使用类似 Perceiver 的 MLP 更新表示。
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # 加入形状为 (C_A, D) 的高空气压层编码。
        atmos_levels_tensor = torch.tensor(self.atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # 聚合物理气压层。
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # 拼接地表层和聚合后的高空气压层。
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # 向三维 token 加入位置和尺度编码。
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # 编码形状为 (L, D)。
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # 展平气压层和空间 token。
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # 加入预报时效编码。
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # 加入绝对时间编码。
        absolute_times_list = [t.timestamp() / 3600 for t in hours]  # 时间单位为小时
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """将 patch 表示还原为 ``(B, V, C, H, W)`` 网格。"""
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    B, C = x.size(0), x.size(2)
    H = H // P
    W = W // P
    assert x.size(1) == H * W
    assert x.size(-1) == V * P**2

    x = x.reshape(shape=(B, H, W, C, P, P, V))
    x = rearrange(x, "B H W C P1 P2 V -> B V C H P1 W P2")
    x = x.reshape(shape=(B, V, C, H * P, W * P))
    return x


def check_lat_lon_dtype(lat: torch.Tensor, lon: torch.Tensor) -> None:
    """检查经纬度张量是否至少使用 float32 精度。"""
    assert lat.dtype in [
        torch.float32,
        torch.float64,
    ], f"Latitude num. unstable: {lat.dtype}."
    assert lon.dtype in [
        torch.float32,
        torch.float64,
    ], f"Longitude num. unstable: {lon.dtype}."


def init_weights(m: nn.Module) -> None:
    """使用截断正态分布初始化网络权重，并初始化 LayerNorm。"""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


class Diffusion(nn.Module):
    """用于估计扩散扰动的简单一维卷积模块。"""

    def __init__(self, planes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(planes, 2 * planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 将形状从 [B, L, D] 转换为一维卷积需要的 [B, D, L]。
        B, L, D = x.shape
        y = x.view(B, D, L)
        y = self.net(y)
        return y.view(B, L, D)


class Block(nn.Module):
    """支持跳跃连接和梯度检查点的 Transformer 块。"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FrameworkAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, act_layer=nn.GELU)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip_tensor=None):
        if self.skip_linear is not None and skip_tensor is not None:
            x = self.skip_linear(torch.cat([x, skip_tensor], dim=-1))
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PEDadd_UViT(nn.Module):
    """在三个阶段加入扩散扰动的 U 形 Vision Transformer。"""

    def __init__(
        self,
        embed_dim=1536,
        depth=8,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        skip=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 三个阶段的扩散扰动模块
        self.diff1 = Diffusion(embed_dim)
        self.diff2 = Diffusion(embed_dim)
        self.diff3 = Diffusion(embed_dim)

        # 编码器块
        half = depth // 2
        self.in_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    norm_layer,
                    skip=False,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(half)
            ]
        )

        self.mid_block = Block(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            norm_layer,
            skip=False,
            use_checkpoint=use_checkpoint,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    norm_layer,
                    skip=skip,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(half)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.apply(vit_fill)

    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x, use_diffusion=True):
        """执行 U 形 Transformer 前向传播，并可选择是否加入扩散扰动。"""
        self.mus = []
        self.sigmas = []
        self.scales = []

        skips = []
        # 编码阶段
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # 第一阶段扩散扰动
        out = x
        sigma = self.diff1(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # 中间 Transformer 块
        x = self.mid_block(x)

        # 第二阶段扩散扰动
        out = x
        sigma = self.diff2(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # 解码阶段
        for blk in self.out_blocks:
            skip = skips.pop()
            x = blk(x, skip)

        # 第三阶段扩散扰动
        out = x
        sigma = self.diff3(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # 最终归一化和线性映射
        x = self.norm(x)
        return self.linear(x)

    def _record(self, mu, sigma):
        self.mus.append(mu)
        self.sigmas.append(sigma)
        self.scales.append(float(sigma.mean().detach()))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange("b c t h w -> b (c t) h w")(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class SCA(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DEConv(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


class FGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class Perceiver3DDecoder(nn.Module):
    """基于 Perceiver 的多尺度、多源、多变量三维气象解码器。"""

    def __init__(
        self,
        surf_vars: Tuple[str, str],
        atmos_vars: Tuple[str, str],
        atmos_levels: Tuple[int, int],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
    ) -> None:
        """初始化气压层解聚合器以及地表和高空变量输出头。"""
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels
        self.embed_dim = embed_dim

        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            residual_latent=True,
            ln_eps=perceiver_ln_eps,
        )

        self.surf_heads = nn.ParameterDict({name: nn.Linear(embed_dim, patch_size**2) for name in surf_vars})
        self.atmos_heads = nn.ParameterDict({name: nn.Linear(embed_dim, patch_size**2) for name in atmos_vars})

        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """将 latent 气压层解聚合到配置的物理气压层。"""
        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)  # (BxL, C, D)
        x = x.flatten(0, 1)  # (BxL, C', D)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert level_embed.dim() == 3, f"Expected 3 dims, found {level_embed.dim()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."

        x = self.level_decoder(level_embed, x)  # (BxL, C, D)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        patch_res: Tuple[int, int, int],
        lead_time: timedelta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将主干 token 解码为地表变量和高空变量预测结果。"""

        # 获取主干输出形状。
        B, L, D = x.shape

        # 提取经纬度并转换为 float32。
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        # 恢复 latent 气压层维度。
        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )

        # 使用独立输出头解码每个地表变量。
        x_surf = torch.stack([self.surf_heads[name](x[..., :1, :]) for name in self.surf_vars], dim=-1)
        x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)
        surf_preds = unpatchify(x_surf, len(self.surf_vars), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)

        # 编码物理气压层。
        atmos_levels_encode = levels_expansion(torch.tensor(self.atmos_levels, device=x.device), self.embed_dim).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)  # (C_A, D)

        # 将 latent 气压层解聚合为物理气压层。
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :])  # (B, L, C_A, D)

        # 使用独立输出头解码每个高空变量。
        x_atmos = torch.stack([self.atmos_heads[name](x_atmos) for name in self.atmos_vars], dim=-1)
        x_atmos = x_atmos.reshape(*x_atmos.shape[:3], -1)  # (B, L, C_A, V_A*p*p)
        atmos_preds = unpatchify(x_atmos, len(self.atmos_vars), H, W, self.patch_size)

        return surf_preds, atmos_preds
