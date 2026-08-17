#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: shenzhewen &&
Date: 2026-04-20 12:00:00
Description: TIANQUAN-S2S: A SUBSEASONAL-TO-SEASONAL GLOBAL WEATHER MODEL VIA INCORPORATE CLIMATOLOGY STATE

Channel layout (s2s 67-channel):
    PL vars: z,t,u,v,q  x 13 levels = 65 channels
    Surface vars:  = 2 channels

Copyright (c) 2026 by XMetAI/shenzhewen, All Rights Reserved.
"""

# ==============================================================================
# 1. 标准库
# ==============================================================================
import math
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional, Tuple, Union

# ==============================================================================
# 2. 第三方科学计算库
# ==============================================================================
import numpy as np
import pandas as pd
import xarray as xr

# NumPy 兼容性处理 (针对旧代码调用 np.float 的补丁)
if not hasattr(np, "float"):
    np.float = float

# ==============================================================================
# 3. PyTorch 及深度学习生态
# ==============================================================================
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
from xmetai.layers import PerceiverResampler
from xmetai.layers.weight_init import vit_fill

# ==============================================================================
# 4. xmetai 框架
# ==============================================================================
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

from .base_predictor import BasePredictor

# ==============================================================================
# 5. 类型定义与全局常量
# ==============================================================================

# 地球半径 (单位: km)
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
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    This function is a PyTorch port of the PyPI package `area`. In particular, it is heavily
    inspired by the following file:

        https://github.com/scisco/area/blob/9d9549d6ebffcbe4bffe11b71efa2d406d1c9fe9/area/__init__.py

    Args:
        polygon (:class:`torch.Tensor`): Polygon of the shape `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of the polygon, and 2
            concatenates first latitudes and then longitudes. The polygon does not have be closed.

    Returns:
        :class:`torch.Tensor`: Area in square kilometers.
    """
    # Be sure to close the loop.
    polygon = torch.cat((polygon, polygon[..., :1, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]  # Number of points of the polygon

    rad = torch.deg2rad  # Convert degrees to radians.

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


# Determine a reasonable smallest value for the scale embedding by assuming a smallest delta in
# latitudes and longitudes.
_delta = 0.01  # Reasonable smallest delta in latitude and longitude
_min_patch_area: float = area(
    torch.tensor(
        [
            # The smallest patches will be at the poles. Just use the north pole.
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
""":class:`.FourierExpansion`: Fourier expansion for the encoding of latitudes and longitudes in
degrees."""

scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of patch areas in squared
kilometers."""

lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
""":class:`.FourierExpansion`: Fourier expansion for the lead time encoding in hours."""

levels_expansion = FourierExpansion(0.01, 1e5)
""":class:`.FourierExpansion`: Fourier expansion for the pressure level encoding in hPa."""

absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
""":class:`.FourierExpansion`: Fourier expansion for the absolute time encoding in hours."""


class LevelPatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    def __init__(
        self,
        var_names: Tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        history_size: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """Initialise.

        Args:
            var_names (tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimensionality.
            history_size (int, optional): Number of history dimensions. Defaults to `1`.
            norm_layer (torch.nn.Module, optional): Normalisation layer to be applied at the very
                end. Defaults to no normalisation layer.
            flatten (bool): At the end of the forward pass, flatten the two spatial dimensions
                into a single dimension. See :meth:`LevelPatchEmbed.forward` for more details.
        """
        super().__init__()

        self.var_names = var_names
        self.kernel_size = (history_size,) + to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict(
            {
                # Shape (C_out, C_in, T, H, W). `C_in = 1` here because we're embedding every
                # variable separately.
                name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            }
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Setting `a = sqrt(5)` in kaiming_uniform is the same as initialising with
        # `uniform(-1/sqrt(k), 1/sqrt(k))`, where `k = weight.size(1) * prod(*kernel_size)`.
        # For more details, see
        #
        #   https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        #
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # The following initialisation is taken from
        #
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
        #
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: Tuple[str,]) -> torch.Tensor:
        """Run the embedding.

        Args:
            x (:class:`torch.Tensor`): Tensor to embed of a shape of `(B, V, T, H, W)`.
            var_names (tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`torch.Tensor`: Embedded tensor a shape of `(B, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, D, H', W')`.

        """
        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[1]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[2]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        # Select the weights of the variables and history dimensions that are present in the batch.
        weight = torch.cat(
            [
                # (C_out, C_in, T, H, W)
                self.weights[name][:, :, ...]
                for name in var_names
            ],
            dim=1,
        )
        # Adjust the stride if history is smaller than maximum.
        stride = (T,) + self.kernel_size[1:]

        # The convolution maps (B, V, T, H, W) to (B, D, 1, H/P, W/P)
        proj = F.conv3d(x, weight, self.bias, stride=stride)
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)  # (B, D, L)
            proj = proj.transpose(1, 2)  # (B, L, D)

        x = self.norm(proj)
        return x


class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

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
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to 1e-5.
        """
        super().__init__()

        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        # Latent tokens
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        # One latent level will be used by the surface level.
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings
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

        # Learnable pressure level aggregation
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

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
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
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """

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
            # Accept x_static as (C,H,W), (B,C,H,W), or (B,T,C,H,W), then broadcast to (B,T,C,H,W).
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

        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Patch embed the atmospheric levels.
        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = self.atmos_token_embeds(x_atmos, self.atmos_vars)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        atmos_levels_tensor = torch.tensor(self.atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in hours]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """Unpatchify hidden representation.

    Args:
        x (torch.Tensor): Patchified input of shape `(B, L, C, V * P^2)` where `P` is the
            patch size.
        V (int): Number of variables.
        H (int): Number of latitudes.
        W (int): Number of longitudes.

    Returns:
        torch.Tensor: Unpatchified representation of shape `(B, V, C, H, W)`.
    """
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
    """Assert that `lat` and `lon` are at least `float32`s."""
    assert lat.dtype in [
        torch.float32,
        torch.float64,
    ], f"Latitude num. unstable: {lat.dtype}."
    assert lon.dtype in [
        torch.float32,
        torch.float64,
    ], f"Longitude num. unstable: {lon.dtype}."


def init_weights(m: nn.Module) -> None:
    """Initialise weights of a module with a truncated normal distribution.

    `nn.LayerNorm` is initialised with a `weight` of 1 and a `bias` of 0.

    Args:
        m (torch.nn.Module): Module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Diffusion(nn.Module):
    """Simple 1D convolutional diffusion block."""

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
        # x: [B, L, D] -> [B, D, L]
        B, L, D = x.shape
        y = x.view(B, D, L)
        y = self.net(y)
        return y.view(B, L, D)


class Block(nn.Module):
    """Transformer block with optional skip and checkpointing."""

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
    """
    U-shaped Vision Transformer with additive diffusion at three stages.
    """

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

        # diffusion modules
        self.diff1 = Diffusion(embed_dim)
        self.diff2 = Diffusion(embed_dim)
        self.diff3 = Diffusion(embed_dim)

        # encoder blocks
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
        """
        Forward pass with optional additive noise from diffusion modules.
        Returns output tensor of shape [B, L, D].
        """
        self.mus = []
        self.sigmas = []
        self.scales = []

        skips = []
        # encode
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        # diffusion stage 1
        out = x
        sigma = self.diff1(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # middle block
        x = self.mid_block(x)

        # diffusion stage 2
        out = x
        sigma = self.diff2(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # decode
        for blk in self.out_blocks:
            skip = skips.pop()
            x = blk(x, skip)

        # diffusion stage 3
        out = x
        sigma = self.diff3(x) if use_diffusion else torch.zeros_like(x)
        x = x + sigma * torch.randn_like(x)
        self._record(out, sigma)

        # final norm+linear
        x = self.norm(x)
        return self.linear(x)

    def _record(self, mu, sigma):
        self.mus.append(mu)
        self.sigmas.append(sigma)
        self.scales.append(float(sigma.mean().detach()))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


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
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture.
    decoder and encoder

    """

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
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            embed_dim (int, optional): Embedding dim.. Defaults to `1024`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `1`.
            head_dim (int, optional): Dimension of the attention heads used in the aggregation
                blocks. Defaults to `64`.
            num_heads (int, optional): Number of attention heads used in the aggregation blocks.
                Defaults to `8`.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimensionality.
                Defaults to `4.0`.
            drop_rate (float, optional): Drop-out rate for input patches. Defaults to `0.0`.
            perceiver_ln_eps (float, optional): Layer norm. epsilon for the Perceiver blocks.
                Defaults to `1e-5`.
        """
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
        """Deaggregate pressure level information.

        Args:
            level_embed (torch.Tensor): Level embedding of shape `(B, L, C, D)`.
            x (torch.Tensor): Aggregated input of shape `(B, L, C', D)`.

        Returns:
            torch.Tensor: Deaggregate output of shape `(B, L, C, D)`.
        """
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
        """Forward pass of MultiScaleEncoder.

        Args:
            x (torch.Tensor): Backbone output of shape `(B, L, D)`.
            batch (:class:`aurora.batch.Batch`): Batch to make predictions for.
            patch_res (tuple[int, int, int]): Patch resolution
            lead_time (timedelta): Lead time.

        Returns:
            :class:`aurora.batch.Batch`: Prediction for `batch`.
        """

        # Compress the latent dimension from the U-net skip concatenation.
        B, L, D = x.shape

        # Extract the lat, lon and convert to float32.
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        # Unwrap the latent level dimension.
        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )

        # Decode surface vars. Run the head for every surface-level variable.
        x_surf = torch.stack([self.surf_heads[name](x[..., :1, :]) for name in self.surf_vars], dim=-1)
        x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)
        surf_preds = unpatchify(x_surf, len(self.surf_vars), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)

        # Embed the atmospheric levels.
        atmos_levels_encode = levels_expansion(torch.tensor(self.atmos_levels, device=x.device), self.embed_dim).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)  # (C_A, D)

        # De-aggregate the hidden levels into the physical levels.
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :])  # (B, L, C_A, D)

        # Decode the atmospheric vars.
        x_atmos = torch.stack([self.atmos_heads[name](x_atmos) for name in self.atmos_vars], dim=-1)
        x_atmos = x_atmos.reshape(*x_atmos.shape[:3], -1)  # (B, L, C_A, V_A*p*p)
        atmos_preds = unpatchify(x_atmos, len(self.atmos_vars), H, W, self.patch_size)

        return surf_preds, atmos_preds


def mse(pred, y, vars, lat=None, mask=None):
    """
    Mean Squared Error per variable.

    Args:
        pred (Tensor[B, V, H, W]): Predictions.
        y (Tensor[B, V, H, W]): Ground truth.
        vars (list[str]): Variable names.
        mask (Tensor or None): Optional mask of shape [H, W].
    Returns:
        dict: MSE per variable and overall loss.
    """
    error = (pred - y) ** 2
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = error[:, i].mean()
        if mask is not None:
            loss_dict["loss"] = (error.mean(dim=1) * mask).sum() / mask.sum()
        else:
            loss_dict["loss"] = error.mean(dim=1).mean()
    return loss_dict


def crps(y_pred, y_true, transform, vars, lat, clim, log_postfix=None, anomaly=None):
    """Continuous Ranked Probability Score per variable and overall.

    Pure PyTorch CRPS via the standard energy-score decomposition
    (Gneiting & Raftery, 2007):

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the ensemble and y is the
    observation.  Latitude-``cos`` weighting is applied before spatial
    averaging.

    Args:
        y_pred (Tensor[B, M, V, H, W]): Ensemble predictions.
        y_true (Tensor[B, V, H, W]): Observation / ground truth.
        transform: Optional transform applied to both pred and truth.
        vars (list[str]): Variable names for per-variable logging.
        lat (Tensor[H]): Latitude vector for area weighting.
        clim, log_postfix, anomaly: Unused; kept for API compatibility.
    """
    if transform is not None:
        y_pred = transform(y_pred)
        y_true = transform(y_true)

    B, M, V, H, W = y_pred.shape

    # Latitude weights: cos(lat), mean-normalised.
    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(lat, device=y_pred.device, dtype=y_pred.dtype)
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = (w_lat / w_lat.mean()).view(1, 1, -1, 1)  # (1, 1, H, 1)

    crps_dict = {}
    with torch.no_grad():
        # --- per-variable CRPS --------------------------------------------------
        for i, var in enumerate(vars):
            pred_i = y_pred[:, :, i]  # (B, M, H, W)
            true_i = y_true[:, i]  # (B, H, W)

            # Observation term: E|X - y|
            abs_err = torch.abs(pred_i - true_i.unsqueeze(1)).mean(dim=1)  # (B, H, W)

            # Pair term: 0.5 * E|X - X'| via sorted-gap formula (O(M log M))
            sorted_pred = pred_i.sort(dim=1).values  # (B, M, H, W)
            gaps = sorted_pred[:, 1:] - sorted_pred[:, :-1]  # (B, M-1, H, W)
            idx = torch.arange(1, M, device=pred_i.device, dtype=pred_i.dtype)
            pair_weights = idx * (M - idx)  # (M-1,)
            pair_weights = pair_weights.view(1, -1, 1, 1)
            pair = (gaps * pair_weights).sum(dim=1) / (M * (M - 1))  # (B, H, W)

            crps_raw = abs_err - pair  # (B, H, W)
            crps_dict[f"crps_{var}_{log_postfix}"] = (crps_raw * w_lat.squeeze(1)).mean().item()

        # --- overall CRPS (all variables jointly) ------------------------------
        # Reshape to (B, M, V*H*W) — treat spatial+vars as a single dim.
        flat_pred = y_pred.reshape(B, M, -1)  # (B, M, V*H*W)
        flat_true = y_true.reshape(B, -1)  # (B, V*H*W)

        abs_all = torch.abs(flat_pred - flat_true.unsqueeze(1)).mean(dim=1)
        sorted_all = flat_pred.sort(dim=1).values
        gaps_all = sorted_all[:, 1:] - sorted_all[:, :-1]
        weights_all = torch.arange(1, M, device=flat_pred.device, dtype=flat_pred.dtype)
        weights_all = weights_all * (M - weights_all)
        pair_all = (gaps_all * weights_all.view(1, -1)).sum(dim=1) / (M * (M - 1))

        crps_all = (abs_all - pair_all).mean()  # scalar
        crps_dict["crps"] = crps_all.item()

    return crps_dict


@MODELS.register()
class TianQuan(BasePredictor):
    """
    ResCast sub-seasonal-to-seasonal forecasting model.

    Args:
        default_vars (list[str]): List of variable names used as inputs.
        root_dir (str): Directory to save outputs (e.g., predictions).
        img_size (tuple[int,int]): Spatial resolution of input (H, W).
        patch_size (int): Size of each patch for embedding.
        embed_dim (int): Dimension of embedding space.
        decoder_depth (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        drop_path (float): DropPath rate.
        drop_rate (float): Dropout rate.
    """

    def __init__(
        self,
        default_vars,
        img_size=(128, 256),
        window_size=(8, 8),
        patch_size=2,
        embed_dim=384,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4,
        drop_path=0.1,
        drop_rate=0.1,
        const_chans=0,
        needed_ids=None,
        climate_zarr="/mnt/d/cla.zarr",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if needed_ids is None:
            needed_ids = []
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.default_vars = default_vars
        self.const_chans = const_chans

        # Define atmospheric pressure levels and variable lists
        self.levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.upper_variables = [
            "geopotential",
            "wind_speed",
            "temperature",
            "relative_humidity",
            "specific_humidity",
        ]
        self.surface_variables = ["2m_temperature", "wind_speed_10m"] if img_size == (128, 256) else ["2m_temperature", "10m_wind_speed"]
        self.static_vars = ["lsm", "orog"]

        self.needed_ids = needed_ids

        # climit mean
        self.climate_zarr = climate_zarr
        self.climate_mean = xr.open_zarr(self.climate_zarr, consolidated=True)  # (time, channel, h, w)

        # Data fusion and anomaly branches
        self.anomaly_decoder = DEConv(default_conv, len(default_vars), 3)
        self.data_decoder = SCA(default_conv, len(default_vars), 3)
        self.fusion_encoder = FGA(len(default_vars))

        # Perceiver encoder & decoder
        self.encoder = Perceiver3DEncoder(
            surf_vars=tuple(self.surface_variables),
            static_vars=tuple(self.static_vars),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            latent_levels=8,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=64,
            drop_rate=drop_rate,
            depth=4,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = Perceiver3DDecoder(
            surf_vars=tuple(self.surface_variables),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        # Variable embedding
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # UViT backbone
        self.backbone = PEDadd_UViT(embed_dim=embed_dim, depth=decoder_depth)

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and layer weights."""
        pos = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.apply(vit_fill)

    def load_constants(self, path):
        """
        Load static constant fields (land_mask, soil_type, orography).
        Expects NetCDF files constants_1.40625.nc or constants_5.625.nc.
        """
        fname = "constants_1.40625.nc" if self.img_size == (128, 256) else "constants_5.625.nc"
        ds = xr.open_dataset(os.path.join(path, fname))
        lsm = torch.from_numpy(ds["lsm"].values.astype(np.float32))
        slt = torch.from_numpy(ds["slt"].values.astype(np.float32))
        orog = torch.from_numpy(ds["orography"].values.astype(np.float32))
        return torch.stack([lsm, slt, orog], dim=0)

    def create_var_embedding(self, dim):
        """Create learnable embeddings for each variable."""
        emb = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        var_map = {v: i for i, v in enumerate(self.default_vars)}
        return emb, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """Map variable names to their embedding indices."""
        ids = np.array([self.var_map[v] for v in vars])
        return torch.from_numpy(ids).to(device)

    def hours_to_dates(self, hours, start_date="1979-01-01T00:00:00"):
        """
        Convert forecast hours back to datetime objects.
        Args:
            hours (torch.Tensor): Hours since reference.
            start_date (str): ISO timestamp of reference.
        Returns:
            List[pd.Timestamp]
        """
        base = pd.Timestamp(start_date)
        deltas = pd.to_timedelta(hours.cpu().numpy(), unit="h")
        return (base + deltas).tolist()

    def forward_encoder(self, x, anomaly, hours, variables, lats, lons, region_info, lead_time, const):
        """
        Encode inputs via anomaly & data branches, fuse, Perceiver, and UViT backbone.
        Returns tuple of surface and atmospheric predictions.
        """
        b, t, v, h, w = x.size()
        # Merge time & variable dims
        x_flat = rearrange(x, "b t v h w -> (b t) v h w")
        anom_flat = rearrange(anomaly, "b t v h w -> (b t) v h w")

        a = self.anomaly_decoder(anom_flat)
        d = self.data_decoder(x_flat)
        fused = self.fusion_encoder(a, d)
        fused = rearrange(fused, "(b t) v h w -> b t v h w", t=t)

        # Split surface vs atmosphere
        atmo = fused[:, :, 2:].reshape(b, t, len(self.upper_variables), len(self.levels), h, w)
        surf = fused[:, :, :2].reshape(b, t, len(self.surface_variables), h, w)

        enc = self.encoder(
            surf,
            const.to(x.device),
            atmo,
            lead_time,
            lats,
            lons,
            self.hours_to_dates(hours),
        )
        patches = (
            self.encoder.latent_levels,
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        ub = self.backbone(enc, use_diffusion=False)
        surf_pred, atmo_pred = self.decoder(ub, lats, lons, patches, lead_time)
        return surf_pred, atmo_pred

    def _times_to_hours(self, times, start_date="1979-01-01T00:00:00"):
        """
        Convert times to hours since start_date for model input.
        Args:
            times: list / tensor / array of datetime or pd.Timestamp
        Returns:
            hours: torch.Tensor (int64 dtype)
        """
        base_time = pd.Timestamp(start_date)

        # 把每个时间转成和基准时间相差的小时数
        hours = []
        for t in times:
            dt = pd.Timestamp(t)
            delta_hours = (dt - base_time) // pd.Timedelta("1h")  # 取整小时
            hours.append(delta_hours)

        return torch.tensor(hours, dtype=torch.int64)

    def resize_input(self, input):
        import torch.nn.functional as F
        from einops import rearrange

        # 1. 记录原始的 batch (n) 和 time (t) 维度大小，方便后面还原
        n, t = input.shape[0], input.shape[1]

        # 2. 压平成 4 维以适配 interpolate
        x = rearrange(input, "n t c h w -> (n t) c h w")

        # 3. 将 ListConfig 强制转换为标准的整数元组
        target_size = tuple(int(i) for i in self.img_size)

        # 4. 用转换后的 target_size 进行对比和插值
        if x.shape[-2:] != target_size:
            x = F.interpolate(x.float(), size=target_size, mode="bilinear", align_corners=False).to(input.dtype)

        # 5. 核心修复：将形状从 4 维恢复回 5 维
        x = rearrange(x, "(n t) c h w -> n t c h w", n=n, t=t)

        return x

    def resize_output(self, output, input):
        """
        Interpolate output (B, C, H, W) to match input spatial dimensions (H2, W2).
        input shape is (B, T, C', H2, W2).
        """
        # 提取 input 的最后两个维度作为目标尺寸 (H2, W2)
        target_size = input.shape[-2:]

        # Interpolate if output spatial size differs from target
        if output.shape[-2:] != target_size:
            output = F.interpolate(output.float(), size=target_size, mode="bilinear", align_corners=False).to(  # Interpolation requires float dtype
                output.dtype
            )  # Restore original dtype (handles half-precision cases)

        return output

    def resize_const(self, const):
        if const is not None:
            target_size = tuple(int(i) for i in self.img_size)
            if const.dim() == 3:
                # 形状: (C, H, W)
                c_tmp = const.unsqueeze(0).float()
                c_tmp = F.interpolate(c_tmp, size=target_size, mode="bilinear", align_corners=False)
                resized_const = c_tmp.squeeze(0).to(const.dtype)
            elif const.dim() == 4:
                # 形状: (B, C, H, W)
                resized_const = F.interpolate(const.float(), size=target_size, mode="bilinear", align_corners=False).to(const.dtype)
            else:
                # 如果出现其他意外维度，也可以尝试直接走你自己的 resize_input 或者报错
                resized_const = const
        else:
            resized_const = None
        return resized_const

    def replace_year_with_reference(self, idx, year=1979):
        replaced = []
        for ts in idx:
            try:
                # 尝试直接替换年份
                new_ts = ts.replace(year=year)
            except ValueError:
                # 如果触发 ValueError，检查是否是闰年 2 月 29 日的越界问题
                if ts.month == 2 and ts.day == 29:
                    # 如果目标参考年份不是闰年，将其平移到 2月 28日
                    new_ts = ts.replace(year=year, day=28)
                else:
                    # 如果是其他原因导致的 ValueError，则正常向上抛出
                    raise
            replaced.append(new_ts)

        return replaced

    def forward_step(self, input, const=None, times=None, step=None, climate_tensor=None, timestamp=None):
        """
        Single-step forward pass compatible with XMetAI framework.
        Follows UTransformer conventions: data transform, feature forward, and residual output.
        """
        if hasattr(self, "logger"):
            self.logger.debug(f"Processing step: {step}, times: {times}")

        if hasattr(self, "in_frames"):
            input = input[:, -self.in_frames :]  # e.g. bxTxcxhxw or bxtxvxhxw

        batch_size = input.shape[0]
        # 保留你原始的通道数v，方便后面裁剪
        v = input.shape[2]
        resized_input = self.resize_input(input)
        # resize
        resize_const = self.resize_const(const)
        resized_climate_tensor = self.resize_input(climate_tensor)

        if hasattr(self, "training") and (self.training or getattr(self, "use_pertub", False)):
            if getattr(self, "pertub_mixup", 0) > 0 and batch_size > 1:
                alpha = torch.rand(1, device=resized_input.device) * self.pertub_mixup
                indices = torch.randperm(batch_size, device=resized_input.device)
                resized_input = resized_input * (1 - alpha) + resized_input[indices] * alpha
                if hasattr(self, "logger"):
                    self.logger.debug(f"Applied input mixup with alpha: {alpha.item():.3f}")

        # 获取 resize 之后的宽高
        H, W = resized_input.shape[-2], resized_input.shape[-1]

        anomaly = resized_climate_tensor

        hours = self._times_to_hours(times).to(self.device)
        lead_time = timedelta(hours=24)
        lats = torch.linspace(-90, 90, H).view(-1, 1).repeat(1, W).to(self.device)
        lons = torch.linspace(0, 360, W).view(1, -1).repeat(H, 1).to(self.device)
        region_info = {"min_h": 0, "max_h": H - 1, "min_w": 0, "max_w": W - 1}

        surf_pred, atmo_pred = self.forward_encoder(resized_input, anomaly, hours, self.default_vars, lats, lons, region_info, lead_time, const=resize_const)

        # 拼接并 reshape 输出
        output = torch.cat([surf_pred, atmo_pred.reshape(batch_size, -1, H, W)], dim=1)

        # 确保输出和输入变量数一致，补齐或者裁剪
        if output.shape[1] > v:
            output = output[:, :v]
        elif output.shape[1] < v:
            pad = torch.zeros(batch_size, v - output.shape[1], H, W, device=output.device)
            output = torch.cat([output, pad], dim=1)

        if hasattr(self, "logger"):
            self.logger.debug(f"head output shape (before resize): {output.shape}")

        # 调用你的方法将输出还原为原尺寸
        output = self.resize_output(output, input)

        if hasattr(self, "logger"):
            self.logger.debug(f"final output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")

        output = output.unsqueeze(1) + input[:, -1:]
        return output

    def compute_step_loss(self, output, target, step):
        if "channel_weights" in self._buffers:
            loss_rec = charbonnier_loss(output, target, weight=self.channel_weights * self.weight)
        else:
            loss_rec = charbonnier_loss(output, target, weight=self.weight)
        step_losses = {"loss_rec": loss_rec}
        # accumulate PINN losses per-step when enabled
        if self.pinn_enable:
            pinn_losses = self.compute_pinn_losses(output, target)
            for k, v in pinn_losses.items():
                step_losses[k] = v
        return step_losses

    def forward(self, data):
        """
        Standard framework entry point, receives a unified data dictionary.
        Combines custom spatial resizing with the framework's autoregressive multi-step training logic.
        """
        # 1. 测试/推理模式
        if not self.training:
            return self.test(data)

        # 2. 训练模式，预处理数据解包
        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        targets = data["targets"]
        const = data.get("const", None) if self.const_chans > 0 else None

        # 3. 规范的日志记录
        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        # 4. 准备自回归循环变量
        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames
        loss_dict = {}

        # 5. 严格仿照的自回归多步预测循环
        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            # 切片获取当前步的 target
            target = targets[:, k : k + self.out_frames]

            self.reset_input(input)
            # 读取 climate_tensor
            curr_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=k * self.freq) for t in times], year=1979)
            pred_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=(k + 1) * self.freq) for t in times], year=1979)

            curr_climate_tensor = torch.stack(
                [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(input, non_blocking=True) for idx in curr_climate_time],
                dim=0,
            )

            pred_climate_tensor = torch.stack(
                [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(input, non_blocking=True) for idx in pred_climate_time],
                dim=0,
            )  # (bxchannelxhxw)

            climate_tensor = torch.stack([curr_climate_tensor, pred_climate_tensor], dim=1)

            # 单步预测 (注意这里严格传递了 step=t，要求你的 forward_step 支持该 kwargs)
            output = self.forward_step(input, const=const, times=times, step=t, climate_tensor=climate_tensor)

            # 计算当前步的 Loss
            step_losses = self.compute_step_loss(output, target, step=k)

            # 累加 Loss
            for k_loss, v_loss in step_losses.items():
                loss_dict[k_loss] = loss_dict.get(k_loss, 0) + v_loss

            self.logger.debug(f"step {k} loss_rec: {step_losses.get('loss_rec', 0)} with future frames {future_frames}.")

            # 准备下一步输入 (将当前 output 拼接到 input 后面，并截取最新的 in_frames 帧)
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        # 6. Loss 平均并返回
        for k_loss, v_loss in loss_dict.items():
            loss_dict[k_loss] = v_loss / future_frames

        return loss_dict

    def evaluate(
        self,
        x,
        y,
        anomaly,
        hours,
        variables,
        out_vars,
        transform,
        metrics,
        lats,
        lons,
        lat,
        clim,
        log_postfix,
        region_info,
        partition,
        lead_time=None,
    ):
        """
        Compute evaluation metrics (excluding CRPS) on a batch.
        """
        _, preds = self.forward(
            x,
            y,
            anomaly,
            hours,
            variables,
            out_vars,
            lats,
            lons,
            metric=None,
            lat=lat,
            region_info=region_info,
            lead_time=lead_time,
        )
        mi, ma = region_info["min_h"], region_info["max_h"]
        mj, mj = region_info["min_w"], region_info["max_w"]
        y_reg = y[:, :, mi : ma + 1, mj : mj + 1]
        lat_reg = lat[mi : ma + 1]
        clim_reg = clim[:, mi : ma + 1, mj : mj + 1]

        return [m(preds, y_reg, transform, out_vars, lat_reg, clim_reg, log_postfix) for m in metrics if m is not crps]

    def evaluate_ensemble(
        self,
        x,
        y,
        anomaly,
        hours,
        variables,
        out_vars,
        transform,
        metrics,
        lats,
        lons,
        lat,
        clim,
        log_postfix,
        region_info,
        partition,
    ):
        """
        Ensemble evaluation via Gaussian perturbations.
        """
        all_preds = []
        for i in range(100):
            x_in = x if i == 0 else (x + 0.3 * torch.randn_like(x))
            _, p = self.forward(
                x_in,
                y,
                anomaly,
                hours,
                variables,
                out_vars,
                lats,
                lons,
                metric=None,
                lat=lat,
                region_info=region_info,
            )
            all_preds.append(p)
        stack = torch.stack(all_preds, dim=1)
        mean_pred = stack.mean(dim=1)

        mi, ma = region_info["min_h"], region_info["max_h"]
        mj, mj = region_info["min_w"], region_info["max_w"]
        y_reg = y[:, :, mi : ma + 1, mj : mj + 1]
        lat_reg = lat[mi : ma + 1]
        clim_reg = clim[:, mi : ma + 1, mj : mj + 1]

        out = []
        for m in metrics:
            if m is crps:
                out.append(m(stack, y_reg, transform, out_vars, lat_reg, clim_reg, log_postfix))
            else:
                out.append(
                    m(
                        mean_pred,
                        y_reg,
                        transform,
                        out_vars,
                        lat_reg,
                        clim_reg,
                        log_postfix,
                    )
                )
        return out

    @torch.no_grad()
    def test_cascade(self, data):
        """
        Autoregressive inference / test entry point.
        Handles dynamic climate_tensor loading from forward,
        and relies on forward_step's internal resize for spatial resolution.
        """
        # 1. 提取基础数据与配置
        # 注意：不再在外部进行插值，而是保留原始尺寸传入 forward_step
        inputs = data["input"]  # (B, in_frames, V, H_orig, W_orig)
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        B, F_in, V, H_orig, W_orig = inputs.shape

        # 2. 初始化输出容器
        total_step = max(self.test_frames)
        steps_to_save = np.array(self.test_frames) // self.out_frames

        # 按照原始尺寸返回，以便框架计算指标
        outputs = inputs.new_zeros(B, len(steps_to_save), V, H_orig, W_orig)

        # 3. 级联预测逻辑
        # step_range 通常为 [1]，如果存在多轮级联扰动等则累加取平均
        for num_step in self.step_range:
            curr_step_idx = 0
            # 复制一份原始输入用于本轮滚动预测
            rolling_inputs = inputs.clone()

            for step in range(total_step):
                # ==========================================
                # A. 仿照 forward 构造动态 climate_tensor
                # ==========================================
                curr_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=step * self.freq) for t in times], year=1979)
                pred_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=(step + 1) * self.freq) for t in times], year=1979)

                # 从 zarr/dataset 中取出对应时间的气候态并转为 tensor (bxchannelxhxw)
                curr_climate_tensor = torch.stack(
                    [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(inputs, non_blocking=True) for idx in curr_climate_time],
                    dim=0,
                )
                pred_climate_tensor = torch.stack(
                    [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(inputs, non_blocking=True) for idx in pred_climate_time],
                    dim=0,
                )

                # (B, 2, C, H, W)
                climate_tensor = torch.stack([curr_climate_tensor, pred_climate_tensor], dim=1)

                # ==========================================
                # B. 执行单步前向推理
                # ==========================================
                # forward_step 内部已包含 self.resize_input 和 self.resize_output
                # 因此输出的 output 尺寸直接是 (B, 1, V, H_orig, W_orig)
                output = self.forward_step(rolling_inputs, const=const, times=times, step=step, climate_tensor=climate_tensor)

                # ==========================================
                # C. 滚动更新输入窗口
                # ==========================================
                # 拼接最新预测帧，并根据 in_frames 截取最新的历史数据
                rolling_inputs = torch.cat([rolling_inputs, output], dim=1)[:, -self.in_frames :]

                # ==========================================
                # D. 按需记录结果
                # ==========================================
                if (step + 1) in steps_to_save:
                    # 此时的 output 已经是残差相加且还原尺寸后的张量，直接累加即可
                    outputs[:, curr_step_idx : curr_step_idx + 1] += output
                    curr_step_idx += 1

        # 4. 如果有多个 step_range (例如 multi-ensemble / 扰动测试)，取平均值
        outputs = outputs / len(self.step_range)

        return outputs

    @torch.no_grad()
    def export_onnx(self, inputs, step, hour, doy):
        raise NotImplementedError(
            "TianQuan ONNX export is not supported yet. " "The current forward path depends on climatology tensors and datetime-aligned inputs."
        )
