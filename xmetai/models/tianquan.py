"""
Author: shenzhewen &&
Date: 2026-04-20 12:00:00
Description: TIANQUAN-S2S: A SUBSEASONAL-TO-SEASONAL GLOBAL WEATHER MODEL VIA INCORPORATE CLIMATOLOGY STATE

Channel layout (s2s 67-channel):
    PL vars: z,t,u,v,q  x 13 levels = 65 channels
    Surface vars:  = 2 channels

Copyright (c) 2026 by XMetAI/shenzhewen, All Rights Reserved.
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
from datetime import timedelta
from functools import lru_cache
from typing import List, Optional, Tuple, TypeVar, Union

# ==============================================================================
# 2. 第三方科学计算库
# ==============================================================================
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy import stats

# NumPy 兼容性处理 (针对旧代码调用 np.float 的补丁)
if not hasattr(np, "float"):
    np.float = float

# ==============================================================================
# 3. PyTorch 深度学习体系
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import to_2tuple
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp

from xmetai.builder import MODELS

from .base_predictor import BasePredictor

# ==============================================================================
# 4. 类型定义与全局常量
# ==============================================================================
T = TypeVar("T", Tuple[int, int], Tuple[int, int, int])

# 地球半径 (单位: km)
radius_earth = 6378137 / 1000
# ==============================================================================
# 5. 气象变量映射配置
# ==============================================================================

# 基础变量名映射
NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

# 1.40625分辨率或特定数据集的映射
NAME_TO_VAR_14 = {
    "2m_temperature": "t2m_avg",
    "wind_speed_10m": "wind_speed_10m_avg",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr_avg",
    "surface_net_thermal_radiation": "str_avg",
    "boundary_layer_height": "blh_avg",
    "sshf": "sshf_avg",
    "slhf": "slhf_avg",
    "msl": "msl_avg",
    "total_precipitation": "tp_sum",
    "land_sea_mask": "lsm_avg",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z_avg",
    "wind_speed": "wind_speed_avg",
    "temperature": "t_avg",
    "relative_humidity": "r_avg",
    "specific_humidity": "q_avg",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

# 变量分类定义
SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]

DEFAULT_PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

# 自动生成包含气压层级的完整变量映射
NAME_LEVEL_TO_VAR_LEVEL = {}

# 地表变量映射
for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

# 高空压力层变量映射 (例如: temperature_500 -> t_500)
for var in PRESSURE_LEVEL_VARS:
    for level in DEFAULT_PRESSURE_LEVELS:
        name_key = f"{var}_{level}"
        var_val = f"{NAME_TO_VAR[var]}_{level}"
        NAME_LEVEL_TO_VAR_LEVEL[name_key] = var_val

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

# ==============================================================================
# 6. 地理区域边界配置 (Lat: 纬度, Lon: 经度)
# ==============================================================================
BOUNDARIES = {
    "NorthAmerica": {"lat_range": (15, 65), "lon_range": (220, 300)},  # 8x14
    "SouthAmerica": {"lat_range": (-55, 20), "lon_range": (270, 330)},  # 14x10
    "Europe": {"lat_range": (30, 65), "lon_range": (0, 40)},  # 6x8
    "SouthAsia": {"lat_range": (-15, 45), "lon_range": (25, 110)},  # 10x14
    "EastAsia": {"lat_range": (5, 65), "lon_range": (70, 150)},  # 10x12
    "Australia": {"lat_range": (-50, 10), "lon_range": (100, 180)},  # 10x14
    "Global": {"lat_range": (-90, 90), "lon_range": (0, 360)},  # 32x64
    "China": {"lat_range": (0, 55), "lon_range": (70, 135)},  # 8x10
    "Sichuan": {"lat_range": (20, 40), "lon_range": (90, 110)},  # 2x2
}


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
    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)

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


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(f"The input tensor is not within the configured range" f" `[{self.lower}, {self.upper}]`.")

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


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


def patch_root_area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:
    """For a rectangular patch on a sphere, compute the square root of the area of the patch in
    units km^2. The root is taken to return units of km, and thus stay scalable between different
    resolutions.

    Args:
        lat_min (torch.Tensor): Minimum latitutes of patches.
        lon_min (torch.Tensor): Minimum longitudes of patches.
        lat_max (torch.Tensor): Maximum latitudes of patches.
        lon_max (torch.Tensor): Maximum longitudes of patches.

    Returns:
        torch.Tensor: Square root of the area.
    """
    # Calculate area of latitude-longitude grid using the following formula. Phis are latitudes
    # and thetas are longitudes.
    #
    #   area = R**2 * pi * (sin(phi_1) - sin(phi_2)) * (theta_1 - theta_2)
    #
    # Taken from
    #
    #   https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/
    #
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = 6371**2 * torch.pi * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min))) * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))

    assert (area > 0.0).all()
    return torch.sqrt(area)


def pos_scale_enc_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: Tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and scale encoding for a latitude-longitude grid.

    Requires batch dimensions in the input and returns a batch dimension.

    Args:
        encode_dim (int): Output encoding dimension `D`. Must be a multiple of four: splits
            across latitudes and longitudes and across sines and cosines.
        grid (torch.Tensor): Latitude-longitude grid of dimensions `(B, 2, H, W)`. `grid[:, 0]`
            should be the latitudes of `grid[:, 1]` should be the longitudes.
        patch_dims (tuple): Patch dimensions. Different x-values and y-values are supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(B, H/patch[0] * W/patch[1], D)`.
    """
    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    # Take the 2D pooled values of the mesh. This is the same as subsequent 1D pooling over the
    # x-axis and then ove the y-axis.
    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    # Compute the square root of the area of the patches specified by the latitude-longitude
    # grid.
    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    # Use half of dimensions for the latitudes of the midpoints of the patches and the other
    # half for the longitudes. Before computing the encodings, flatten over the spatial dimensions.
    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)  # (B, L, D)

    # No need to split things up for the scale encoding.
    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)  # (B, L, D)

    return pos_encode, scale_encode


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Construct a meshgrid of latitude and longitude coordinates.

    `torch.meshgrid(*tensors, indexing="xy")` gives the same behavior as calling
    `numpy.meshgrid(*arrays, indexing="ij")`::

        lat = torch.tensor([1, 2, 3])
        lon = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
        grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
        grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])

    Args:
        lat (torch.Tensor): Vector of latitudes.
        lon (torch.Tensor): Vector of longitudes.

    Returns:
        torch.Tensor: Meshgrid of shape `(2, len(lat), len(lon))`.
    """
    assert lat.dim() == 1
    assert lon.dim() == 1

    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)

    return grid


def pos_scale_enc(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: Union[int, List, Tuple],
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> torch.Tensor:
    """Positional encoding of latitude-longitude data.

    Does not support batch dimensions in the input and does not return batch dimensions either.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (torch.Tensor): Latitudes, `H`. Can be either a vector or a matrix.
        lon (torch.Tensor): Longitudes, `W`. Can be either a vector or a matrix.
        patch_dims (Union[list, tuple]): Patch dimensions. Different x-values and y-values are
            supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(H/patch[0] * W/patch[1], D)`.
    """
    if lat.dim() == lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, " f"but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )

    grid = grid[None]  # Add batch dimension.

    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )

    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)  # Return without batch dimension.


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
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[0]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[1]} != 0."
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
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
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
        x_static: torch.Tensor,
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

        if self.static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
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


class MLP(nn.Module):
    """A simple one-hidden-layer MLP."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimensionality.
            hidden_features (int): Width of the hidden layer.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Cross attention module from the Perceiver architecture."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            head_dim (int): Attention head dimensionality.
            num_heads (int): Number of heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

    def scaled_dot_product_attention(self, query, key, value, mask=None, dropout_p=0.0):
        import math

        """实现 scaled dot product attention"""
        d_k = query.size(-1)  # 计算 query 的最后一维大小，通常是维度 C
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout_p, training=True)

        output = torch.matmul(attention_weights, value)
        return output

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the cross-attention module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, Latent_D)`
                where typically `L1 < L2` and `Latent_D <= Context_D`. `Latent_D` is equal to
                `self.latent_dim`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, Context_D)`.

        Returns:
            :class:`torch.Tensor`: Latent values of shape `(B, L1, Latent_D)`.
        """
        h = self.num_heads

        q = self.to_q(latents)  # (B, L1, D2) to (B, L1, D)
        k, v = self.to_kv(x).chunk(2, dim=-1)  # (B, L2, D1) to twice (B, L2, D)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = self.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")  # (B, L1, D)
        return self.to_out(out)  # (B, L1, Latent_D)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler module from the Flamingo paper."""

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
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            depth (int, optional): Number of attention layers.
            head_dim (int, optional): Attention head dimensionality. Defaults to `64`.
            num_heads (int, optional): Number of heads. Defaults to `16`
            mlp_ratio (float, optional): Rimensionality of the hidden layer divided by that of the
                input for all MLPs. Defaults to `4.0`.
            drop (float, optional): Drop-out rate. Defaults to no drop-out.
            residual_latent (bool, optional): Use residual attention w.r.t. the latent features.
                Defaults to `True`.
            ln_eps (float, optional): Epsilon in the layer normalisation layers. Defaults to
                `1e-5`.
        """
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
                        MLP(dim=latent_dim, hidden_features=mlp_hidden_dim, dropout=drop),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, D1)`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, D1)`.

        Returns:
            torch.Tensor: Latent features of shape `(B, L1, D1)`.
        """
        for attn, ff, ln1, ln2 in self.layers:
            # We use post-res-norm like in Swin v2 and most Transformer architectures these days.
            # This empirically works better than the pre-norm used in the original Perceiver.
            attn_out = ln1(attn(latents, x))
            # HuggingFace suggests using non-residual attention in Perceiver might work better when
            # the semantics of the query and the output are different:
            #
            #   https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/perceiver/modeling_perceiver.py#L398
            #
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents


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


class Attention(nn.Module):
    """
    Multi-head attention with optional local window or memory-efficient global.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B L (k h d) -> k B h L d", k=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.window_size and self.window_size < L:
            out = self._local_attention(q, k, v)
        else:
            out = self._global_attention(q, k, v)

        out = rearrange(out, "B h L d -> B L (h d)")
        out = self.proj(out)
        return self.proj_drop(out)

    def _local_attention(self, q, k, v):
        """Compute attention in non-overlapping local windows."""
        B, H, L, D = q.shape
        w = self.window_size
        # reshape to (B * windows)...
        q_w = rearrange(q, "B H (n w) D -> (B n) H w D", w=w)
        k_w = rearrange(k, "B H (n w) D -> (B n) H w D", w=w)
        v_w = rearrange(v, "B H (n w) D -> (B n) H w D", w=w)

        attn = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_w = attn @ v_w  # (B*n, H, w, D)
        return rearrange(out_w, "(B n) H w D -> B H (n w) D", B=B)

    def _global_attention(self, q, k, v):
        """Chunked global attention to reduce memory usage."""
        B, H, L, D = q.shape
        chunk = 256
        out_chunks = []
        for i in range(0, L, chunk):
            q_c = q[:, :, i : i + chunk, :]
            attn = (q_c @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out_chunks.append(attn @ v)
        return torch.cat(out_chunks, dim=2)


class Block(nn.Module):
    """Transformer block with optional skip and checkpointing."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
        img_size=(128, 256),
        window_size=(8, 8),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, window_size=8)
        # 更新成我们的东西 windows size 硬编码
        # self.attn = WindowAttentionV2(dim, (1, 1), num_heads, qkv_bias, qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, act_layer=nn.GELU)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
        self.img_size = img_size
        self.window_size = window_size

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
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        skip=True,
        img_size=(128, 256),
        window_size=(8, 8),
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
                    qk_scale,
                    norm_layer,
                    skip=False,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    window_size=window_size,
                )
                for _ in range(half)
            ]
        )

        self.mid_block = Block(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            norm_layer,
            skip=False,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            window_size=window_size,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    norm_layer,
                    skip=skip,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    window_size=window_size,
                )
                for _ in range(half)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights: truncated normal for Linear, constant for LayerNorm."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

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
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
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
        super(PixelAttention, self).__init__()
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
        super(SCA, self).__init__()
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
        super(DEConv, self).__init__()
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
        super(FGA, self).__init__()
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
        assert len(level_embed.shape) == 3, f"Expected 3 dims, found {level_embed.dims()}."
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


def get_max_min_blocks(x, block_size=8):
    """
    Partition tensor [B, V, H, W] into non-overlapping blocks and compute each block’s max and min.

    Args:
        x (Tensor[B, V, H, W]): Input tensor.
        block_size (int): Block height and width.

    Returns:
        max_vals (Tensor[B, V, num_blocks]), min_vals (Tensor[B, V, num_blocks])
    """
    B, V, H, W = x.shape
    assert H % block_size == 0 and W % block_size == 0

    # Create blocks
    blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    B, V, nH, nW, bH, bW = blocks.shape
    blocks = blocks.contiguous().view(B, V, nH * nW, bH, bW)

    max_vals = blocks.max(dim=-1)[0].max(dim=-1)[0]
    min_vals = blocks.min(dim=-1)[0].min(dim=-1)[0]
    return max_vals, min_vals


def expand_blocks_to_tensor(block_vals, orig_shape, block_size=8):
    """
    Expand per-block values back to a full-resolution tensor by tiling.

    Args:
        block_vals (Tensor[B, V, num_blocks]): Values per block.
        orig_shape (tuple): Desired shape (B, V, H, W).
        block_size (int): Block size.

    Returns:
        Tensor[B, V, H, W]
    """
    B, V, num_blocks = block_vals.shape
    _, _, H, W = orig_shape
    out = torch.zeros(orig_shape, device=block_vals.device, dtype=block_vals.dtype)
    idx = 0
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            out[:, :, i : i + block_size, j : j + block_size] = block_vals[:, :, idx].unsqueeze(-1).unsqueeze(-1)
            idx += 1
    return out


def lat_weighted_mse(pred, y, vars, lat, mask=None, anomaly=None, extreme=False, weight=None):
    """
    针对 5D 真值 y 和 4D 预测值 pred 进行了兼容性处理的损失函数。
    """
    # -------------------------------------------------------------------------
    # 1. 维度对齐：核心修复
    # 如果 y 是 (B, T, V, H, W) 而 pred 是 (B, V, H, W)，我们需要将 y 降维
    # -------------------------------------------------------------------------
    if y.dim() == 5 and pred.dim() == 4:
        # 取最后一个时间步，使 y 变为 (B, V, H, W)
        y = y[:, -1]

        # 同理处理 anomaly
    if anomaly is not None and anomaly.dim() == 5 and pred.dim() == 4:
        anomaly = anomaly[:, -1]

    # -------------------------------------------------------------------------
    # 2. 计算基础误差 (确保计算在正确的设备上)
    # -------------------------------------------------------------------------
    if anomaly is not None:
        error = (pred - y) ** 2 + (pred - anomaly) ** 2
    elif extreme:
        pmax, pmin = get_max_min_blocks(pred)
        ymax, ymin = get_max_min_blocks(y)
        max_err = (pmax - ymax) ** 2
        min_err = (pmin - ymin) ** 2
        error = (pred - y) ** 2 + 0.1 * expand_blocks_to_tensor(max_err, pred.shape) + 0.1 * expand_blocks_to_tensor(min_err, pred.shape)
    elif weight is not None:
        # 确保 weight 也是 Tensor 且在正确的 device 上
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, device=pred.device, dtype=pred.dtype)
        error = (pred - y) ** 2 * weight.view(1, -1, 1, 1).to(pred.device)
    else:
        error = (pred - y) ** 2

    # -------------------------------------------------------------------------
    # 3. 纬度加权计算 (完全使用 Torch，避免 CUDA/NumPy 冲突)
    # -------------------------------------------------------------------------
    # 确保 lat 是在当前设备上的 Tensor
    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(lat, device=error.device, dtype=error.dtype)

    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    # 维度对齐到 (1, H, 1) 以便与 (B, H, W) 的通道进行广播
    w_lat = w_lat.view(1, -1, 1).to(error.device)

    # -------------------------------------------------------------------------
    # 4. 计算 Loss (注意：总 Loss 的计算必须在 no_grad 之外，否则模型不更新！)
    # -------------------------------------------------------------------------
    loss_dict = {}

    # 我们先在有梯度的情况下计算总 loss
    if mask is not None:
        # 确保 mask 维度对齐
        current_loss = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        # 使用自动平衡逻辑计算总 Loss
        base = (error[:, 0] * w_lat).mean()
        agg = 0.0
        for i in range(len(vars)):
            term = (error[:, i] * w_lat).mean()
            # 使用 .detach() 动态缩放比例，但保留 term 的梯度路径
            agg += term / (term / (base + 1e-8)).detach()
        current_loss = agg

    # -------------------------------------------------------------------------
    # 5. 记录日志 (使用 no_grad 记录各变量具体数值，不影响反向传播)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        for i, var in enumerate(vars):
            term = error[:, i] * w_lat
            if mask is not None:
                loss_dict[var] = (term * mask).sum() / mask.sum()
            else:
                loss_dict[var] = term.mean()

    # 将最终用于 backward 的 loss 存入字典
    loss_dict["loss"] = current_loss

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation latitude-weighted MSE, logging per-variable with a suffix.

    Args:
        transform: function to apply to pred and y before error computation.
        clim: (unused) climatology placeholder.
    """
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(error.device).to(error.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()
        loss_dict["w_mse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation latitude-weighted RMSE (root MSE).

    Args:
        transform: callable to apply to pred and y.
    """
    pred_t = transform(pred)
    y_t = transform(y)
    error = (pred_t - y_t) ** 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(error.device).to(error.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            rmse_i = torch.sqrt((error[:, i] * w_lat).mean(dim=(-2, -1)))
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = rmse_i.mean()
        loss_dict["w_rmse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def rmse(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation unweighted RMSE.
    """
    pred_t = transform(pred)
    y_t = transform(y)
    error = (pred_t - y_t) ** 2

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            rmse_i = torch.sqrt(error[:, i].mean(dim=(-2, -1)))
            loss_dict[f"rmse_{var}_{log_postfix}"] = rmse_i.mean()
        loss_dict["rmse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Latitude-weighted anomaly correlation (accuracy).

    Args:
        clim (Tensor[V, H, W]): Climatology to remove mean seasonal cycle.
    """
    pred_t = transform(pred) - clim.unsqueeze(0)
    y_t = transform(y) - clim.unsqueeze(0)

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(pred.device).to(pred.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            p = pred_t[:, i] - pred_t[:, i].mean()
            g = y_t[:, i] - y_t[:, i].mean()
            num = (w_lat * p * g).sum()
            den = torch.sqrt((w_lat * p**2).sum() * (w_lat * g**2).sum())
            loss_dict[f"acc_{var}_{log_postfix}"] = (num / den).item()
        loss_dict["acc"] = np.mean(list(loss_dict.values()))
    return loss_dict


def crps(y_pred, y_true, transform, vars, lat, clim, log_postfix=None, anomaly=None):
    """
    Continuous Ranked Probability Score per variable and overall.

    Args:
        y_pred (Tensor[B, M, V, H, W]): Ensemble predictions.
        y_true (Tensor[B, V, H, W])
    """
    if transform is not None:
        y_pred = transform(y_pred)
        y_true = transform(y_true)

    crps_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            yp = y_pred[:, :, i].cpu().numpy()
            yt = y_true[:, i].cpu().numpy()
            da_true = xr.DataArray(yt, dims=["batch", "lat", "lon"])
            da_pred = xr.DataArray(yp, dims=["batch", "member", "lat", "lon"])
            crps_var = xs.crps_ensemble(da_true, da_pred).item()
            crps_dict[f"crps_{var}_{log_postfix}"] = crps_var

        # overall CRPS
        ytp = y_pred.cpu().numpy()
        ytt = y_true.cpu().numpy()
        da_true_all = xr.DataArray(ytt, dims=["batch", "var", "lat", "lon"])
        da_pred_all = xr.DataArray(ytp, dims=["batch", "member", "var", "lat", "lon"])
        crps_dict["crps"] = xs.crps_ensemble(da_true_all, da_pred_all).item()
    return crps_dict


def remove_nans(pred, gt):
    """
    Remove NaNs and infinities from two 1D tensors.

    Returns filtered (pred, gt).
    """
    m = ~torch.isnan(pred) & ~torch.isinf(pred) & ~torch.isnan(gt) & ~torch.isinf(gt)
    return pred[m], gt[m]


def pearson(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    Pearson correlation at specified forecast steps.

    Args:
        pred, y (Tensor[B, T, V, H, W])
        log_steps (list[int]), log_days (list[int])
    """
    pred_t = transform(pred)
    y_t = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                p = pred_t[:, step - 1, i].reshape(-1)
                g = y_t[:, step - 1, i].reshape(-1)
                p, g = remove_nans(p, g)
                loss_dict[f"pearsonr_{var}_day_{day}"] = stats.pearsonr(p.cpu().numpy(), g.cpu().numpy())[0]
        loss_dict["pearsonr"] = np.mean(list(loss_dict.values()))
    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    Combined normalized RMS error and bias, weighted by latitude.

    Uses lat_weighted_nrmses and lat_weighted_nrmseg.
    """

    # Normalized error (space mean squared difference of time means)
    def nrmses():
        loss = {}
        pred_t = transform(pred)
        y_t = transform(y)
        w = torch.from_numpy(np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean()).view(1, -1, 1).to(pred.device)
        for i, var in enumerate(vars):
            pr = pred_t[:, i]
            gt = y_t[:, i]
            mse = ((pr.mean(dim=0) - gt.mean(dim=0)) ** 2 * w).mean()
            loss[f"w_nrmses_{var}"] = torch.sqrt(mse) / clim
        return loss

    # Normalized error of ensemble-mean bias
    def nrmseg():
        loss = {}
        pred_t = transform(pred)
        y_t = transform(y)
        w = torch.from_numpy(np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean()).view(1, -1, 1).to(pred.device)
        for i, var in enumerate(vars):
            pr = (pred_t[:, i] * w).mean(dim=(-2, -1))
            gt = (y_t[:, i] * w).mean(dim=(-2, -1))
            loss[f"w_nrmseg_{var}"] = torch.sqrt(((pr - gt) ** 2).mean()) / clim
        return loss

    e1 = nrmses()
    e2 = nrmseg()
    combined = {k: e1[k] + 5 * e2[k] for k in e1}
    combined.update(e1)
    combined.update(e2)
    return combined


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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
        self.static_vars = ["lsm", "slt", "orog"]

        # 影响损失函数
        # level_weight = [lvl / sum(self.levels) for lvl in self.levels]
        # self.weight = level_weight

        # Load static constants (land mask, soil type, topography)
        # self.constants = self.load_constants(os.path.join(root_dir, "constants"))

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
        self.backbone = PEDadd_UViT(embed_dim=embed_dim, depth=decoder_depth, qk_scale=True, img_size=img_size, window_size=window_size)

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and layer weights."""
        pos = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier/truncated-normal init for Linear & LayerNorm."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

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
        把你的时间 times 转成模型需要的 hours (总小时数)
        Args:
            times: list / tensor / array of datetime 或 pd.Timestamp
        Returns:
            hours: torch.Tensor (int64 类型)
        """
        base_time = pd.Timestamp(start_date)

        # 把每个时间转成和基准时间相差的小时数
        hours = []
        for t in times:
            dt = pd.Timestamp(t)
            delta_hours = (dt - base_time) // pd.Timedelta("1h")  # 取整小时
            hours.append(delta_hours)

        return torch.tensor(hours, dtype=torch.int64)

    def forward_step(self, input, const=None, times=None, step=None, timestamp=None):
        """
        这个方法是为了兼容 XMetAI 框架的单步预测逻辑。
        框架可能会调用这个函数来进行自回归滚动预测。
        """
        b, t, v, h, w = input.size()

        # XMetAI 框架的数据集可能没有提供 anomaly，我们先用全 0 代替，保证模型能跑
        anomaly = torch.zeros_like(input)

        # 我们随机生成输入，将来会改
        H, W = h, w
        # batch中每个样本的绝对时间
        hours = self._times_to_hours(times).to(self.device)
        # 预测 6 小时，即 21600 分钟
        lead_time = timedelta(hours=6)
        # 固定值，数据集真实区域
        lats = torch.linspace(-90, 90, H).view(-1, 1).repeat(1, W).to(self.device)
        lons = torch.linspace(0, 360, W).view(1, -1).repeat(H, 1).to(self.device)

        # 计算损失需要
        region_info = {"min_h": 0, "max_h": H - 1, "min_w": 0, "max_w": W - 1}
        # 一个常量提前弄好
        const = torch.cat([const, const[-1:]], dim=0)
        # 调用你原本的 encoder 前向传播
        surf_pred, atmo_pred = self.forward_encoder(input, anomaly, hours, self.default_vars, lats, lons, region_info, lead_time, const=const)

        # 拼接输出，恢复成框架期望的 (b, 1, v, h, w) 形状
        # 注意：这里需要根据你的变量顺序把 surf 和 atmo 拼成和 input 一样的顺序
        # 为了最小改动能跑通，这里我们强制 reshape 并拼凑
        preds = torch.cat([surf_pred, atmo_pred.reshape(b, -1, h, w)], dim=1)

        # 确保输出和输入变量数一致，补齐或者裁剪
        if preds.shape[1] > v:
            preds = preds[:, :v]
        elif preds.shape[1] < v:
            pad = torch.zeros(b, v - preds.shape[1], h, w, device=preds.device)
            preds = torch.cat([preds, pad], dim=1)

        return preds.unsqueeze(1)  # 返回形状: (B, 1, V, H, W)

    def forward(self, data):
        """
        框架标准入口，统一接收 data 字典。
        """
        # 测试/推理模式
        if not self.training:
            return self.test(data)

        # 训练模式，预处理数据
        data = self.process_data(data)
        input = data["input"]  # shape: (B, in_frames, C, H, W)
        targets = data["targets"]  # shape: (B, out_frames, C, H, W)
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        # --- 空间尺寸变换逻辑
        target_size = tuple(int(x) for x in self.img_size)  # 强制转数字

        # 1. 处理 input: 合并 B 和 F 维度进行插值
        B, F_in, C, H, W = input.shape
        input = input.view(B * F_in, C, H, W)  # 变 4D
        input = F.interpolate(input, size=target_size, mode="bilinear", align_corners=False)
        input = input.view(B, F_in, C, *target_size)  # 还原 5D

        # 2. 处理 targets: 同样的操作
        B, F_out, C, H, W = targets.shape
        targets = targets.view(B * F_out, C, H, W)
        targets = F.interpolate(targets, size=target_size, mode="bilinear", align_corners=False)
        targets = targets.view(B, F_out, C, *target_size)

        # 3. 处理静态数据 const (假设是 3D 或 4D)
        if const is not None:
            # 如果是 (C, H, W)，需要临时补一个 Batch 维变成 4D 才能插值
            if const.dim() == 3:
                const = const.unsqueeze(0)  # (1, C, H, W)
                const = F.interpolate(const, size=target_size, mode="bilinear", align_corners=False)
                const = const.squeeze(0)  # (C, 128, 256)
            else:
                const = F.interpolate(const, size=target_size, mode="bilinear", align_corners=False)
        # ------------------------------------------

        # 直接进行单步预测
        output = self.forward_step(input, const, times=times)

        # 计算 Loss (由于我们要最少改动接入框架，我们直接用基类的 loss_dict 机制)
        # 假设只预测 1 步 (out_frames=1)
        target = targets[:, 0:1]

        # 简单的 MSE 损失，你也可以换成你自定义的 lat_weighted_mse
        loss_rec = torch.nn.functional.mse_loss(output, target)

        loss_dict = {"loss_rec": loss_rec}
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

    def predict(
        self,
        x,
        anomaly,
        hours,
        variables,
        out_vars,
        transform,
        lats,
        lons,
        region_info,
        max_predict_range,
    ):
        """
        Generate final NetCDF predictions. Saves via `np2nc_preds`.
        """
        surf_p, atmo_p = self.forward_encoder(x, anomaly, hours, variables, lats, lons, region_info)
        preds = x.clone()
        # Map upper & surface predictions back into full variable stack
        for i, var in enumerate(self.upper_variables):
            for j, lvl in enumerate(self.levels):
                idx = self.get_var_ids((f"{var}_{lvl}",), x.device)
                preds[:, idx] = atmo_p[:, j, i : i + 1]
        for i, var in enumerate(self.surface_variables):
            idx = self.get_var_ids((var,), x.device)
            preds[:, idx] = surf_p[:, i : i + 1]

        preds = transform(preds)
        out_ids = self.get_var_ids(tuple(out_vars), x.device)
        mi, ma = region_info["min_h"], region_info["max_h"]
        mj, mj = region_info["min_w"], region_info["max_w"]
        final = preds[:, out_ids, mi : ma + 1, mj : mj + 1].cpu().numpy()

        return final

    @torch.no_grad()
    def test_cascade(self, data):
        # 1. 提取基础数据与配置
        inputs = data["input"]  # (B, F_in, V, H_orig, W_orig)
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        target_size = (128, 256)
        B, F_in, V, H_orig, W_orig = inputs.shape

        # 2. 空间尺寸预处理：将初始输入和常量插值到 (128, 256)
        # 合并 B, F 维度进行插值
        inputs_interp = inputs.view(B * F_in, V, H_orig, W_orig)
        inputs_interp = F.interpolate(inputs_interp, size=target_size, mode="bilinear", align_corners=False)
        inputs_interp = inputs_interp.view(B, F_in, V, *target_size)

        if const is not None:
            if const.dim() == 3:  # (C, H, W)
                const = F.interpolate(const.unsqueeze(0), size=target_size, mode="bilinear").squeeze(0)
            else:  # (B, C, H, W)
                const = F.interpolate(const, size=target_size, mode="bilinear")

        # 3. 初始化输出容器 (按照原始尺寸返回，以便框架计算指标)
        total_step = max(self.test_frames)
        steps_to_save = np.array(self.test_frames) // self.out_frames
        # 这里的输出 channel 通常对应 V
        outputs = inputs.new_zeros(B, len(steps_to_save), V, H_orig, W_orig)

        # 4. 级联预测逻辑
        # step_range 通常为 [1]，如果存在多轮级联则累加取平均
        for num_step in self.step_range:
            curr_step_idx = 0
            # 复制一份插值后的输入用于本轮滚动
            rolling_inputs = inputs_interp.clone()

            for step in range(total_step):
                # 执行单步预测。注意参数顺序需对应你定义的 forward_step
                # forward_step(input, const, times, step, ...)
                output = self.forward_step(rolling_inputs, const, times=times, step=step)
                # output 形状: (B, 1, V, 128, 256)

                # 更新滚动窗口：去掉最旧的一帧，拼接最新预测的一帧
                # 假设模型是滑窗模式：(B, in_frames, V, H, W)
                rolling_inputs = torch.cat([rolling_inputs, output], dim=1)[:, 1:]

                # 如果当前步数在 test_frames 记录列表中
                if (step + 1) in steps_to_save:
                    # 将预测结果插值回原始尺寸 (H_orig, W_orig)
                    out_rescaled = F.interpolate(output.squeeze(1), size=(H_orig, W_orig), mode="bilinear", align_corners=False)  # (B, V, 128, 256)
                    outputs[:, curr_step_idx : curr_step_idx + 1] += out_rescaled.unsqueeze(1)
                    curr_step_idx += 1

        # 5. 如果有多个 step_range，取平均值
        outputs = outputs / len(self.step_range)

        return outputs

    @torch.no_grad()
    def export_onnx(self, inputs, step, hour, doy):
        normed_input = self.normalize(inputs)
        normed_input = normed_input.to(dtype=self.dtype)
        step = step.to(dtype=self.dtype)
        hour = hour.to(dtype=self.dtype)
        doy = doy.to(dtype=self.dtype)

        self.reset_input(normed_input)
        batch_size = inputs.shape[0]
        timestamp = {
            "step": step.repeat(batch_size),
            "hour": hour.repeat(batch_size),
            "doy": doy.repeat(batch_size),
        }

        height = normed_input.shape[-2]
        width = normed_input.shape[-1]
        graph = self._build_graph_from_shape(height, width, normed_input.device)
        if self.use_multiscale:
            factors = self.coarse_factors or [2]
            ms_graph = [self._build_multiscale_from_shape(height, width, factor, normed_input.device) for factor in factors]
            graph = (*graph, ms_graph)
        output = self.forward_step(normed_input, None, graph, step=0, timestamp=timestamp)
        output = self.inv_normalize(output)
        return output
