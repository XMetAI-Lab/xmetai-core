"""
Earth-aware Spectral Convolution Layers

Implements spectral convolution on the sphere using torch-harmonics SHT,
respecting Earth's spherical topology (longitude periodicity, latitude quadrature).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_harmonics import InverseRealSHT, RealSHT

__all__ = ["EarthSpectralConv2d"]


def _normalize_grid_size(
    size: Union[int, Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"Expected int or (nlat, nlon) tuple, got {size!r}")


class EarthSpectralConv2d(nn.Module):
    """
    Spherical spectral convolution with Earth-aware grid conventions.

    Applies Real SHT, a learnable spectral operator, and inverse Real SHT.
    Input tensors use shape ``(B, C, H, W)`` where ``H`` is latitude and ``W`` is longitude.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[int, Tuple[int, int]],
        out_size: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        bias: bool = False,
        grid: str = "equiangular",
        grid_out: Optional[str] = None,
        operator_type: str = "driscoll-healy",
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
    ):
        super().__init__()

        if groups <= 0:
            raise ValueError(f"groups must be a positive integer, got {groups}")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        in_nlat, in_nlon = _normalize_grid_size(img_size)
        if out_size is None:
            out_nlat, out_nlon = in_nlat, in_nlon
        else:
            out_nlat, out_nlon = _normalize_grid_size(out_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.in_nlat = in_nlat
        self.in_nlon = in_nlon
        self.out_nlat = out_nlat
        self.out_nlon = out_nlon
        self.grid = grid
        self.grid_out = grid_out or grid
        self.operator_type = operator_type

        self.sht = RealSHT(in_nlat, in_nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.isht = InverseRealSHT(out_nlat, out_nlon, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=self.grid_out)

        self.modes_lat = self.isht.lmax
        self.modes_lon = self.isht.mmax

        weight_shape: Tuple[int, ...]
        if operator_type == "diagonal":
            weight_shape = (groups, in_channels // groups, out_channels // groups, self.modes_lat, self.modes_lon)
            self.contract_func = "bgixy,gioxy->bgoxy"
        elif operator_type == "block-diagonal":
            weight_shape = (
                groups,
                in_channels // groups,
                out_channels // groups,
                self.modes_lat,
                self.modes_lon,
                self.modes_lon,
            )
            self.contract_func = "bgixy,gioxyn->bgoxn"
        elif operator_type == "driscoll-healy":
            weight_shape = (groups, in_channels // groups, out_channels // groups, self.modes_lat)
            self.contract_func = "bgixy,giox->bgoxy"
        else:
            raise ValueError(f"Unknown operator_type {operator_type!r}")

        scale = math.sqrt(2.0 / (in_channels // groups))
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.in_nlat or x.shape[-1] != self.in_nlon:
            raise ValueError(f"Expected input spatial size ({self.in_nlat}, {self.in_nlon}), " f"got ({x.shape[-2]}, {x.shape[-1]})")

        dtype = x.dtype
        batch_shape = x.shape[:-3]

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            coeffs = self.sht(x).contiguous()

        batch_size = math.prod(batch_shape) if batch_shape else 1
        channels = coeffs.shape[-3]
        coeffs = coeffs.reshape(batch_size, self.groups, channels // self.groups, self.modes_lat, self.modes_lon)
        out = torch.einsum(self.contract_func, coeffs, self.weight)
        out = out.reshape(*batch_shape, self.out_channels, self.modes_lat, self.modes_lon).contiguous()

        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.isht(out)

        if hasattr(self, "bias"):
            out = out + self.bias

        return out.to(dtype=dtype)
