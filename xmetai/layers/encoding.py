from __future__ import annotations

import math
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import to_2tuple

__all__ = [
    "FourierExpansion",
    "StepEncoding",
    "TimeEncoding",
    "lat_lon_meshgrid",
    "patch_root_area",
    "pos_scale_enc",
    "pos_scale_enc_grid",
]


# ==============================================================================
# Continuous Coordinate Encoding
# ==============================================================================


class FourierExpansion(nn.Module):
    """Encode a continuous scalar into a high-dimensional Fourier feature vector.

    Maps each input value :math:`x` to a vector of :math:`d` Fourier features:

    .. math::

        \\text{FourierExpansion}(x) = \\big[\\sin(2\\pi x / \\lambda_1), \\dots,
        \\sin(2\\pi x / \\lambda_{d/2}), \\cos(2\\pi x / \\lambda_1), \\dots,
        \\cos(2\\pi x / \\lambda_{d/2})\\big]

    where wavelengths :math:`\\lambda_i` are **logarithmically spaced** between
    ``lower`` and ``upper``. This is more principled for geo-spatial coordinates
    than arithmetic spacing because it captures both fine-grained local patterns
    (small λ) and large-scale global patterns (large λ).

    Computation is performed in ``float64`` internally and cast back to
    ``float32`` to avoid numerical precision issues with very small or very
    large wavelengths.

    This pattern is used extensively in:
    - Microsoft Aurora (spherical coordinate encoding for weather)
    - NeRF / SIREN (Fourier feature networks for coordinate-based MLPs)
    - Transformer position encoding for continuous domains

    Parameters
    ----------
    lower : float
        Minimum wavelength in the same units as the input.
    upper : float
        Maximum wavelength in the same units as the input.
    assert_range : bool, default True
        If ``True``, assert that all input values lie within
        ``[lower, upper]`` (zeros are always allowed). Disable for inputs
        that naturally fall outside this range (e.g., absolute time in hours).

    Shapes
    ------
    - Input ``x``: :math:`(\\dots, n)` — any leading batch dimensions,
      last dimension is the coordinate(s) to encode.
    - Input ``d``: scalar (int), must be even.
    - Output: :math:`(\\dots, n, d)` — Fourier features appended as a new
      last dimension.

    Examples
    --------
    >>> # Encode latitudes: values in [-90, 90]
    >>> lat_encoder = FourierExpansion(lower=0.01, upper=180)
    >>> lats = torch.linspace(-90, 90, 128)          # (128,)
    >>> features = lat_encoder(lats, d=32)            # (128, 32)

    >>> # Encode pressure levels: values in [50, 1000] hPa
    >>> level_encoder = FourierExpansion(lower=0.01, upper=1e5)
    >>> levels = torch.tensor([50., 100., 200., 500., 850., 1000.])
    >>> feats = level_encoder(levels, d=64)           # (6, 64)

    >>> # Encode lead times: 0 to ~500 hours, minutes resolution
    >>> time_encoder = FourierExpansion(lower=1/60, upper=24*7*3)
    >>> lead_hours = torch.tensor([24., 48., 72., 168.])
    >>> feats = time_encoder(lead_hours, d=24)        # (4, 24)

    Notes
    -----
    - Wavelengths are computed as ``torch.logspace(log10(lower), log10(upper), d//2)``.
    - The input range ``[lower, upper]`` should span the expected minimum to
      maximum values. Choose ``lower`` based on the smallest resolving power
      needed, not necessarily the smallest input value. For example, for
      latitudes use ``lower=0.01`` (0.01 degrees ≈ 1 km resolution) rather
      than ``lower=-90``.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Encode ``x`` into ``d`` Fourier features.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape ``(..., n)``.
        d : int
            Number of output Fourier features. Must be even.

        Returns
        -------
        torch.Tensor
            Fourier features of shape ``(..., n, d)``, dtype ``float32``.

        Raises
        ------
        AssertionError
            If ``assert_range=True`` and any element of ``x`` falls outside
            ``[lower, upper]`` (zeros are allowed).
        ValueError
            If ``d`` is not a multiple of 2.
        """
        if self.assert_range:
            in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
            in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
            if not in_range_or_zero:
                raise AssertionError(f"Input values must lie within [{self.lower}, {self.upper}]. " f"Set assert_range=False to disable this check.")

        if d % 2 != 0:
            raise ValueError(f"d must be even, got {d}.")

        # Use float64 internally for numerical stability with extreme wavelengths.
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


# ==============================================================================
# Discrete Step / Time Encoding
# ==============================================================================


class StepEncoding(nn.Module):
    def __init__(self, dim, max_period=1e4):
        super().__init__()
        self.dim_half = dim // 2
        self.max_period = torch.tensor(max_period, dtype=torch.float32)

    def forward(self, x):
        emb = torch.log(self.max_period) / (self.dim_half - 1)
        emb = torch.exp(torch.arange(self.dim_half, dtype=torch.float32) * -emb)
        emb = rearrange(x.float(), "i -> i 1") * rearrange(emb.to(x.device), "j -> 1 j")
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.to(x)


class TimeEncoding(nn.Module):
    def __init__(self, freq=6):
        super().__init__()
        self.freq = freq

    def forward(self, i, times):
        bs = times.shape[0]
        hours = np.array([pd.Timedelta(hours=t * self.freq) for t in [i - 1, i, i + 1]])
        times = times[:, None] + hours[None]
        times = [pd.Period(t, "H") for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        emb = torch.from_numpy(np.array(times, dtype=np.float32))
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.reshape(bs, -1)


# ==============================================================================
# Spherical Coordinate Utilities
# ==============================================================================


def patch_root_area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:
    """Compute the square root of the area of a rectangular lat-lon patch on a sphere.

    Units: km.  The square root is taken so that values scale linearly with
    resolution — important when the output is fed into a position encoder.

    Uses the spherical rectangle area formula:

    .. math::

        \\text{area} = R^2 \\pi (\\sin\\phi_1 - \\sin\\phi_2) (\\theta_1 - \\theta_2)

    where :math:`R = 6371` km, :math:`\\phi` are latitudes, and
    :math:`\\theta` are longitudes in **radians**.

    Parameters
    ----------
    lat_min : torch.Tensor
        Minimum latitudes of patches (degrees).  Must be ``< lat_max``.
    lon_min : torch.Tensor
        Minimum longitudes of patches (degrees).  Must be ``>= 0``.
    lat_max : torch.Tensor
        Maximum latitudes of patches (degrees).  Must be ``<= 90``.
    lon_max : torch.Tensor
        Maximum longitudes of patches (degrees).  Must be ``<= 360``.

    Returns
    -------
    torch.Tensor
        Square root of patch area in km.  Same shape as the inputs.

    References
    ----------
    - `Sphere Grid Area <https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/>`_
    """
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = 6371**2 * torch.pi * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min))) * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))
    assert (area > 0.0).all()
    return torch.sqrt(area)


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Construct a 2-channel (lat, lon) meshgrid from coordinate vectors.

    Uses ``torch.meshgrid(..., indexing="xy")`` and then permutes axes so the
    returned grid is shaped ``(2, H, W)`` with lat varying along ``H`` and
    lon along ``W``::

        >>> lat = torch.tensor([1., 2., 3.])
        >>> lon = torch.tensor([4., 5., 6.])
        >>> grid = lat_lon_meshgrid(lat, lon)
        >>> grid.shape
        torch.Size([2, 3, 3])
        >>> grid[0]  # latitudes (repeated across columns)
        tensor([[1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]])
        >>> grid[1]  # longitudes (repeated across rows)
        tensor([[4., 4., 4.],
                [5., 5., 5.],
                [6., 6., 6.]])

    Parameters
    ----------
    lat : torch.Tensor
        Latitude vector of shape ``(H,)``.
    lon : torch.Tensor
        Longitude vector of shape ``(W,)``.

    Returns
    -------
    torch.Tensor
        Meshgrid of shape ``(2, H, W)`` where channel 0 = latitudes and
        channel 1 = longitudes.
    """
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
    """Position + scale encoding for a batched latitude-longitude grid.

    Pools the grid into patches, then applies ``pos_expansion`` to the patch
    mid-points (latitudes and longitudes separately, each using half of
    ``encode_dim``) and ``scale_expansion`` to the patch root-areas.

    Parameters
    ----------
    encode_dim : int
        Output encoding dimension.  Must be a multiple of 4 (half for lat,
        half for lon, split into sine / cosine).
    grid : torch.Tensor
        Latitude-longitude grid of shape ``(B, 2, H, W)``.
        ``grid[:, 0]`` = latitudes, ``grid[:, 1]`` = longitudes.
    patch_dims : tuple[int, int]
        Patch dimensions ``(patch_h, patch_w)``.
    pos_expansion : FourierExpansion
        Fourier expansion for latitude / longitude coordinates.
    scale_expansion : FourierExpansion
        Fourier expansion for patch root-area.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Position encoding: ``(B, L, D)`` where ``L = H/patch_h * W/patch_w``.
        - Scale encoding: ``(B, L, D)``.

    Examples
    --------
    >>> pos_exp = FourierExpansion(0.01, 720)
    >>> scale_exp = FourierExpansion(4.8e-3, 5.1e8)
    >>> grid = torch.randn(2, 2, 128, 256)  # (B=2, C=2, H=128, W=256)
    >>> pe, se = pos_scale_enc_grid(64, grid, (4, 4), pos_exp, scale_exp)
    >>> pe.shape  # (2, 2048, 64)
    >>> se.shape  # (2, 2048, 64)
    """
    assert encode_dim % 4 == 0, f"encode_dim must be a multiple of 4, got {encode_dim}."
    assert grid.dim() == 4, f"grid must be 4-D (B, 2, H, W), got {grid.dim()}D."

    # Pooled patch mid-points.
    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    # Patch extents for area computation.
    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)  # (B, L, D)

    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)  # (B, L, D)

    return pos_encode, scale_encode


def pos_scale_enc(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: Union[int, List[int], Tuple[int, int]],
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around :func:`pos_scale_enc_grid`.

    Builds the meshgrid internally and strips the batch dimension from the
    results so the caller gets simple ``(L, D)`` tensors.

    Parameters
    ----------
    encode_dim : int
        Output encoding dimension (must be a multiple of 4).
    lat : torch.Tensor
        Latitudes.  Either 1-D ``(H,)`` or 2-D ``(H, W)``.
    lon : torch.Tensor
        Longitudes.  Either 1-D ``(W,)`` or 2-D ``(H, W)``.
    patch_dims : int or tuple[int, int]
        Patch dimensions.  A single ``int`` is expanded to a square patch.
    pos_expansion : FourierExpansion
        Fourier expansion for latitude / longitude values.
    scale_expansion : FourierExpansion
        Fourier expansion for patch root-area.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Position encoding: ``(L, D)``.
        - Scale encoding: ``(L, D)``.

    Examples
    --------
    >>> pos_exp = FourierExpansion(0.01, 720)
    >>> scale_exp = FourierExpansion(4.8e-3, 5.1e8)
    >>> lats = torch.linspace(-90, 90, 128)
    >>> lons = torch.linspace(0, 360, 256)
    >>> pe, se = pos_scale_enc(64, lats, lons, 4, pos_exp, scale_exp)
    >>> pe.shape  # (2048, 64)
    >>> se.shape  # (2048, 64)
    """
    if lat.dim() == 1 and lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == 2 and lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(f"lat and lon must both be 1-D or both 2-D, got {lat.dim()}D and {lon.dim()}D.")

    grid = grid[None]  # add batch dim B=1
    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )
    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)
