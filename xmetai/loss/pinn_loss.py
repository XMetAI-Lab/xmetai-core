"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-02-14 18:30:00
Description: Physics-informed neural network (PINN) loss functions for weather prediction.

Implements six physics-based soft constraints that penalize violations of
atmospheric governing equations in the model output. All losses operate on
normalized model output (B, T, C, H, W) and require a ChannelRegistry to
locate pressure-level variables by name.

Supported constraints:
    1. Mass continuity  — penalizes horizontal wind divergence
    2. Geostrophic balance — f v ≈ ∂Φ/∂x,  f u ≈ −∂Φ/∂y
    3. Hydrostatic balance — ∂Φ/∂p ≈ −R_d T / p
    4. Energy conservation — d/dt(c_p T + Φ) ≈ 0
    5. Moisture conservation — d/dt(q) + q ∇·V ≈ 0
    6. Thermal wind balance — ∂u/∂lnp ≈ (R_d / f) ∂T/∂y

Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

__all__ = [
    "ChannelRegistry",
    "mass_continuity_loss",
    "geostrophic_balance_loss",
    "hydrostatic_balance_loss",
    "energy_conservation_loss",
    "moisture_conservation_loss",
    "thermal_wind_loss",
    "pinn_loss",
    "potential_vorticity_loss",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665  # m/s²
EARTH_RADIUS = 6367500.0  # m  (mean of polar 6357 km and equatorial 6378 km)
OMEGA = 7.292e-5  # rad/s — Earth's angular velocity
R_D = 287.05  # J/(kg·K) — dry air gas constant
C_P = 1004.0  # J/(kg·K) — specific heat at constant pressure

# Default 13 pressure levels used in s2s datasets (hPa)
LEVELS_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
# Pressure-level variable short names (order must match channel layout)
PL_NAMES = ["z", "t", "u", "v", "q"]


# ---------------------------------------------------------------------------
# Channel Registry — auto-locates pressure-level channels from name list
# ---------------------------------------------------------------------------
class ChannelRegistry:
    """Map variable names to channel indices and provide pressure-level arrays.

    Args:
        channel_names: Ordered list of channel names as stored in the Zarr
            dataset (e.g. ``["z1000", "z925", ..., "q50", "msl", ...]``).
        levels: Pressure levels in hPa.  Defaults to :data:`LEVELS_13`.
        pl_names: Upper-air variable short names.  Defaults to ``["z", "t", "u", "v", "q"]``.
    """

    def __init__(
        self,
        channel_names: list[str],
        levels: list[int] | None = None,
        pl_names: list[str] | None = None,
    ):
        self.channel_names = list(channel_names)
        self.levels = levels or LEVELS_13
        self.pl_names = pl_names or PL_NAMES
        self.n_levels = len(self.levels)

        # Build index maps: var -> list of (level_hPa, channel_idx)
        self._var_indices: dict[str, list[tuple[int, int]]] = {}
        for var in self.pl_names:
            entries = []
            for lvl in self.levels:
                name = f"{var}{lvl}"
                if name in self.channel_names:
                    entries.append((lvl, self.channel_names.index(name)))
            self._var_indices[var] = entries

        # Validate that all expected channels are present
        for var in self.pl_names:
            if len(self._var_indices[var]) != self.n_levels:
                found = len(self._var_indices[var])
                logger.warning(f"ChannelRegistry: variable '{var}' has {found}/{self.n_levels} levels")

    def indices(self, var: str) -> list[int]:
        """Return channel indices for *var* across all pressure levels (ordered)."""
        return [idx for _, idx in self._var_indices[var]]

    def pressure_pa(self, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Pressure levels as a 1-D tensor in **Pascals** (shape ``[n_levels]``)."""
        return torch.tensor([lvl * 100.0 for lvl in self.levels], device=device, dtype=dtype)

    def has_var(self, var: str) -> bool:
        return var in self._var_indices and len(self._var_indices[var]) == self.n_levels


# ---------------------------------------------------------------------------
# Grid helpers — finite differences on a regular lat-lon sphere
# ---------------------------------------------------------------------------


def _make_lat_lon(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1-D lat / lon tensors in **radians**.

    Latitude runs from +90° → −90° (N→S, shape ``[H]``).
    Longitude runs from 0° → 360°−Δ (shape ``[W]``).
    """
    lat_deg = torch.linspace(90.0, -90.0, H, device=device, dtype=dtype)
    lon_deg = torch.linspace(0.0, 360.0, W + 1, device=device, dtype=dtype)[:-1]
    return lat_deg * (math.pi / 180.0), lon_deg * (math.pi / 180.0)


def _dfdx(f: torch.Tensor, lat_rad: torch.Tensor, dlon: float) -> torch.Tensor:
    """∂f/∂x on a sphere via centred finite-difference along the *last* dim (lon).

    ``f`` shape: ``(..., H, W)``.  Returns same shape.
    Uses periodic (circular) padding along longitude.
    """
    # Circular padding: wrap around in longitude (torch.cat avoids F.pad ndim constraints)
    f_pad = torch.cat([f[..., -1:], f, f[..., :1]], dim=-1)
    df_dlon = (f_pad[..., 2:] - f_pad[..., :-2]) / (2.0 * dlon)
    # cos(lat) factor: shape broadcastable to (..., H, 1)
    cos_lat = lat_rad.cos().clamp(min=1e-6)
    cos_lat = cos_lat.view(*([1] * (f.ndim - 2)), -1, 1)
    return df_dlon / (EARTH_RADIUS * cos_lat)


def _dfdy(f: torch.Tensor, dlat: float) -> torch.Tensor:
    """∂f/∂y on a sphere via centred finite-difference along dim −2 (lat).

    ``f`` shape: ``(..., H, W)``.  Returns ``(..., H-2, W)`` (interior points).
    """
    df_dlat = (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dlat)
    return df_dlat / EARTH_RADIUS


def _coriolis(lat_rad: torch.Tensor) -> torch.Tensor:
    """Coriolis parameter ``f = 2 Ω sin(φ)`` with latitude clamp to avoid div-by-zero at equator."""
    f = 2.0 * OMEGA * lat_rad.sin()
    # Clamp the magnitude while preserving sign to avoid div-by-zero near the equator.
    return f.sign() * f.abs().clamp(min=1e-5)


def _extract_var(
    field: torch.Tensor,
    registry: ChannelRegistry,
    var: str,
) -> torch.Tensor:
    """Slice channels for *var* from ``field`` of shape ``(B, T, C, H, W)``.

    Returns ``(B, T, n_levels, H, W)``.
    """
    idx = registry.indices(var)
    return field[:, :, idx]


# ---------------------------------------------------------------------------
# 1. Mass continuity loss — horizontal wind divergence
# ---------------------------------------------------------------------------


def mass_continuity_loss(
    output: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise non-zero horizontal wind divergence on each pressure level.

    .. math::
        \mathcal{L}_{\text{cont}} = \left\| \frac{\partial u}{\partial x}
        + \frac{\partial v}{\partial y} \right\|^2

    Args:
        output: Model prediction tensor ``(B, T, C, H, W)`` (normalised space).
        registry: Channel mapping.
        weight: Optional latitude weight ``(1, H, 1)`` or broadcastable.
        reduction: ``"mean"`` or ``"sum"``.
    """
    B, T, C, H, W = output.shape
    lat_rad, _ = _make_lat_lon(H, W, output.device, output.dtype)
    dlat = math.pi / (H - 1)
    dlon = 2.0 * math.pi / W

    u = _extract_var(output, registry, "u")  # (B,T,L,H,W)
    v = _extract_var(output, registry, "v")

    du_dx = _dfdx(u, lat_rad, dlon)  # (B,T,L,H,W)
    dv_dy = _dfdy(v, dlat)  # (B,T,L,H-2,W)

    # Trim du_dx to match interior latitude points
    du_dx = du_dx[..., 1:-1, :]

    divergence = du_dx + dv_dy  # (B,T,L,H-2,W)

    loss = divergence**2
    if weight is not None:
        # Trim weight to interior latitudes
        w = weight
        if w.ndim == 3 and w.shape[1] == H:
            w = w[:, 1:-1, :]
        loss = loss * w
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ---------------------------------------------------------------------------
# 2. Geostrophic balance loss
# ---------------------------------------------------------------------------


def geostrophic_balance_loss(
    output: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise departure from geostrophic wind balance.

    .. math::
        \mathcal{L}_{\text{geo}} = \left\| f v - \frac{1}{\rho}\frac{\partial \Phi}{\partial x} \right\|^2
        + \left\| f u + \frac{1}{\rho}\frac{\partial \Phi}{\partial y} \right\|^2

    For pressure-level data, geopotential :math:`\Phi = g \cdot z` where
    *z* is the geopotential height channel.

    Args:
        output: ``(B, T, C, H, W)``.
        registry: Channel mapping.
        weight: Optional latitude weight.
        reduction: ``"mean"`` or ``"sum"``.
    """
    B, T, C, H, W = output.shape
    lat_rad, _ = _make_lat_lon(H, W, output.device, output.dtype)
    dlat = math.pi / (H - 1)
    dlon = 2.0 * math.pi / W

    z = _extract_var(output, registry, "z")  # geopotential height
    u = _extract_var(output, registry, "u")
    v = _extract_var(output, registry, "v")

    phi = z * GRAVITY  # geopotential Φ = g·z

    dphi_dx = _dfdx(phi, lat_rad, dlon)  # (B,T,L,H,W)
    dphi_dy = _dfdy(phi, dlat)  # (B,T,L,H-2,W)

    f = _coriolis(lat_rad)  # (H,)
    f_full = f.view(1, 1, 1, H, 1)
    f_interior = f[1:-1].view(1, 1, 1, H - 2, 1)

    # Geostrophic balance equations:
    #   f·v = ∂Φ/∂x   →  residual_x = f·v - ∂Φ/∂x
    #   f·u = -∂Φ/∂y   →  residual_y = f·u + ∂Φ/∂y

    # For x-component, use full latitude grid
    res_x = f_full * v - dphi_dx

    # For y-component, trim to interior
    u_interior = u[..., 1:-1, :]
    res_y = f_interior * u_interior + dphi_dy

    loss = res_x**2
    if weight is not None:
        loss = loss * weight
    loss_x = loss.mean() if reduction == "mean" else loss.sum()

    loss_y = res_y**2
    if weight is not None:
        w = weight
        if w.ndim == 3 and w.shape[1] == H:
            w = w[:, 1:-1, :]
        loss_y = loss_y * w
    loss_y = loss_y.mean() if reduction == "mean" else loss_y.sum()

    return loss_x + loss_y


# ---------------------------------------------------------------------------
# 3. Hydrostatic balance loss
# ---------------------------------------------------------------------------


def hydrostatic_balance_loss(
    output: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise violation of the hydrostatic equation on pressure levels.

    .. math::
        \Delta\Phi \approx -R_d \, \bar{T} \, \Delta\ln p

    where :math:`\bar{T}` is the layer-mean temperature and the differences
    are taken between adjacent pressure levels.

    Args:
        output: ``(B, T, C, H, W)``.
        registry: Channel mapping.
        weight: Optional latitude weight.
        reduction: ``"mean"`` or ``"sum"``.
    """
    z = _extract_var(output, registry, "z")  # (B,T,L,H,W)
    t = _extract_var(output, registry, "t")  # temperature (K)

    phi = z * GRAVITY  # geopotential

    p = registry.pressure_pa(device=output.device, dtype=output.dtype)  # (L,)
    ln_p = p.log()

    # Finite differences between adjacent levels
    dphi = phi[:, :, 1:] - phi[:, :, :-1]  # (B,T,L-1,H,W)
    dln_p = (ln_p[1:] - ln_p[:-1]).view(1, 1, -1, 1, 1)  # (1,1,L-1,1,1)
    t_mean = 0.5 * (t[:, :, 1:] + t[:, :, :-1])  # layer-mean T

    # Hydrostatic: dΦ = -R_d T d(ln p)
    residual = dphi + R_D * t_mean * dln_p

    loss = residual**2
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ---------------------------------------------------------------------------
# 4. Energy conservation loss
# ---------------------------------------------------------------------------


def energy_conservation_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise change in dry static energy between output and target.

    Dry static energy per unit mass:

    .. math::
        s = c_p T + \Phi

    The loss penalises :math:`\|s_{\text{out}} - s_{\text{tgt}}\|^2`
    column-integrated across pressure levels.

    Args:
        output: ``(B, T, C, H, W)`` — model prediction.
        target: ``(B, T, C, H, W)`` — ground truth.
        registry: Channel mapping.
        weight: Optional latitude weight.
        reduction: ``"mean"`` or ``"sum"``.
    """
    t_out = _extract_var(output, registry, "t")
    z_out = _extract_var(output, registry, "z")
    t_tgt = _extract_var(target, registry, "t")
    z_tgt = _extract_var(target, registry, "z")

    s_out = C_P * t_out + GRAVITY * z_out  # (B,T,L,H,W)
    s_tgt = C_P * t_tgt + GRAVITY * z_tgt

    # Column-integrated difference (sum over levels)
    ds = (s_out - s_tgt).sum(dim=2)  # (B,T,H,W)

    loss = ds**2
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ---------------------------------------------------------------------------
# 5. Moisture conservation loss
# ---------------------------------------------------------------------------


def moisture_conservation_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise violation of moisture continuity.

    Simplified budget (ignoring sources/sinks):

    .. math::
        \frac{\partial q}{\partial t} + q \nabla \cdot \mathbf{V} \approx 0

    The temporal tendency is approximated as ``q_out − q_tgt`` (assuming
    unit time-step weighting — the absolute magnitude is absorbed by the
    loss coefficient in training).

    Args:
        output: ``(B, T, C, H, W)`` — prediction.
        target: ``(B, T, C, H, W)`` — ground truth (used for ∂q/∂t).
        registry: Channel mapping.
        weight: Optional latitude weight.
        reduction: ``"mean"`` or ``"sum"``.
    """
    B, T, C, H, W = output.shape
    lat_rad, _ = _make_lat_lon(H, W, output.device, output.dtype)
    dlat = math.pi / (H - 1)
    dlon = 2.0 * math.pi / W

    q_out = _extract_var(output, registry, "q")  # (B,T,L,H,W)
    q_tgt = _extract_var(target, registry, "q")
    u = _extract_var(output, registry, "u")
    v = _extract_var(output, registry, "v")

    dq_dt = q_out - q_tgt  # temporal tendency proxy

    # Horizontal divergence of wind
    du_dx = _dfdx(u, lat_rad, dlon)
    dv_dy = _dfdy(v, dlat)  # (B,T,L,H-2,W)

    # Trim all fields to interior latitudes to match dv_dy
    du_dx = du_dx[..., 1:-1, :]
    dq_dt = dq_dt[..., 1:-1, :]
    q_interior = q_out[..., 1:-1, :]

    div_V = du_dx + dv_dy
    residual = dq_dt + q_interior * div_V

    loss = residual**2
    if weight is not None:
        w = weight
        if w.ndim == 3 and w.shape[1] == H:
            w = w[:, 1:-1, :]
        loss = loss * w
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ---------------------------------------------------------------------------
# 6. Thermal wind balance loss
# ---------------------------------------------------------------------------


def thermal_wind_loss(
    output: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise departure from thermal-wind balance.

    .. math::
        \frac{\partial u_g}{\partial \ln p} = \frac{R_d}{f}
        \frac{\partial T}{\partial y}

    Computed as finite differences between adjacent pressure levels and
    interior latitude points.

    Args:
        output: ``(B, T, C, H, W)``.
        registry: Channel mapping.
        weight: Optional latitude weight.
        reduction: ``"mean"`` or ``"sum"``.
    """
    B, T, C, H, W = output.shape
    lat_rad, _ = _make_lat_lon(H, W, output.device, output.dtype)
    dlat = math.pi / (H - 1)

    u = _extract_var(output, registry, "u")  # (B,T,L,H,W)
    t = _extract_var(output, registry, "t")

    p = registry.pressure_pa(device=output.device, dtype=output.dtype)
    ln_p = p.log()  # (L,)

    # ∂u/∂(ln p) between adjacent levels
    du = u[:, :, 1:] - u[:, :, :-1]  # (B,T,L-1,H,W)
    dln_p = (ln_p[1:] - ln_p[:-1]).view(1, 1, -1, 1, 1)
    du_dlnp = du / dln_p

    # ∂T/∂y on interior latitudes, then average adjacent levels
    dt_dy = _dfdy(t, dlat)  # (B,T,L,H-2,W)
    dt_dy_mean = 0.5 * (dt_dy[:, :, 1:] + dt_dy[:, :, :-1])  # (B,T,L-1,H-2,W)

    # Trim du_dlnp to interior latitudes
    du_dlnp = du_dlnp[..., 1:-1, :]

    # Coriolis on interior latitudes
    f = _coriolis(lat_rad[1:-1])  # (H-2,)
    f = f.view(1, 1, 1, -1, 1)

    # Thermal-wind: ∂u/∂(ln p) = (R_d / f) ∂T/∂y
    residual = du_dlnp - (R_D / f) * dt_dy_mean

    loss = residual**2
    if weight is not None:
        w = weight
        if w.ndim == 3 and w.shape[1] == H:
            w = w[:, 1:-1, :]
        loss = loss * w
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ---------------------------------------------------------------------------
# Combined PINN loss — convenience wrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Potential Vorticity (PV) structure constraint
# ---------------------------------------------------------------------------
def potential_vorticity_loss(
    output: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Penalise deviation from potential vorticity (PV) structure.

    PV (Ertel) for pressure coordinates (approximate):
        PV = (f + ζ) * (∂θ/∂p)
    其中ζ为相对涡度，θ为位温。
    这里只做简化：
        ζ ≈ dv/dx - du/dy
        θ = T * (p0/p)^(R_d/c_p)

    Args:
        output: (B, T, C, H, W)
        registry: ChannelRegistry
        weight: Optional latitude weight
        reduction: "mean" or "sum"
    """
    B, T, C, H, W = output.shape
    lat_rad, _ = _make_lat_lon(H, W, output.device, output.dtype)
    dlat = math.pi / (H - 1)
    dlon = 2.0 * math.pi / W

    u = _extract_var(output, registry, "u")  # (B,T,L,H,W)
    v = _extract_var(output, registry, "v")
    t = _extract_var(output, registry, "t")
    p = registry.pressure_pa(device=output.device, dtype=output.dtype)  # (L,)
    p0 = 100000.0  # 1000 hPa in Pa

    # 1. 计算ζ (相对涡度)
    dv_dx = _dfdx(v, lat_rad, dlon)  # (B,T,L,H,W)
    du_dy = _dfdy(u, dlat)  # (B,T,L,H-2,W)
    # 对齐纬度
    dv_dx = dv_dx[..., 1:-1, :]
    zeta = dv_dx - du_dy  # (B,T,L,H-2,W)

    # 2. 计算θ (位温)
    # θ = T * (p0/p)^(R_d/c_p)
    exponent = R_D / C_P
    theta = t * (p0 / p.view(1, 1, -1, 1, 1)) ** exponent  # (B,T,L,H,W)
    # 对齐纬度
    theta = theta[..., 1:-1, :]

    # 3. 计算∂θ/∂p (用有限差分)
    # 先对p轴做差分
    dtheta_dp = (theta[:, :, 1:] - theta[:, :, :-1]) / (p[1:].view(1, 1, -1, 1, 1) - p[:-1].view(1, 1, -1, 1, 1))
    # 取中间层的ζ和f
    zeta_mid = 0.5 * (zeta[:, :, 1:] + zeta[:, :, :-1])  # (B,T,L-1,H-2,W)
    f = _coriolis(lat_rad[1:-1])  # (H-2,)
    f = f.view(1, 1, 1, -1, 1)

    # 4. PV = (f + ζ) * ∂θ/∂p
    pv = (f + zeta_mid) * dtheta_dp  # (B,T,L-1,H-2,W)

    # 5. 约束目标：PV结构平滑或接近某个climatology（这里只做平滑约束）
    # 例如：鼓励PV在空间上平滑
    dpv_dx = _dfdx(pv, lat_rad[1:-1], dlon)
    dpv_dy = _dfdy(pv, dlat)
    # 对齐纬度
    dpv_dx = dpv_dx[..., 1:-1, :]
    smoothness = dpv_dx**2 + dpv_dy**2

    loss = smoothness
    if weight is not None:
        w = weight
        if w.ndim == 3 and w.shape[1] == H:
            w = w[:, 2:-2, :]  # 因为纬度两次裁剪
        loss = loss * w
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


def pinn_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    registry: ChannelRegistry,
    weight: Optional[torch.Tensor] = None,
    coefficients: Optional[dict[str, float]] = None,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Compute all PINN losses and return a dictionary suitable for ``loss_dict``.

    Args:
        output: ``(B, T, C, H, W)`` — model prediction.
        target: ``(B, T, C, H, W)`` — ground truth.
        registry: :class:`ChannelRegistry` for channel lookup.
        weight: Optional latitude weight tensor.
        coefficients: Mapping from loss name to scalar coefficient.
            Defaults to ``1e-4`` for every term.
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        ``dict[str, Tensor]`` with keys:
        ``loss_mass``, ``loss_geo``, ``loss_hydro``, ``loss_energy``,
        ``loss_moisture``, ``loss_thermal``.
    """
    default_coeff = 1e-4
    coeff = coefficients or {}

    loss_dict: dict[str, torch.Tensor] = {}

    c = coeff.get("mass", default_coeff)
    if c > 0 and registry.has_var("u") and registry.has_var("v"):
        loss_dict["loss_mass"] = c * mass_continuity_loss(output, registry, weight, reduction)

    c = coeff.get("geo", default_coeff)
    if c > 0 and registry.has_var("z") and registry.has_var("u") and registry.has_var("v"):
        loss_dict["loss_geo"] = c * geostrophic_balance_loss(output, registry, weight, reduction)

    c = coeff.get("hydro", default_coeff)
    if c > 0 and registry.has_var("z") and registry.has_var("t"):
        loss_dict["loss_hydro"] = c * hydrostatic_balance_loss(output, registry, weight, reduction)

    c = coeff.get("energy", default_coeff)
    if c > 0 and registry.has_var("t") and registry.has_var("z"):
        loss_dict["loss_energy"] = c * energy_conservation_loss(output, target, registry, weight, reduction)

    c = coeff.get("moisture", default_coeff)
    if c > 0 and registry.has_var("q") and registry.has_var("u") and registry.has_var("v"):
        loss_dict["loss_moisture"] = c * moisture_conservation_loss(output, target, registry, weight, reduction)

    c = coeff.get("thermal", default_coeff)
    if c > 0 and registry.has_var("u") and registry.has_var("t"):
        loss_dict["loss_thermal"] = c * thermal_wind_loss(output, registry, weight, reduction)

    c = coeff.get("pv", 0.0)  # PV loss is opt-in (default disabled)
    if c > 0 and registry.has_var("u") and registry.has_var("v") and registry.has_var("t"):
        loss_dict["loss_pv"] = c * potential_vorticity_loss(output, registry, weight, reduction)

    return loss_dict
