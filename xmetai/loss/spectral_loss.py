"""Dimensionless, scale-separated power loss for global latitude-longitude fields."""

import math

import torch
from torch import nn

__all__ = ["SphericalBandPowerLoss"]


class SphericalBandPowerLoss(nn.Module):
    """Match spherical harmonic band energies with squared log power ratios.

    Inputs have shape (..., H, W), on a global, equiangular grid including both
    poles and excluding a duplicate longitude endpoint. Each leading element
    (sample, frame, channel) and each band receives equal weight. Bands are
    inclusive total spherical harmonic degrees, not planar FFT frequencies.

    The spatial mean is removed. For each field, the power floor is the target
    nonzero-degree energy times ``relative_floor``, clamped to ``eps``. Thus
    changing a channel's units does not change its loss except near ``eps``.
    The result is mean(log((pred_band + floor)/(target_band + floor))**2).
    Target statistics are detached. Spectral arithmetic remains float32 under
    autocast; cached quadrature/basis buffers do not enter checkpoints.
    """

    BANDS = ((1, 4), (5, 20), (21, 40), (41, 64), (65, 128))

    def __init__(
        self,
        image_size: tuple[int, int],
        relative_floor: float = 1e-6,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.bands = self.BANDS
        if len(self.image_size) != 2 or any(not isinstance(size, int) or size < 3 for size in self.image_size):
            raise ValueError("image_size must contain two integer dimensions >= 3")
        if not math.isfinite(relative_floor) or relative_floor <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("relative_floor and eps must be finite and positive")
        self.relative_floor = relative_floor
        self.eps = eps
        self.lmax = self.bands[-1][1]
        nlat, nlon = self.image_size
        n = nlat - 1
        if n % 2 or 2 * self.lmax >= n or self.lmax >= nlon // 2:
            raise ValueError("Need an odd latitude count with 2*lmax < H-1 and lmax < W//2")

        # Clenshaw-Curtis quadrature integrates in x=cos(colatitude).
        theta = torch.arange(nlat, dtype=torch.float64) * (math.pi / n)
        weights = torch.zeros(nlat, dtype=torch.float64)
        weights[[0, -1]] = 1 / (n * n - 1)
        interior = torch.ones(n - 1, dtype=torch.float64)
        for k in range(1, n // 2):
            interior -= 2 * torch.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
        interior -= torch.cos(n * theta[1:-1]) / (n * n - 1)
        weights[1:-1] = 2 * interior / n

        # Stable recurrence for orthonormal associated Legendre functions.
        basis = torch.zeros(self.lmax + 1, self.lmax + 1, nlat, dtype=torch.float64)
        basis[0, 0] = 1 / math.sqrt(4 * math.pi)
        x, sint = theta.cos(), theta.sin()
        for m in range(self.lmax + 1):
            if m:
                basis[m, m] = -math.sqrt((2 * m + 1) / (2 * m)) * sint * basis[m - 1, m - 1]
            if m < self.lmax:
                basis[m + 1, m] = math.sqrt(2 * m + 3) * x * basis[m, m]
            for degree in range(m + 2, self.lmax + 1):
                a = math.sqrt((4 * degree * degree - 1) / (degree * degree - m * m))
                b = math.sqrt(((degree - 1) ** 2 - m * m) / (4 * (degree - 1) ** 2 - 1))
                basis[degree, m] = a * (x * basis[degree - 1, m] - b * basis[degree - 2, m])
        multiplicity = torch.full((self.lmax + 1,), 2.0)
        multiplicity[0] = 1.0
        self.register_buffer("latitude_weights", (weights / 2).float(), persistent=False)
        self.register_buffer("weighted_basis", (basis * weights).float(), persistent=False)
        self.register_buffer("multiplicity", multiplicity, persistent=False)

    def _power(self, field: torch.Tensor) -> torch.Tensor:
        field = field.float().reshape(-1, *self.image_size)
        mean = (field.mean(dim=-1) * self.latitude_weights.float()).sum(dim=-1)
        field = field - mean[:, None, None]
        fft = torch.fft.rfft(field, dim=-1, norm="forward")[..., : self.lmax + 1]
        basis = self.weighted_basis.float()
        real = torch.einsum("lmh,nhm->nlm", basis, fft.real)
        imag = torch.einsum("lmh,nhm->nlm", basis, fft.imag)
        return ((real.square() + imag.square()) * self.multiplicity.float()).sum(dim=-1) * (2 * math.pi) ** 2

    def band_power(self, field: torch.Tensor) -> torch.Tensor:
        """Return band energies with shape (*field.shape[:-2], number_of_bands)."""
        if field.ndim < 2 or tuple(field.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected global fields on {self.image_size}, got {field.shape}")
        with torch.autocast(device_type=field.device.type, enabled=False):
            power = self._power(field)
            bands = torch.stack([power[:, low : high + 1].sum(-1) for low, high in self.bands], dim=-1)
        return bands.reshape(*field.shape[:-2], len(self.bands))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if output.shape != target.shape or output.ndim < 2 or tuple(output.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected matching (..., {self.image_size[0]}, {self.image_size[1]}) fields, got {output.shape}, {target.shape}")
        with torch.autocast(device_type=output.device.type, enabled=False):
            pred_power = self._power(output)
            target_power = self._power(target.detach())
            floor = (self.relative_floor * target_power[:, 1:].sum(dim=-1)).clamp_min(self.eps)
            losses = []
            for low, high in self.bands:
                pred_band = pred_power[:, low : high + 1].sum(dim=-1)
                target_band = target_power[:, low : high + 1].sum(dim=-1)
                losses.append(((pred_band + floor).log() - (target_band + floor).log()).square())
            return torch.stack(losses, dim=-1).mean()
