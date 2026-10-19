"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-10-11 08:15:54
Description: According to Subich et al. (arXiv:2501.19374, Eq. 6), this implements the spectrally adjusted mean squared error (AMSE) for spherical fields.
Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import math

import torch

from xmetai.loss.spectral_loss import SphericalBandPowerLoss


class SphericalAMSELoss(torch.nn.Module):
    """Compare global (..., H, W) fields using per-degree amplitude and coherence.

    The equiangular grid includes both poles and excludes duplicate longitude.
    All degrees from zero through lmax are included, without spatial centering.
    By default lmax is the largest degree supported by the existing quadrature.
    Unresolved degrees are not penalized. Sum over degrees / (4*pi) gives an
    area-mean squared-error scale; leading sample/frame/channel axes are averaged.
    Inputs should already be normalized across variables (including any log
    transform). Optional tensor weights multiply the loss directly: wavenumber
    weights broadcast against (..., C, lmax+1), channel weights against (..., C)
    after summing degrees. Omitted weights skip weighting; no masks are applied.

    For stable zero-energy gradients, eps is added to both powers and the cross
    spectrum, equivalent to augmenting both coefficient vectors with the same
    sqrt(eps) component. This preserves zero loss for identical fields and
    approaches Eq. 6 away from zero energy. Target coefficients are detached.
    """

    def __init__(self, image_size: tuple[int, int], lmax: int | None = None, eps: float = 1e-12) -> None:
        super().__init__()
        if len(image_size) != 2 or any(not isinstance(size, int) or size < 3 for size in image_size):
            raise ValueError("image_size must contain two integer dimensions >= 3")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if lmax is None:
            lmax = min((image_size[0] - 2) // 2, image_size[1] // 2 - 1)
        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError("lmax must be a nonnegative integer")
        self.transform = SphericalBandPowerLoss(image_size, bands=((0, lmax),))
        self.image_size = tuple(image_size)
        self.eps = eps

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, channel_weights: torch.Tensor | None = None, wavenumber_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if output.shape != target.shape or output.ndim < 2 or tuple(output.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected matching (..., {self.image_size[0]}, {self.image_size[1]}) fields, got {output.shape}, {target.shape}")
        with torch.autocast(device_type=output.device.type, enabled=False):
            pred_real, pred_imag = self.transform._coefficients(output)
            true_real, true_imag = self.transform._coefficients(target.detach())
        multiplicity = self.transform.multiplicity.float()
        pred_power = ((pred_real.square() + pred_imag.square()) * multiplicity).sum(-1) + self.eps
        true_power = ((true_real.square() + true_imag.square()) * multiplicity).sum(-1) + self.eps
        cross = ((pred_real * true_real + pred_imag * true_imag) * multiplicity).sum(-1) + self.eps
        pred_amplitude, true_amplitude = pred_power.sqrt(), true_power.sqrt()
        coherence = (cross / (pred_amplitude * true_amplitude)).clamp(-1, 1)
        loss = (pred_amplitude - true_amplitude).square() + 2 * torch.maximum(pred_power, true_power) * (1 - coherence)
        loss = loss.reshape(*output.shape[:-2], self.transform.lmax + 1) / (4 * math.pi)
        if wavenumber_weights is not None:
            loss = loss * wavenumber_weights
        loss = loss.sum(-1)
        if channel_weights is not None:
            if output.ndim < 3 or channel_weights.numel() != output.shape[-3]:
                raise ValueError("channel_weights must contain one value per channel")
            loss = loss * channel_weights.reshape(-1)
        return loss.mean()
