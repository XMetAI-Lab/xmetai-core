from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "linear_precip_weight",
    "precip_mm_from_normalized",
    "precip_loss",
]


def precip_mm_from_normalized(target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, precip_mm_scale: float) -> torch.Tensor:
    """Recover observed mm without an upper clamp on log precipitation."""
    target = target.float()
    log_tp = (target * std.to(target) + mean.to(target)).clamp_min(0.0)
    return torch.expm1(log_tp) * precip_mm_scale


def linear_precip_weight(obs_mm: torch.Tensor, slope: float) -> torch.Tensor:
    """Weight 1 below 0.1 mm, otherwise 1 + slope * p, without a ceiling."""
    return torch.where(obs_mm < 0.1, torch.ones_like(obs_mm), 1.0 + slope * obs_mm)


def occurrence_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    occurrence_scale: float = 10.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary wet/dry loss for precipitation channels."""
    logits = (output - float(threshold)) * float(occurrence_scale)
    wet_target = (target > float(threshold)).to(output)
    loss = F.binary_cross_entropy_with_logits(logits, wet_target, reduction="none")
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction: {reduction}.")


def spatial_distribution_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    occurrence_scale: float = 10.0,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Weight training-space errors by target excess above threshold, capped at 10.

    Inputs and threshold use the same training space, including standardization.
    Dry pixels receive no auxiliary weight; the global reconstruction loss
    remains responsible for their errors. Weights are not normalized per scene.
    """
    del occurrence_scale
    if output.dtype in (torch.float16, torch.bfloat16):
        output = output.float()
    if target.dtype in (torch.float16, torch.bfloat16):
        target = target.float()
    target = target.detach()
    intensity_weight = F.relu(target - float(threshold)).clamp_max(10.0)
    point_loss = torch.sqrt((output - target).square() + eps) - math.sqrt(eps)
    loss = point_loss * intensity_weight
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction: {reduction}.")


def precip_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    precip_indices: Sequence[int],
    threshold: float,
    coefficients: Optional[dict[str, float]] = None,
    occurrence_scale: float = 10.0,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Compute precipitation-specific auxiliary losses.

    Args:
        output: Prediction tensor ``(B, T, C, H, W)`` in training space.
        target: Target tensor ``(B, T, C, H, W)`` in training space.
        precip_indices: Channel indices to use as precipitation variables.
        threshold: Wet/dry threshold in the same space as ``output`` and ``target``.
        coefficients: Mapping for ``occurrence`` and ``distribution`` terms.
        occurrence_scale: Logit scale for the wet/dry soft boundary.
        eps: Smoothing constant inside the Charbonnier square root.
        reduction: ``"mean"`` or ``"sum"``.
    """
    if occurrence_scale <= 0:
        raise ValueError(f"occurrence_scale must be positive, got {occurrence_scale}.")
    if output.ndim != 5 or target.ndim != 5:
        raise ValueError(f"Expected output and target to be 5D tensors, got {output.shape} and {target.shape}.")
    if output.shape != target.shape:
        raise ValueError(f"Output shape {output.shape} must match target shape {target.shape}.")
    indices = list(precip_indices)
    if len(indices) == 0:
        return {}

    coeff = coefficients or {}
    pred = output[:, :, indices]
    obs = target[:, :, indices]

    loss_dict: dict[str, torch.Tensor] = {}
    c = coeff.get("occurrence", 0.0)
    if c > 0:
        loss_dict["loss_occurrence"] = c * occurrence_loss(
            pred,
            obs,
            threshold,
            occurrence_scale=occurrence_scale,
            reduction=reduction,
        )

    c = coeff.get("distribution", 0.0)
    if c > 0:
        loss_dict["loss_distribution"] = c * spatial_distribution_loss(
            pred,
            obs,
            threshold,
            occurrence_scale=occurrence_scale,
            eps=eps,
            reduction=reduction,
        )

    return loss_dict
