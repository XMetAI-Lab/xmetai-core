from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "precip_loss",
]


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
    """Match normalized spatial precipitation mass for each sample/time/channel."""
    scale = float(occurrence_scale)
    pred_amount = F.softplus((output - float(threshold)) * scale) / scale
    target_amount = F.relu(target - float(threshold))

    pred_mass = pred_amount.sum(dim=(-2, -1), keepdim=True)
    target_mass = target_amount.sum(dim=(-2, -1), keepdim=True)
    pred_prob = pred_amount / pred_mass.clamp_min(eps)
    target_prob = target_amount / target_mass.clamp_min(eps)

    valid_target = (target_mass > eps).to(output).squeeze(-1).squeeze(-1)
    loss = 0.5 * (pred_prob - target_prob).abs().sum(dim=(-2, -1))
    loss = loss * valid_target
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
        eps: Numerical stability constant for distribution normalization.
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
