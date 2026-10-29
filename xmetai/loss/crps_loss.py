import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

__all__ = [
    "crps_loss",
    "afcrps_loss",
    "multiscale_afcrps_loss",
]


def all_gather(x, group=None, dim=0):
    if not dist.is_initialized():
        return x.unsqueeze(dim)
    world_size = dist.get_world_size(group)
    x = x.contiguous()
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    with torch.no_grad():
        dist.all_gather(gathered, x.detach(), group=group)
    return torch.stack(gathered, dim)


def all_gather_grad(x, group=None):
    if not dist.is_initialized():
        return x.unsqueeze(0)
    rank = dist.get_rank(group)
    xs = all_gather(x, group, 0)
    # Only this rank's member retains its graph. Compensate for the subsequent
    # distributed parameter-gradient average without changing the loss value.
    members = xs.shape[0]
    xs[rank] = x.detach() + (x - x.detach()) * members
    return xs


def crps_loss(output: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sorted empirical CRPS, gathering one member per rank (BxTxCxHxW).

    All ranks must share the observation. A single member reduces to MAE.
    Weights broadcast over fields; reduction is mean(score * weight), not
    normalization by the sum of weights. All public scores use this convention.
    """
    return _distributed_crps(output, target, weight, 0.0, [])


def afcrps_loss(output: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, alpha: float = 0.95) -> torch.Tensor:
    """Sorted almost-fair CRPS (AIFS-ENS).

    For M >= 2, alpha=0 gives empirical CRPS and alpha=1 gives fair CRPS.
    A single member gives MAE for alpha=0 and differentiable zero otherwise.
    Input and weight conventions match :func:`crps_loss`.
    """
    return _distributed_crps(output, target, weight, alpha, [])


def multiscale_afcrps_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    alpha: float = 0.95,
    filter_scales: Optional[list[float]] = None,
) -> torch.Tensor:
    """Sum sorted afCRPS over Gaussian bands, without an extra full-field term.

    For [16, 8], score D16(x), D8(x)-D16(x), and x-D8(x), applying the same
    decomposition to the target. Bands have unit weight and are summed, not
    averaged. Empty scales give the full-field score. Scales are positive,
    distinct Gaussian standard deviations in grid spacings, sorted internally.
    Set alpha=0 for empirical CRPS or alpha=1 for fair CRPS in each band.

    Filtering requires a global regular lat-lon grid containing both poles and
    even longitudes without a duplicate endpoint. This separable grid-space
    Gaussian approximates, rather than implements, the MIR spherical filter.
    Dtype/autocast are preserved. One gather and a non-reentrant checkpoint
    avoid retaining multiscale activations, recomputing all fields in backward.
    Input and weight conventions match :func:`crps_loss`.
    """
    return _distributed_crps(output, target, weight, alpha, filter_scales)


def _distributed_crps(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: float,
    filter_scales: Optional[list[float]],
) -> torch.Tensor:
    scales = sorted(filter_scales, reverse=True) if filter_scales is not None else []
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("filter_scales must contain finite positive grid spacings.")
    if len(set(scales)) != len(scales):
        raise ValueError("filter_scales must contain distinct scales.")
    if scales and (output.ndim < 2 or output.shape[-2] < 2 or output.shape[-1] < 2 or output.shape[-1] % 2):
        raise ValueError("Filtering requires a global grid with H >= 2 and even W >= 2.")
    outputs = all_gather_grad(output)  # (num_gpus, B, T, C, H, W)

    if dist.is_initialized() and outputs.shape[0] != dist.get_world_size():
        outputs = outputs.unsqueeze(0)

    members = outputs.shape[0]
    if members < 2 and alpha != 0.0:
        return output.sum() * 0.0

    if not scales:
        return _afcrps_from_members(outputs, target, weight, alpha)

    target = target.broadcast_to(outputs.shape[1:])
    weight = weight.broadcast_to(outputs.shape[1:]) if weight is not None else None
    if torch.is_grad_enabled() and any(field is not None and field.requires_grad for field in (outputs, target, weight)):
        loss = checkpoint(_multiscale_afcrps_field, outputs, target, weight, alpha, scales, use_reentrant=False, preserve_rng_state=False)
        assert isinstance(loss, torch.Tensor)
        return loss
    return _multiscale_afcrps_field(outputs, target, weight, alpha, scales)


def _multiscale_afcrps_field(outputs: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor], alpha: float, scales: list[float]) -> torch.Tensor:
    """Score all spatial fields in a batch, sorting only across members."""
    previous_outputs = _gaussian_filter_global(outputs, scales[0])
    previous_target = _gaussian_filter_global(target, scales[0])
    loss = _afcrps_from_members(previous_outputs, previous_target, weight, alpha)
    for scale in scales[1:]:
        smooth_outputs = _gaussian_filter_global(outputs, scale)
        smooth_target = _gaussian_filter_global(target, scale)
        loss = loss + _afcrps_from_members(smooth_outputs - previous_outputs, smooth_target - previous_target, weight, alpha)
        previous_outputs, previous_target = smooth_outputs, smooth_target
    return loss + _afcrps_from_members(outputs - previous_outputs, target - previous_target, weight, alpha)


def _gaussian_filter_global(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a normalized, four-sigma Gaussian with periodic/pole continuation.

    Latitude continuation reflects across a pole and shifts longitude by half
    a turn. Modular indices support kernels larger than either grid dimension.
    Leading dimensions are independent fields, not convolution channels.
    """
    height, width = field.shape[-2:]
    radius = math.ceil(4 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    grid = field.reshape(-1, 1, height, width)
    longitude = torch.arange(-radius, width + radius, device=field.device).remainder(width)
    grid = torch.conv2d(grid.index_select(-1, longitude), kernel.view(1, 1, 1, -1))
    latitude = torch.arange(-radius, height + radius, device=field.device).remainder(2 * (height - 1))
    across_pole = latitude > height - 1
    latitude = torch.where(across_pole, 2 * (height - 1) - latitude, latitude)
    grid = grid.index_select(-2, latitude)
    grid = torch.where(across_pole.view(1, 1, -1, 1), grid.roll(width // 2, dims=-1), grid)
    return torch.conv2d(grid, kernel.view(1, 1, -1, 1)).reshape_as(field)


def _afcrps_from_members(outputs: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor], alpha: float) -> torch.Tensor:
    """Reduce an already gathered ensemble without further communication."""
    members = outputs.shape[0]
    if members == 1:
        crps = torch.abs(outputs[0] - target)
        return torch.mean(crps * weight) if weight is not None else torch.mean(crps)
    epsilon = (1.0 - alpha) / members

    # First term: average absolute deviation from observation across members.
    abs_err_sum = torch.abs(outputs - target.unsqueeze(0)).sum(dim=0)

    # The weighted adjacent gaps telescope to sum_i (2*i - M + 1) * x_sorted[i].
    # Contract directly: do not materialize gaps or an ensemble-sized weighted product.
    sorted_outputs = outputs.sort(dim=0).values
    pair_weight = 2 * torch.arange(members, device=outputs.device, dtype=outputs.dtype) - (members - 1)
    pairwise_sum = 2.0 * torch.einsum("m,m...->...", pair_weight, sorted_outputs)

    coef = 1.0 / (2.0 * members * (members - 1))
    crps = coef * (2.0 * (members - 1) * abs_err_sum - (1.0 - epsilon) * pairwise_sum)

    loss = torch.mean(crps * weight) if weight is not None else torch.mean(crps)
    return loss
