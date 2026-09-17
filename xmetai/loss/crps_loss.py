import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

__all__ = [
    "afcrps_loss_lite",
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


def fcrps_loss_lite(output: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """lite version of fire CRPS loss to reduce memory usage
    Args:
        output (torch.Tensor): predict member with same target/observation in dimension BxTxCxHxW.
        target (torch.Tensor): observation tensor in dimension BxTxCxHxW.
        weight (torch.Tensor, optional): weight for loss calculation. Defaults to None. Maybe latitude weights.
    """
    # output: (B, T, C, H, W)
    # all_gather_grad will stack along dim=0, yielding outputs shape: (num_gpus, B, T, C, H, W)
    outputs = all_gather_grad(output)

    # Check if gathered correctly
    if dist.is_initialized() and outputs.shape[0] != dist.get_world_size():
        # Fallback if somehow not running in fully initialized distributed mode
        outputs = outputs.unsqueeze(0)

    members = outputs.shape[0]
    if members < 2:
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

    weight_idx = torch.arange(1, members, device=output.device, dtype=output.dtype)
    pred_weight = (weight_idx * (members - weight_idx)).view(-1, *([1] * (outputs.dim() - 1)))

    sorted_outputs = outputs.sort(dim=0).values
    diff = torch.sum((sorted_outputs[1:] - sorted_outputs[:-1]) * pred_weight, dim=0) / (members * (members - 1))
    crps = torch.abs(outputs - target.unsqueeze(0)).mean(0) - diff
    if weight is not None:
        loss = torch.mean(crps * weight)
    else:
        loss = torch.mean(crps)
    return loss


def afcrps_loss_lite(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    alpha: float = 0.95,
    filter_scales: list[float] = [],
) -> torch.Tensor:
    """Lite version of afCRPS (AIFS-ENS) loss to reduce memory usage.
    This mirrors :func:`afcrps_loss` but avoids constructing full member-by-member tensors.

    ``filter_scales`` specifies Gaussian standard deviations in grid spacings.
    An empty list preserves the original full-field score. Positive,
    distinct scales are sorted from coarse to fine. For [16, 8], the equally
    weighted scored fields are D16(x), D8(x)-D16(x), and x-D8(x), with the same
    decomposition applied to the target. No full-field term is added.

    Filtering retains the grid resolution and follows input dtype and autocast. It assumes
    a global regular latitude-longitude grid containing both poles and an even
    number of longitudes without a duplicate endpoint. The separable grid-space
    Gaussian is an approximation, not the paper's MIR spherical-distance filter.

    The ensemble is gathered once in its original dtype. All batch/time/channel fields
    are filtered together without forced upcasting. A single non-reentrant checkpoint
    recomputes filtering and scoring during backward rather than retaining multiscale
    activations. Recomputed intermediates span the full batch of fields.
    """
    scales = sorted(filter_scales, reverse=True)
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
    if members < 2:
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

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


def afcrps_loss(output: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, alpha: float = 0.95) -> torch.Tensor:
    """almost fire CRPS loss according to AIFS-ENS(2412.15832)
    Args:
        output (torch.Tensor): predict member with same target/observation in dimension BxTxCxHxW.
        target (torch.Tensor): observation tensor in dimension BxTxCxHxW.
        weight (torch.Tensor, optional): weight for loss calculation. Defaults to None. Maybe latitude weights.
        alpha (float, optional): inflation coefficient. Defaults to 0.95.
    """
    outputs = all_gather_grad(output)  # (members, B, T, C, H, W)
    if dist.is_initialized() and outputs.shape[0] != dist.get_world_size():
        outputs = outputs.unsqueeze(0)

    members = outputs.shape[0]
    if members < 2:
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

    epsilon = (1.0 - alpha) / members
    # var shape: (members, members, B, T, C, H, W)
    var = torch.abs(outputs.unsqueeze(dim=0) - outputs.unsqueeze(dim=1))

    diag = torch.eye(members, dtype=torch.bool, device=output.device)
    # diag shape expanding for (members, members, 1, ...)
    for _ in range(outputs.dim() - 1):
        diag = diag.unsqueeze(dim=-1)

    # target unsqueeze for member dim: (1, B, ...)
    # then outputs - target -> (members, B, ...)
    # expand over new member dim -> (members, members, B, ...)
    err_abs = torch.abs(outputs - target.unsqueeze(0))
    err_r = err_abs.unsqueeze(1).expand(members, members, *outputs.shape[1:])

    mem_err = err_r * ~diag
    mem_err_transpose = mem_err.transpose(0, 1)

    coef = 1.0 / (2.0 * members * (members - 1))
    crps = coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(0, 1))

    if weight is not None:
        loss = torch.mean(crps * weight)
    else:
        loss = torch.mean(crps)
    return loss
