from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "all_gather",
    "all_gather_grad",
    "fcrps_loss_lite",
    "afcrps_loss_lite",
    "afcrps_loss",
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
    xs[rank] = x
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
) -> torch.Tensor:
    """Lite version of afCRPS (AIFS-ENS) loss to reduce memory usage.
    This mirrors :func:`afcrps_loss` but avoids constructing full member-by-member tensors.
    """
    outputs = all_gather_grad(output)  # (num_gpus, B, T, C, H, W)

    if dist.is_initialized() and outputs.shape[0] != dist.get_world_size():
        outputs = outputs.unsqueeze(0)

    members = outputs.shape[0]
    if members < 2:
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)

    epsilon = (1.0 - alpha) / members

    # First term: average absolute deviation from observation across members.
    abs_err_sum = torch.abs(outputs - target.unsqueeze(0)).sum(dim=0)

    # Second term: pair-wise absolute deviation between members using sorted gaps.
    sorted_outputs = outputs.sort(dim=0).values
    gaps = sorted_outputs[1:] - sorted_outputs[:-1]

    weight_idx = torch.arange(1, members, device=output.device, dtype=output.dtype)
    pair_weight = weight_idx * (members - weight_idx)
    # View needs to match (members-1, *([1]*(outputs.dim()-1)))
    # For outputs=M, B, T, C, H, W (dim=6), views to (M-1, 1, 1, 1, 1, 1)
    pair_weight = pair_weight.view(-1, *([1] * (outputs.dim() - 1)))
    pairwise_sum = 2.0 * torch.sum(gaps * pair_weight, dim=0)

    coef = 1.0 / (2.0 * members * (members - 1))
    crps = coef * (2.0 * (members - 1) * abs_err_sum - (1.0 - epsilon) * pairwise_sum)

    loss = torch.mean(crps * weight) if weight is not None else torch.mean(crps)
    return loss


def afcrps_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    alpha: float = 0.95,
) -> torch.Tensor:
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
