import torch

__all__ = ["field_dynamics_consistency_loss"]


def _frame_delta(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B,T,C,H,W), got shape={tuple(sequence.shape)}")
    if sequence.shape[1] < 2:
        raise ValueError(f"Expected at least two frames to compute frame deltas, got shape={tuple(sequence.shape)}")
    return sequence[:, 1:] - sequence[:, :-1]


def field_dynamics_consistency_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    input: torch.Tensor,
    target_input: torch.Tensor | None = None,
    group_slice_indices: list[list[int]] | tuple[list[int], ...] | None = None,
) -> torch.Tensor:
    """Match forecast frame deltas to teacher-forced target deltas in field space.

    This is the field-space counterpart of latent dynamics consistency:
    mean_g || Delta output_g - stop_grad(Delta target_g) ||_2^2.
    """
    if output.shape != target.shape:
        raise ValueError(f"output and target must have the same shape, got output={tuple(output.shape)}, target={tuple(target.shape)}")
    if output.ndim != 5:
        raise ValueError(f"Expected output/target as 5D tensors (B,T,C,H,W), got shape={tuple(output.shape)}")
    if input.ndim != 5:
        raise ValueError(f"Expected input as 5D tensor (B,T,C,H,W), got shape={tuple(input.shape)}")

    target_input = input if target_input is None else target_input
    if target_input.ndim != 5:
        raise ValueError(f"Expected target_input as 5D tensor (B,T,C,H,W), got shape={tuple(target_input.shape)}")
    if input.shape[0] != output.shape[0] or input.shape[2:] != output.shape[2:]:
        raise ValueError(f"input must match output batch/channel/spatial dimensions, got input={tuple(input.shape)}, output={tuple(output.shape)}")
    if target_input.shape[0] != target.shape[0] or target_input.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"target_input must match target batch/channel/spatial dimensions, got target_input={tuple(target_input.shape)}, target={tuple(target.shape)}"
        )

    prior_sequence = torch.cat([input[:, -1:], output], dim=1)
    post_sequence = torch.cat([target_input[:, -1:], target], dim=1)
    prior_delta = _frame_delta(prior_sequence)
    post_delta = _frame_delta(post_sequence).detach()
    delta_error = (prior_delta - post_delta).pow(2)

    if group_slice_indices is None:
        return delta_error.mean()

    group_losses = []
    for group_indices in group_slice_indices:
        if len(group_indices) == 0:
            continue
        group_losses.append(delta_error[:, :, group_indices].mean())
    if len(group_losses) == 0:
        raise ValueError("group_slice_indices must contain at least one non-empty group.")
    return torch.stack(group_losses).mean()
