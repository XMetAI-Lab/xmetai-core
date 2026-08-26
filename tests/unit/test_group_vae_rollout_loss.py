import numpy as np
import torch

from xmetai.models.group_vae import _BaseGroupVAE


def _build_tiny_group_vae(latent_dyn_weight: float, *, alpha: float = 1.0, rollout_max: int = 2) -> _BaseGroupVAE:
    return _BaseGroupVAE(
        group_slice_indices=[[0]],
        embed_dim=8,
        image_size=(4, 4),
        window_size=2,
        patch_size=1,
        down_times=0,
        encoder_depths=[1],
        decoder_depths=[1],
        num_heads=1,
        vt_num_heads=1,
        vt_num_layers=0,
        vt_window_size=2,
        in_names=["x"],
        in_chans=1,
        out_chans=1,
        in_frames=2,
        out_frames=1,
        test_frames=[1],
        test_chans=[0],
        test_names=["x"],
        buffers={
            "mean": np.zeros((1, 1, 1), dtype=np.float32),
            "std": np.ones((1, 1, 1), dtype=np.float32),
            "weight": np.ones((1, 1, 1), dtype=np.float32),
        },
        latent_dyn_weight=latent_dyn_weight,
        alpha=alpha,
        rollout_max=rollout_max,
        device="cpu",
    )


def _build_batch() -> dict[str, torch.Tensor | int]:
    return {
        "inputs": torch.randn(1, 2, 1, 4, 4),
        "targets": torch.randn(1, 3, 1, 4, 4),
        "times": torch.tensor([0]),
        "future_frames": 3,
    }


def test_rollout_field_dynamics_loss_supports_segmented_backward() -> None:
    torch.manual_seed(0)
    model = _build_tiny_group_vae(latent_dyn_weight=0.25).train()

    losses, state, done = model(_build_batch(), rollout_segment_size=1)

    assert not done
    assert "loss_latent_dyn_consistency" in losses
    assert torch.isfinite(losses["loss_latent_dyn_consistency"])
    losses["loss_latent_dyn_consistency"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    losses, _, done = model(None, rollout_state=state, rollout_segment_size=1)
    assert done
    assert "loss_latent_dyn_consistency" in losses


def test_rollout_field_dynamics_loss_is_disabled_by_default() -> None:
    torch.manual_seed(0)
    model = _build_tiny_group_vae(latent_dyn_weight=0.0).train()

    losses = model(_build_batch())

    assert "loss_latent_dyn_consistency" not in losses


def test_rollout_sampling_uses_physical_blocks_without_mutating_batch() -> None:
    torch.manual_seed(0)
    model = _build_tiny_group_vae(latent_dyn_weight=0.0, alpha=0.5, rollout_max=2).train()
    batch = _build_batch()
    original_targets = batch["targets"]

    _, state, done = model(batch, rollout_segment_size=3)
    expected_steps, expected_weights = model.build_loss_steps_and_weights(3, model.alpha, model.rollout_max)

    assert done
    assert state["loss_block_indices"] == set(expected_steps)
    torch.testing.assert_close(state["rolling_weights"], expected_weights)
    assert batch["targets"] is original_targets


def test_segmented_rollout_preserves_unsegmented_loss_values() -> None:
    torch.manual_seed(0)
    model = _build_tiny_group_vae(latent_dyn_weight=0.25, alpha=0.5, rollout_max=2).train()
    batch = _build_batch()
    segmented_batch = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    torch.manual_seed(1)
    unsegmented_losses = model(batch)

    torch.manual_seed(1)
    segmented_totals: dict[str, torch.Tensor] = {}
    state = None
    done = False
    while not done:
        segment_losses, state, done = model(
            segmented_batch if state is None else None,
            rollout_state=state,
            rollout_segment_size=1,
        )
        for name, value in segment_losses.items():
            segmented_totals[name] = segmented_totals.get(name, torch.zeros_like(value)) + value.detach()

    assert segmented_totals.keys() == unsegmented_losses.keys()
    for name, value in unsegmented_losses.items():
        torch.testing.assert_close(segmented_totals[name], value.detach())
