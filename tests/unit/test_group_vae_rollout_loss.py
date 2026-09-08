import numpy as np
import pytest
import torch

from xmetai.models.group_vae import SphericalBandPowerLoss, _BaseGroupVAE


def _build_tiny_group_vae(latent_dyn_weight: float, *, alpha: float = 1.0, rollout_max: int = 2, out_frames: int = 1, **kwargs) -> _BaseGroupVAE:
    return _BaseGroupVAE(
        group_slice_indices=[[0]],
        embed_dim=8,
        image_size=kwargs.pop("image_size", (4, 4)),
        window_size=2,
        patch_size=kwargs.pop("patch_size", 1),
        down_times=0,
        encoder_depths=[1],
        decoder_depths=[1],
        num_heads=1,
        vt_num_heads=1,
        vt_num_layers=0,
        vt_window_size=2,
        in_names=kwargs.pop("in_names", ["x"]),
        in_chans=1,
        out_chans=1,
        in_frames=out_frames + 1,
        out_frames=out_frames,
        test_frames=[1],
        test_chans=[0],
        test_names=["x"],
        buffers=kwargs.pop(
            "buffers",
            {
                "mean": np.zeros((1, 1, 1), dtype=np.float32),
                "std": np.ones((1, 1, 1), dtype=np.float32),
                "weight": np.ones((1, 1, 1), dtype=np.float32),
            },
        ),
        latent_dyn_weight=latent_dyn_weight,
        alpha=alpha,
        rollout_max=rollout_max,
        device="cpu",
        **kwargs,
    )


def _build_batch(in_frames: int = 2, future_frames: int = 3) -> dict[str, torch.Tensor | int]:
    return {
        "inputs": torch.randn(1, in_frames, 1, 4, 4),
        "targets": torch.randn(1, future_frames, 1, 4, 4),
        "times": torch.tensor([0]),
        "future_frames": future_frames,
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


def test_segmented_rollout_stops_after_last_selected_block() -> None:
    model = _build_tiny_group_vae(latent_dyn_weight=0.0, rollout_max=2, out_frames=2).train()

    _, state, done = model(_build_batch(in_frames=3, future_frames=5), rollout_segment_size=1)

    assert state["loss_block_indices"] == {0}
    assert done


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


@pytest.mark.parametrize("posterior_reconstruction", [0.0, 0.2])
def test_latent_regularization_is_rollout_normalized_and_posterior_gated(posterior_reconstruction: float) -> None:
    model = _build_tiny_group_vae(
        0.0,
        alpha=0.7,
        rollout_max=-1,
        loss_weight=(0.0, posterior_reconstruction, 1.0, 1.0, 0.0),
        kl_min_weight=1.0,
        kl_max_weight=1.0,
    ).train()
    prior = torch.tensor(2.0, requires_grad=True)
    posterior = torch.tensor(3.0, requires_grad=True)

    def constant_latent(module, args, output):
        amplitude = prior if module is model.vae_encoder_group[0] else posterior
        return torch.ones_like(output) * amplitude

    handles = [encoder.register_forward_hook(constant_latent) for encoder in (model.vae_encoder_group[0], model.posterior_encoder_group[0])]
    try:
        for steps in [1, 3, 5]:
            regularizer = model(_build_batch(future_frames=steps))["loss_latent_reg"]
            expected = 4.0 + (9.0 if posterior_reconstruction else 0.0)
            torch.testing.assert_close(regularizer, torch.tensor(expected))
        prior_gradient, posterior_gradient = torch.autograd.grad(regularizer, (prior, posterior), allow_unused=True)
        torch.testing.assert_close(prior_gradient, torch.tensor(4.0))
        if posterior_reconstruction:
            torch.testing.assert_close(posterior_gradient, torch.tensor(6.0))
        else:
            assert posterior_gradient is None
    finally:
        for handle in handles:
            handle.remove()


def test_spectral_loss_backpropagates_and_segments_sum_to_full_rollout(monkeypatch) -> None:
    monkeypatch.setattr(SphericalBandPowerLoss, "BANDS", ((1, 2), (3, 4)))
    model = _build_tiny_group_vae(
        0.0,
        alpha=0.7,
        rollout_max=-1,
        image_size=(17, 32),
        patch_size=2,
        spectral_loss_weight=0.1,
        spectral_loss_names=("x",),
    ).train()
    baseline = _build_tiny_group_vae(0.0, image_size=(17, 32), patch_size=2)
    model.load_state_dict(baseline.state_dict(), strict=True)
    batch = {"inputs": torch.randn(1, 2, 1, 17, 32), "targets": torch.randn(1, 3, 1, 17, 32), "times": torch.tensor([0]), "future_frames": 3}
    # Dropout would otherwise make the two trajectories different.
    model.eval()
    model.training = True
    full = model(batch)["loss_spectral"]
    full.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder_group.parameters())
    total = 0.0
    state = None
    done = False
    while not done:
        losses, state, done = model(batch if state is None else None, rollout_state=state, rollout_segment_size=1)
        total = total + losses["loss_spectral"].detach()
    torch.testing.assert_close(total, full.detach())


def test_disabled_posterior_is_not_executed_or_optimized() -> None:
    model = _build_tiny_group_vae(0.0, loss_weight=(0.0, 0.0, 1.0, 0.01, 1.0)).train()
    assert not any(p.requires_grad for p in model.posterior_encoder_group.parameters())

    def forbidden_call(module, args):
        raise AssertionError("Disabled posterior encoder was called")

    handles = [module.register_forward_pre_hook(forbidden_call) for module in model.posterior_encoder_group]
    decoder_calls = []
    handles.append(model.decoder_group[0].register_forward_pre_hook(lambda module, args: decoder_calls.append(1)))
    try:
        losses = model(_build_batch())
        assert set(losses) == {"loss_pred_rec", "loss_latent_reg"}
        # Two selected rollout steps: only prediction decoding, no reconstructions.
        # One skipped step still needs prediction to advance the trajectory.
        assert len(decoder_calls) == 3
        sum(losses.values()).backward()
        assert all(p.grad is None for p in model.posterior_encoder_group.parameters())
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_precip_rollout_threshold_gradients_and_segmented_totals(mixed_precision):
    torch.manual_seed(7)
    model = _build_tiny_group_vae(
        0.0,
        rollout_max=-1,
        alpha=0.7,
        in_names=["tp"],
        logid=[0],
        buffers={
            "mean": np.full((1, 1, 1), 0.3, dtype=np.float32),
            "std": np.full((1, 1, 1), 0.7, dtype=np.float32),
            "weight": np.ones((1, 1, 1), dtype=np.float32),
        },
        loss_weight=(0.0, 0.0, 1.0, 0.0, 0.0),
        precip_loss_enable=True,
        precip_names=("tp",),
        precip_loss_coeffs={"occurrence": 0.01, "distribution": 0.005},
        precip_threshold_mm=0.1,
        precip_mm_scale=1.0,
    ).train()
    assert model.precip_indices == (0,)
    assert model.precip_threshold == pytest.approx((np.log1p(0.1) - 0.3) / 0.7)
    batch = _build_batch(future_frames=3)
    batch["inputs"] = batch["inputs"].abs()
    batch["targets"] = batch["targets"].abs()
    batch["targets"][..., :2, :] = 0.0

    def clone_batch():
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    keys = ("precip_loss_occurrence", "precip_loss_distribution")
    torch.manual_seed(11)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        full = model(clone_batch())
    for key in keys:
        assert torch.isfinite(full[key]) and full[key] > 0
    sum(full[key] for key in keys).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.decoder_group.parameters())
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    totals = dict.fromkeys(keys, 0.0)
    state, done = None, False
    while not done:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
            losses, state, done = model(clone_batch() if state is None else None, rollout_state=state, rollout_segment_size=1)
        sum(losses[key] for key in keys).backward()
        for key in keys:
            totals[key] += losses[key].detach()
    for key in keys:
        torch.testing.assert_close(totals[key], full[key].detach())


@pytest.mark.parametrize(
    "weights, expected",
    [
        ((0.0, 0.0, 1.0, 0.0, 0.0), {"loss_pred_rec"}),
        ((0.0, 0.0, 0.0, 1.0, 1.0), {"loss_latent_reg"}),
        ((1.0, 1.0, 1.0, 1.0, 1.0), {"loss_pred_rec", "loss_vae_input", "loss_vae_target", "loss_latent_reg", "loss_latent_pred_rec"}),
    ],
)
def test_rollout_returns_only_enabled_losses(weights, expected):
    model = _build_tiny_group_vae(0.0, loss_weight=weights).train()
    assert set(model(_build_batch())) == expected
    losses, _, _ = model(_build_batch(), rollout_segment_size=1)
    assert set(losses) == expected
