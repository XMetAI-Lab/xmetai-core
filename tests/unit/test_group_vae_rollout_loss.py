import numpy as np
import pandas as pd
import pytest
import torch

from xmetai.models.group_vae import _BaseGroupVAE


def _build_tiny_group_vae(latent_dyn_weight: float, *, alpha: float = 1.0, rollout_max: int = 2, out_frames: int = 1, **kwargs) -> _BaseGroupVAE:
    return _BaseGroupVAE(
        group_slice_indices=[[0]],
        embed_dim=8,
        image_size=kwargs.pop("image_size", (4, 4)),
        window_size=kwargs.pop("window_size", 2),
        patch_size=kwargs.pop("patch_size", 1),
        down_times=kwargs.pop("down_times", 0),
        encoder_depths=kwargs.pop("encoder_depths", [1]),
        decoder_depths=kwargs.pop("decoder_depths", [1]),
        num_heads=1,
        vt_num_heads=1,
        vt_num_layers=kwargs.pop("vt_num_layers", 0),
        vt_window_size=kwargs.pop("vt_window_size", 2),
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


@pytest.mark.parametrize("patch_shortcut", [False, True])
def test_patch_shortcut_encoding_and_decoder_routing(patch_shortcut):
    model = _build_tiny_group_vae(0.0, patch_decoder_shortcut=patch_shortcut).eval()
    batch = _build_batch()
    batch["times"] = pd.to_datetime(["2024-01-01"])
    x = batch["inputs"]
    timestamp = model.get_timestamp(x, batch["times"], model.start_frames)
    encoded = model._encode_all_groups(x, timestamp)
    assert isinstance(encoded, tuple if patch_shortcut else list)
    patches, received = [], []
    encoder_hook = model.vae_encoder_group[0].patch_embed.register_forward_hook(lambda module, args, output: patches.append(output))
    decoder_hook = model.decoder_group[0].register_forward_pre_hook(lambda module, args, kwargs: received.append(kwargs["patch_shortcut"]), with_kwargs=True)
    try:
        output, history = model._predict_next(x, batch["times"], model.start_frames)
    finally:
        encoder_hook.remove()
        decoder_hook.remove()
    assert received[0] is patches[0] if patch_shortcut else received[0] is None
    assert torch.isfinite(output).all()
    torch.testing.assert_close(history[:, -1:], output)


@pytest.mark.parametrize("patch_shortcut", [False, True])
def test_patch_shortcut_training_and_export_forward(patch_shortcut):
    model = _build_tiny_group_vae(0.0, patch_decoder_shortcut=patch_shortcut, loss_weight=(1, 1, 1, 1, 1)).train()
    batch = _build_batch()
    losses = model(batch)
    sum(losses.values()).backward()
    gradient = model.vae_encoder_group[0].patch_embed.proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    model.eval()
    value = torch.zeros(1)
    output = model.export_onnx(batch["inputs"].clone(), value, value, value)
    assert torch.isfinite(output).all()


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


@pytest.mark.parametrize("steps,segment_size,alpha", [(42, 10, 1.0), (36, 9, 1.0), (7, 3, 0.98)])
def test_partial_tail_segment_preserves_per_lead_weights(steps, segment_size, alpha):
    torch.manual_seed(0)
    model = _build_tiny_group_vae(latent_dyn_weight=0.0, alpha=alpha, rollout_max=-1).train()
    batch = _build_batch(future_frames=steps)
    with torch.no_grad():
        torch.manual_seed(1)
        full_losses = model(batch)
        torch.manual_seed(1)
        totals, state, done = {}, None, False
        while not done:
            losses, state, done = model(batch if state is None else None, rollout_state=state, rollout_segment_size=segment_size)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0) + value
    expected = alpha ** torch.arange(steps, dtype=torch.float32)
    expected /= expected.sum()
    torch.testing.assert_close(state["rolling_weights"], expected)
    assert state["loss_block_indices"] == set(range(steps))
    assert totals.keys() == full_losses.keys()
    for name, value in full_losses.items():
        torch.testing.assert_close(totals[name], value)


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


@pytest.mark.parametrize("image_size,bands", [((17, 32), ((1, 2), (3, 4))), ((120, 240), ((1, 4), (5, 20), (21, 40), (41, 59)))])
def test_spectral_loss_backpropagates_and_segments_sum_to_full_rollout(image_size, bands) -> None:
    model = _build_tiny_group_vae(
        0.0,
        alpha=0.7,
        rollout_max=-1,
        image_size=image_size,
        patch_size=2,
        spectral_loss_weight=0.1,
        spectral_loss_names=("x",),
        spectral_loss_bands=bands,
    ).train()
    baseline = _build_tiny_group_vae(0.0, image_size=image_size, patch_size=2)
    model.load_state_dict(baseline.state_dict(), strict=True)
    batch = {"inputs": torch.randn(1, 2, 1, *image_size), "targets": torch.randn(1, 3, 1, *image_size), "times": torch.tensor([0]), "future_frames": 3}
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


def test_disabled_posterior_is_not_executed() -> None:
    model = _build_tiny_group_vae(0.0, loss_weight=(0.0, 0.0, 1.0, 0.01, 1.0)).train()

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


@pytest.mark.parametrize("amse_weight", [0.0, 0.25, 1.0])
def test_amse_weight_and_segmented_rollout(amse_weight):
    model = _build_tiny_group_vae(0.0, amse_weight=amse_weight, alpha=0.7, rollout_max=-1).train()
    model.eval()
    model.training = True
    batch = _build_batch()
    losses = model(batch)
    if amse_weight == 0:
        assert "loss_amse" not in losses and model.amse_loss is None
        return
    full = losses["loss_amse"]
    full.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder_group.parameters())
    model.amse_weight = 1.0
    torch.testing.assert_close(full.detach(), amse_weight * model(batch)["loss_amse"].detach())
    model.amse_weight = amse_weight
    total, state, done = 0.0, None, False
    while not done:
        losses, state, done = model(batch if state is None else None, rollout_state=state, rollout_segment_size=1)
        total = total + losses["loss_amse"].detach()
    torch.testing.assert_close(total, full.detach())


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_amse_weight(weight):
    with pytest.raises(ValueError, match="amse_weight"):
        _build_tiny_group_vae(0.0, amse_weight=weight)


@pytest.mark.parametrize("spectral_image_size", [None, (121, 240)])
def test_amse_uses_native_grid_without_spectral_loss(spectral_image_size):
    model = _build_tiny_group_vae(0.0, image_size=(120, 240), amse_weight=0.8, spectral_image_size=spectral_image_size)
    expected_size = (120, 240) if spectral_image_size is None else spectral_image_size
    assert model.spectral_loss is None
    assert model.amse_loss.image_size == expected_size
    output = torch.full((1, 1, 1, *expected_size), 1.1, requires_grad=True)
    loss = model.amse_loss(output, torch.ones_like(output))
    torch.testing.assert_close(loss, torch.tensor(0.01), atol=1e-6, rtol=1e-4)
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0
