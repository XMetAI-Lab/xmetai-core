"""Behavior tests for conditional flow matching and JiT forecasts."""

import pytest
import torch

from xmetai import MODELS
from xmetai.models.diffusion_forecast import ConditionalGenerativeForecast, DiffusionForecast, FlowMatchingForecast, JiTForecast, ODEGenerativeForecast


def _base_params():
    return dict(
        in_chans=1,
        out_chans=1,
        in_frames=1,
        out_frames=1,
        sample_steps=2,
        sampling_method="euler",
        sample_init="zeros",
        test_frames=[1],
        test_chans=[0],
        test_names=["a"],
        step_range=[1],
        device="cpu",
        precision="fp32",
    )


def _batch(height=5, width=6, future_steps=1):
    return {
        "inputs": torch.zeros(2, 1, 1, height, width),
        "targets": torch.ones(2, future_steps, 1, height, width),
        "times": torch.tensor([946684800000000000, 946771200000000000]),
        "future_frames": future_steps,
    }


def test_models_are_registered():
    assert MODELS.get("FlowMatchingForecast") is FlowMatchingForecast
    assert MODELS.get("JiTForecast") is JiTForecast


def test_models_share_the_generative_forecast_contract():
    assert issubclass(DiffusionForecast, ConditionalGenerativeForecast)
    assert issubclass(FlowMatchingForecast, ODEGenerativeForecast)
    assert issubclass(JiTForecast, ODEGenerativeForecast)
    assert {model.__module__ for model in (ConditionalGenerativeForecast, ODEGenerativeForecast, DiffusionForecast, FlowMatchingForecast, JiTForecast)} == {
        "xmetai.models.diffusion_forecast"
    }


def test_flow_matching_forward_and_backward_on_rectangular_field():
    model = FlowMatchingForecast(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **_base_params())
    loss = model(_batch())["loss_flow"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.flow_net.out.weight.grad is not None


def test_flow_matching_rollout_uses_independent_stochastic_model_states():
    model = FlowMatchingForecast(
        backbone="darknet_unet",
        backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1),
        rollout_max=2,
        rollout_alpha=0.98,
        rollout_sample_steps=1,
        **_base_params(),
    )
    calls = []

    def record_velocity(cond, state, time, extra):
        calls.append((cond.detach().clone(), state.detach().clone(), time.detach().clone()))
        return torch.zeros_like(state)

    model.predict_velocity = record_velocity
    torch.manual_seed(7)
    loss = model(_batch(future_steps=3))["loss_flow"]

    rollout_calls = [(cond, state) for cond, state, time in calls if torch.count_nonzero(time) == 0]
    assert torch.isfinite(loss)
    assert len(calls) == 4  # two selected loss leads plus two intermediate rollout samples
    assert len(rollout_calls) == 2
    assert torch.count_nonzero(rollout_calls[1][0]) > 0  # the next lead is conditioned on a generated state
    assert not torch.equal(rollout_calls[0][1], rollout_calls[1][1])


@pytest.mark.parametrize(("model_type", "loss_name"), [(DiffusionForecast, "loss_rec"), (FlowMatchingForecast, "loss_flow"), (JiTForecast, "loss_jit")])
def test_all_generative_objectives_support_rollout_training(model_type, loss_name):
    params = dict(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **_base_params())
    params.update(rollout_max=2, rollout_alpha=0.98, rollout_sample_steps=1)
    if model_type is DiffusionForecast:
        params.pop("sampling_method")
        params.update(num_steps=4, sample_steps=2, sample_timestep_max=3, train_timestep_max=3)
    model = model_type(**params)
    rollout_states = []
    sample_residual = model.sample_residual

    def record_sample_residual(cond, extra, sample_steps=None, stochastic=None):
        state = sample_residual(cond, extra, sample_steps, stochastic)
        rollout_states.append(state.detach().clone())
        return state

    model.sample_residual = record_sample_residual
    loss = model(_batch(future_steps=3))[loss_name]
    loss.backward()

    assert torch.isfinite(loss)
    assert len(rollout_states) == 2
    assert not torch.equal(rollout_states[0], rollout_states[1])
    assert model.generative_backbone.out.weight.grad is not None


def test_timestamp_condition_broadcasts_for_ensemble_member_batch():
    model = FlowMatchingForecast(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **_base_params())
    raw = _batch()
    raw = {key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    processed = model.process_data(raw)
    input = processed["input"].repeat(3, 1, 1, 1, 1)

    extra = model.condition_extra(processed, input, step=2)

    assert extra.shape == (3, 3, input.shape[-2], input.shape[-1])
    torch.testing.assert_close(extra[0], extra[1])


def test_jit_forward_and_backward_with_patch_padding():
    model = JiTForecast(
        backbone="utransformer",
        backbone_kwargs=dict(embed_dim=16, image_size=(8, 8), patch_size=2, window_size=2, depths=(1,), num_heads=4, head_type="2fc", embed_type="step"),
        **_base_params(),
    )
    loss = model(_batch())["loss_jit"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.jit.head[-1].weight.grad is not None


def test_jit_cascade_preserves_shape_for_non_patch_multiple():
    model = JiTForecast(
        backbone="utransformer",
        backbone_kwargs=dict(embed_dim=16, image_size=(8, 8), patch_size=2, window_size=2, depths=(1,), num_heads=4, head_type="2fc", embed_type="step"),
        **_base_params(),
    )
    model.eval()
    data = model.process_data(_batch(height=5, width=6))

    output = model.test_cascade(data)

    assert output.shape == (2, 1, 1, 5, 6)


@pytest.mark.parametrize("model_type", [DiffusionForecast, FlowMatchingForecast, JiTForecast])
def test_cascade_averages_independent_member_trajectories(model_type):
    params = _base_params()
    params.update(members=2, test_frames=[1, 2], step_range=[2])
    if model_type is DiffusionForecast:
        params.pop("sampling_method")
        params.update(num_steps=4, sample_steps=2, sample_timestep_max=3, train_timestep_max=3)
    model = model_type(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **params)
    calls = []

    def sample_residual(cond, extra, sample_steps=None, stochastic=None):
        calls.append(cond.detach().clone())
        sample_shape = (cond.shape[0], model.model_out_chans, *cond.shape[-2:])
        if len(calls) == 1:
            return cond.new_ones(sample_shape)
        if len(calls) == 3:
            return cond.new_full(sample_shape, 3.0)
        return cond[:, -1:].square()

    model.sample_residual = sample_residual
    model.eval()
    data = model.process_data({"inputs": torch.zeros(1, 1, 1, 2, 2), "times": torch.tensor([946684800000000000])})

    output = model.test_cascade(data)

    torch.testing.assert_close(output[:, 0], torch.full_like(output[:, 0], 2.0))
    torch.testing.assert_close(output[:, 1], torch.full_like(output[:, 1], 7.0))
    assert len(calls) == 4
    assert torch.count_nonzero(calls[2]) == 0


def test_multiple_members_use_stochastic_initial_samples():
    params = _base_params()
    params.update(members=2)
    model = FlowMatchingForecast(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **params)
    cond = torch.zeros(1, model.in_chans * model.in_frames, 2, 2)
    extra = cond.new_empty(1, 0, 2, 2)
    model.predict_velocity = lambda cond, state, time, extra: torch.zeros_like(state)

    torch.manual_seed(7)
    ensemble_sample = model.sample_residual(cond, extra, sample_steps=1)
    model.members = 1
    single_sample = model.sample_residual(cond, extra, sample_steps=1)

    assert torch.count_nonzero(ensemble_sample) > 0
    assert torch.count_nonzero(single_sample) == 0


def test_experiment_models_have_effectively_equal_parameter_counts():
    shared = dict(
        in_chans=76,
        out_chans=76,
        in_frames=2,
        out_frames=1,
        const_chans=2,
        train_chans=[0],
        device="cpu",
        precision="fp32",
    )
    flow = FlowMatchingForecast(backbone="darknet_unet", backbone_kwargs=dict(base_channels=128, ch_mults=(1, 2, 3), layers_per_block=3), **shared)
    jit = JiTForecast(
        backbone="utransformer",
        backbone_kwargs=dict(embed_dim=232, image_size=(120, 240), patch_size=8, window_size=5, depths=(2, 2, 2, 2), num_heads=8, head_type="2fc"),
        **shared,
    )

    relative_difference = abs(flow.params() - jit.params()) / flow.params()
    assert relative_difference < 0.02


def test_darknet_checkpoint_prefixes_remain_compatible():
    params = dict(backbone="darknet_unet", backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1), **_base_params())
    flow = FlowMatchingForecast(**params)
    diffusion_params = dict(params)
    diffusion_params.pop("sampling_method")
    diffusion_params["sample_steps"] = 2
    diffusion = DiffusionForecast(num_steps=4, **diffusion_params)

    flow_state = flow.state_dict()
    diffusion_state = diffusion.state_dict()
    assert any(key.startswith("flow_net.") for key in flow_state)
    assert any(key.startswith("unet.") for key in diffusion_state)
    assert not any(key.startswith("backbone.") for key in flow_state | diffusion_state)
    flow.load_state_dict(flow_state, strict=True)
    diffusion.load_state_dict(diffusion_state, strict=True)
