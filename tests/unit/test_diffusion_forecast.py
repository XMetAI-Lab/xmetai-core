"""Unit tests for DiffusionForecast."""

import torch
import torch.nn as nn

from xmetai.models.diffusion_forecast import DiffusionForecast


def _make_model(**kwargs):
    params = dict(
        in_chans=1,
        out_chans=1,
        in_frames=1,
        out_frames=1,
        backbone="darknet_unet",
        backbone_kwargs=dict(base_channels=4, ch_mults=(1, 2), layers_per_block=1, dropout=0.0),
        num_steps=4,
        sample_steps=1,
        sample_timestep_max=3,
        train_timestep_max=3,
        sample_init="zeros",
        device="cpu",
        precision="fp32",
        test_frames=[1],
        test_chans=[0],
        test_names=["a"],
        step_range=[1],
    )
    params.update(kwargs)
    return DiffusionForecast(**params)


def test_darknet_zero_init_output_uses_output_layer_alias():
    model = _make_model(zero_init_output=True)

    assert model.unet.conv_out is model.unet.out
    assert torch.count_nonzero(model.unet.out.weight) == 0
    assert torch.count_nonzero(model.unet.out.bias) == 0


def test_forward_and_backward_use_shared_conditioning_contract():
    model = _make_model()
    loss = model(
        {
            "inputs": torch.zeros(2, 1, 1, 4, 4),
            "targets": torch.ones(2, 1, 1, 4, 4),
            "times": torch.tensor([946684800000000000, 946771200000000000]),
        }
    )["loss_rec"]
    loss.backward()

    assert torch.isfinite(loss)
    assert model.unet.out.weight.grad is not None


def test_timestamp_condition_advances_during_rollout():
    class ZeroUNet(nn.Module):
        def forward(self, x, timesteps, return_dict=False):
            return (x.new_zeros(x.shape[0], 1, x.shape[-2], x.shape[-1]),)

    model = _make_model(
        start_frames=3,
        test_frames=[1, 2],
        step_range=[2],
    )
    model.unet = ZeroUNet()
    model.eval()

    seen_steps = []
    original_condition_extra = model.condition_extra

    def record_condition_extra(data, input, step=0):
        seen_steps.append(step)
        return original_condition_extra(data, input, step)

    model.condition_extra = record_condition_extra
    data = model.process_data(
        {
            "inputs": torch.zeros(1, 1, 1, 4, 4),
            "times": torch.tensor([946684800000000000]),
        }
    )

    out = model.test_cascade(data)

    assert seen_steps == [3, 4]
    assert out.shape == (1, 2, 1, 4, 4)


def test_timestamp_condition_is_enabled_by_default():
    model = _make_model()

    assert model.condition_chans == model.in_chans * model.in_frames + model.const_chans + 3


def test_timestamp_condition_can_be_disabled_without_changing_call_contract():
    model = _make_model(use_timestamp=False)
    data = model.process_data({"inputs": torch.zeros(1, 1, 1, 4, 4), "times": torch.tensor([946684800000000000])})

    extra = model.condition_extra(data, data["input"], step=2)

    assert model.condition_chans == model.in_chans * model.in_frames + model.const_chans
    assert model.unet.stem.conv.weight.shape[1] == model.condition_chans + model.model_out_chans
    assert extra.shape == (1, 0, 4, 4)
