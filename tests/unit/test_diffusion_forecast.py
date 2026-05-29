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
        base_channels=4,
        ch_mults=(1, 2),
        attn_res=(),
        num_heads=1,
        dropout=0.0,
        layers_per_block=1,
        backbone="darknet",
        use_attention=False,
        num_steps=4,
        sample_steps=1,
        sample_timestep_max=3,
        train_timestep_max=3,
        sample_init="zeros",
        image_size=(4, 4),
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


def test_timestamp_condition_advances_during_rollout():
    class ZeroUNet(nn.Module):
        def forward(self, x, timesteps, return_dict=False):
            return (x.new_zeros(x.shape[0], 1, x.shape[-2], x.shape[-1]),)

    model = _make_model(
        use_timestamp=True,
        start_frames=3,
        test_frames=[1, 2],
        step_range=[2],
    )
    model.unet = ZeroUNet()
    model.eval()

    seen_steps = []
    original_condition_timestamp = model.condition_timestamp

    def record_condition_timestamp(data, input, step):
        seen_steps.append(step)
        return original_condition_timestamp(data, input, step)

    model.condition_timestamp = record_condition_timestamp
    data = model.process_data(
        {
            "inputs": torch.zeros(1, 1, 1, 4, 4),
            "times": torch.tensor([946684800000000000]),
        }
    )

    out = model.test_cascade(data)

    assert seen_steps == [3, 4]
    assert out.shape == (1, 2, 1, 4, 4)
