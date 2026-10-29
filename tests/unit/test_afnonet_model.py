"""Unit tests for FourcastNet."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from xmetai.core.base_predictor import BasePredictor
from xmetai.models.afno import AFNONet


def test_afnonet_forward_loss():
    torch.manual_seed(0)
    bsz, t_in, chans, height, width = 2, 2, 3, 120, 240
    inputs = torch.randn(bsz, t_in, chans, height, width)
    targets = torch.randn(bsz, 1, chans, height, width)
    times = torch.tensor([946684800000000000, 946684800000000000])

    model = AFNONet(
        in_chans=chans,
        out_chans=chans,
        in_frames=t_in,
        out_frames=1,
        embed_dim=768,
        img_size=(120, 240),
        patch_size=(5, 5),
        depth=2,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        device="cpu",
        precision="fp32",
        test_frames=[1],
        test_chans=[0, 1, 2],
        test_names=["a", "b", "c"],
        step_range=[1],
    )
    model.train()

    loss = model(
        {
            "inputs": inputs,
            "targets": targets,
            "times": times,
        }
    )

    assert "loss_rec" in loss


def test_afnonet_test_cascade():
    bsz, t_in, chans, height, width = 1, 2, 3, 120, 240
    inputs = torch.randn(bsz, t_in, chans, height, width)
    times = torch.tensor([946684800000000000])

    model = AFNONet(
        in_chans=chans,
        out_chans=chans,
        in_frames=t_in,
        out_frames=1,
        embed_dim=768,
        img_size=(120, 240),
        patch_size=(5, 5),
        depth=2,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        device="cpu",
        precision="fp32",
        test_frames=[1],
        test_chans=[0, 1, 2],
        test_names=["a", "b", "c"],
        step_range=[1],
    )
    model.eval()

    data = {
        "inputs": inputs,
        "times": times,
    }
    data = model.process_data(data)
    out = model.test_cascade(data)
    assert out.shape[1] == len(model.test_frames)
    assert out.shape[-2:] == (height, width)


@pytest.mark.parametrize("exporting", [False, True])
def test_export_returns_only_prediction_after_functional_reset(exporting):
    model = SimpleNamespace(
        dtype=torch.float32,
        accumid=[0],
        zero_accum=True,
        maskid=[1],
        mask=torch.tensor(0.5),
        landid=[2],
        land_mask=torch.tensor(0.25),
        logid=[3],
        normalize=lambda data: data.clone(),
        inv_normalize=lambda data: data,
        forward_step=lambda data: data[:, -1:] + 1,
    )
    model.reset_input = MethodType(BasePredictor.reset_input, model)
    inputs = torch.tensor([2.0, 4.0, 8.0, 10.0]).reshape(1, 1, 4, 1, 1).expand(2, 2, 4, 2, 4).clone()
    expected = torch.tensor([1.0, 3.0, 3.0, 8.0]).reshape(1, 1, 4, 1, 1).expand(2, 1, 4, 2, 4)

    with patch("torch.onnx.is_in_onnx_export", return_value=exporting):
        output = AFNONet.export_onnx(model, inputs)

    torch.testing.assert_close(output, expected)
