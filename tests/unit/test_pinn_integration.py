"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-02-14 20:49:05
LastEditTime: 2026-02-14 21:40:45
Description:
Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import math

import numpy as np
import pytest
import torch

from xmetai.core.base_predictor import BasePredictor


class DummyPredictor(BasePredictor):
    def __init__(self, in_names, *args, **kwargs):
        super().__init__(in_chans=len(in_names), out_chans=len(in_names), in_frames=1, out_frames=1, in_names=in_names, *args, **kwargs)

    def forward(self, data):
        data = self.process_data(data)
        input = data["input"]
        target = data.get("targets")
        # identity prediction (last frame)
        output = input[:, -1:]
        loss_rec = torch.nn.functional.mse_loss(output, target)
        loss_dict = {"loss_rec": loss_rec}
        if self.pinn_enable:
            loss_dict.update(self.compute_pinn_losses(output, target))
        if self.precip_loss_enable:
            loss_dict.update(self.compute_precip_losses(output, target))
        return loss_dict


def _make_channel_names():
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    vars = ["z", "t", "u", "v", "q"]
    names = [f"{v}{lvl}" for v in vars for lvl in levels]
    # add a few surface channels
    names += ["msl", "t2m", "sst", "tp"]
    return names


def test_pinn_enable_and_grad_flow():
    torch.manual_seed(0)
    names = _make_channel_names()
    C = len(names)
    H, W = 8, 12
    B = 2

    model = DummyPredictor(in_names=names, device="cpu", precision="fp32", pinn_enable=True, pinn_coeffs={"mass": 1e-3})
    model.train()

    inputs = torch.randn(B, 1, C, H, W, requires_grad=True)
    targets = torch.randn(B, 1, C, H, W)
    times = torch.tensor([946684800000000000, 946684800000000000])

    loss = model({"inputs": inputs, "targets": targets, "times": times})
    # expect at least one pinn key present
    pinn_keys = [k for k in loss.keys() if k.startswith("pinn_")]
    assert len(pinn_keys) > 0

    total = sum(loss.values())
    total.backward()
    assert inputs.grad is not None
    assert inputs.grad.abs().sum() > 0


def test_inv_normalize_preserves_gradients():
    names = ["tp", "t2m"]
    buffers = {
        "mean": np.array([0.0, 1.0], dtype=np.float32).reshape(2, 1, 1),
        "std": np.array([1.0, 2.0], dtype=np.float32).reshape(2, 1, 1),
    }
    model = DummyPredictor(in_names=names, device="cpu", precision="fp32", buffers=buffers, logid=[0])
    normalized = torch.tensor([[[[[0.5]], [[1.0]]]]], requires_grad=True)

    physical = model.inv_normalize(normalized)
    physical.sum().backward()

    assert normalized.grad is not None
    expected = torch.tensor([math.exp(0.5), 2.0])
    assert torch.allclose(normalized.grad.reshape(-1), expected)


def test_normalize_inv_normalize_roundtrip_with_log_channel_statistics():
    names = ["tp", "t2m"]
    buffers = {
        "mean": np.array([0.2, 1.0], dtype=np.float32).reshape(2, 1, 1),
        "std": np.array([2.0, 4.0], dtype=np.float32).reshape(2, 1, 1),
    }
    model = DummyPredictor(in_names=names, device="cpu", precision="fp32", buffers=buffers, logid=[0])
    physical = torch.tensor([[[[[10.0]], [[9.0]]]]])
    expected = torch.tensor([[[[[10.0]], [[9.0]]]]])

    restored = model.inv_normalize(model.normalize(physical))

    assert torch.allclose(restored, expected)


def test_pinn_disabled_no_keys():
    names = _make_channel_names()
    C = len(names)
    H, W = 4, 6
    B = 1

    model = DummyPredictor(in_names=names, device="cpu", precision="fp32", pinn_enable=False)
    model.train()

    inputs = torch.randn(B, 1, C, H, W)
    targets = torch.randn(B, 1, C, H, W)
    times = torch.tensor([946684800000000000])

    loss = model({"inputs": inputs, "targets": targets, "times": times})
    pinn_keys = [k for k in loss.keys() if k.startswith("pinn_")]
    assert len(pinn_keys) == 0
    precip_keys = [k for k in loss.keys() if k.startswith("precip_")]
    assert len(precip_keys) == 0


def test_precip_loss_enable_and_keys():
    names = ["tp", "t2m"]
    C = len(names)
    H, W = 4, 6
    B = 1
    buffers = {
        "mean": np.zeros((C, 1, 1), dtype=np.float32),
        "std": np.ones((C, 1, 1), dtype=np.float32),
        "weight": np.ones((1, H, 1), dtype=np.float32),
    }

    model = DummyPredictor(
        in_names=names,
        device="cpu",
        precision="fp32",
        buffers=buffers,
        logid=[0],
        precip_loss_enable=True,
        precip_loss_coeffs={"occurrence": 1.0, "distribution": 1.0},
        precip_names=("tp",),
        precip_threshold_mm=0.1,
    )
    model.train()

    inputs = torch.randn(B, 1, C, H, W)
    targets = torch.randn(B, 1, C, H, W)
    times = torch.tensor([946684800000000000])

    loss = model({"inputs": inputs, "targets": targets, "times": times})

    assert "precip_loss_occurrence" in loss
    assert "precip_loss_distribution" in loss


def test_precip_threshold_mm_uses_log_transform_for_logid():
    names = ["tp", "t2m"]
    buffers = {
        "mean": np.array([0.2, 0.0], dtype=np.float32).reshape(2, 1, 1),
        "std": np.array([2.0, 1.0], dtype=np.float32).reshape(2, 1, 1),
    }

    model = DummyPredictor(
        in_names=names,
        device="cpu",
        precision="fp32",
        buffers=buffers,
        logid=[0],
        precip_loss_enable=True,
        precip_loss_coeffs={"occurrence": 1.0},
        precip_names=("tp",),
        precip_threshold_mm=0.1,
    )

    expected = (math.log1p(0.1) - 0.2) / 2.0
    assert model.precip_threshold == pytest.approx(expected)


def test_precip_threshold_mm_applies_native_unit_scale():
    names = ["tp", "t2m"]
    buffers = {
        "mean": np.array([0.2, 0.0], dtype=np.float32).reshape(2, 1, 1),
        "std": np.array([2.0, 1.0], dtype=np.float32).reshape(2, 1, 1),
    }

    model = DummyPredictor(
        in_names=names,
        device="cpu",
        precision="fp32",
        buffers=buffers,
        logid=[0],
        precip_loss_enable=True,
        precip_loss_coeffs={"occurrence": 1.0},
        precip_names=("tp",),
        precip_threshold_mm=0.1,
        precip_mm_scale=21.6,
    )

    expected = (math.log1p(0.1 / 21.6) - 0.2) / 2.0
    assert model.precip_threshold == pytest.approx(expected)


def test_precip_loss_enable_fails_without_precip_channel():
    buffers = {
        "mean": np.zeros((1, 1, 1), dtype=np.float32),
        "std": np.ones((1, 1, 1), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="none of precip_names"):
        DummyPredictor(
            in_names=["t2m"],
            device="cpu",
            precision="fp32",
            buffers=buffers,
            precip_loss_enable=True,
            precip_loss_coeffs={"occurrence": 1.0},
            precip_names=("tp",),
        )
