"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-02-14 20:49:05
LastEditTime: 2026-02-14 21:40:45
Description:
Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import math

import torch

from xmetai.models.base_predictor import BasePredictor


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
