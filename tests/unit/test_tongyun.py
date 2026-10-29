"""Feature grouping, reconstruction semantics and public S2S config migration."""

from pathlib import Path

import numpy as np
import pytest
import torch

import xmetai
from xmetai.config import LazyConfig
from xmetai.models.tongyun import TongYun


def _tiny_tongyun(**kwargs):
    return TongYun(
        in_names=["q500", "z500", "u500"],
        in_chans=3,
        out_chans=3,
        in_frames=2,
        out_frames=1,
        embed_dim=8,
        image_size=(4, 4),
        patch_size=1,
        window_size=2,
        encoder_depths=[1],
        decoder_depths=[1],
        num_heads=1,
        vt_num_heads=1,
        vt_num_layers=0,
        drop_mcmc=0,
        test_frames=[1],
        test_chans=[0, 1, 2],
        test_names=["q500", "z500", "u500"],
        device="cpu",
        buffers={"mean": np.zeros((3, 1, 1), np.float32), "std": np.ones((3, 1, 1), np.float32), "weight": np.ones((1, 4, 4), np.float32)},
        **kwargs,
    )


def test_feature_groups_preserve_input_and_output_semantics():
    model = _tiny_tongyun()
    assert model.group_slice_indices == [[1], [2], [0]]
    assert model.group_modes == ["residual", "absolute", "absolute"]
    history = torch.tensor([[2.0, 10.0, 3.0], [5.0, 16.0, 7.0]]).reshape(1, 2, 3, 1, 1).expand(1, 2, 3, 4, 4)
    residual_input = model._prepare_group_input(history, 0)
    torch.testing.assert_close(residual_input[:, 0], history[:, 0, 1:2])
    torch.testing.assert_close(residual_input[:, 1], history[:, 1, 1:2] - history[:, 0, 1:2])
    torch.testing.assert_close(model._prepare_group_input(history, 1), history[:, :, 2:3])
    decoded = [torch.full((1, 1, 1, 4, 4), value) for value in (1.0, 2.0, 3.0)]
    output = model._assemble_prediction(decoded, history)
    expected = torch.tensor([3.0, 17.0, 2.0]).reshape(1, 1, 3, 1, 1).expand_as(output)
    torch.testing.assert_close(output, expected)


def test_tongyun_forecast_loss_reaches_feature_encoders_and_decoders():
    model = _tiny_tongyun()
    batch = {"inputs": torch.randn(1, 2, 3, 4, 4), "targets": torch.randn(1, 1, 3, 4, 4), "times": torch.tensor([0]), "future_frames": 1}
    losses = model(batch)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    for encoder in model.vae_encoder_group:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in encoder.parameters())
    for decoder in model.decoder_group:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in decoder.parameters())


@pytest.mark.parametrize("resume", [False, True])
def test_s2s_config_uses_tongyun_without_legacy_weights_or_freezing(monkeypatch, tmp_path, resume):
    names = ["z500", "u200", "u850", "msl", "t2m", "sst", "ttr", "tp"]
    metadata = {
        "in_names": names,
        "out_names": names,
        "logid": [7],
        "accumid": [6, 7],
        "maskid": [5],
        "buffers": {"mean": np.zeros((len(names), 1, 1), np.float32), "std": np.ones((len(names), 1, 1), np.float32)},
    }
    monkeypatch.setattr(xmetai, "get_data_info", lambda data_paths: metadata)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    if resume:
        (tmp_path / "model_final.pth").touch()
    config = LazyConfig.load(str(Path(__file__).resolve().parents[2] / "configs/s2s_tongyun.py"))
    assert config.model._target_ == TongYun
    assert "level_groups" not in config.model
    assert "vae_type" not in config.model
    assert not config.train.freeze
    assert "s2s_group_vae" not in str(config.train.init_checkpoint)
    assert config.model.in_frames == 2
    assert config.model.freq == 24
