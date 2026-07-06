import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from xmetai.metrics.threshold import FalseAlarmRate, FrequencyBias, ProbabilityOfDetection, ThreatScore
from xmetai.models.radar_flowcast import RadarFlowCast
from xmetai.models.radar_latent_forecast import RadarLatentForecast
from xmetai.models.radar_vae import RadarVAE


def test_radar_vae_forward_and_eval_shapes():
    model = RadarVAE(
        base_channels=8,
        ch_mults=(1, 2, 2),
        layers_per_block=1,
        latent_channels=4,
        time_dim=16,
        strong_thresholds_dbz=(35.0,),
        mcr_weight=0.1,
        mcr_pool_size=4,
        precision="fp32",
        device="cpu",
    )
    inputs = torch.zeros(2, 1, 1, 64, 64)
    inputs[:, :, :, 20:32, 24:36] = 45.0
    data = {"inputs": inputs}

    model.train()
    losses = model(data)
    assert set(losses) == {"loss_rec", "loss_echo_bce", "loss_strong_area", "loss_strong_shape", "loss_kl", "loss_mcr"}
    assert all(torch.isfinite(value) for value in losses.values())

    model.eval()
    outputs = model(data)
    assert outputs["output"].shape == (2, 1, 1, 64, 64)
    assert outputs["target"].shape == (2, 1, 1, 64, 64)
    assert outputs["echo_prob"].shape == (2, 1, 1, 64, 64)


def test_radar_vae_eval_saves_pngs(tmp_path):
    save_dir = tmp_path / "pred"
    boundary_path = tmp_path / "boundary.geojson"
    boundary_path.write_text(json.dumps({"type": "LineString", "coordinates": [[0.0, 1.0], [1.0, 0.0]]}), encoding="utf-8")
    model = RadarVAE(
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        precision="fp32",
        device="cpu",
        save_dir=str(save_dir),
        save_radar_bbox=(0.0, 0.0, 1.0, 1.0),
        save_radar_boundary_geojson=str(boundary_path),
        save_radar_boundary_color=(0, 0, 0, 255),
    )
    inputs = torch.zeros(1, 1, 1, 16, 16)
    inputs[:, :, :, 4:8, 5:9] = 45.0
    data = {
        "inputs": inputs,
        "times": torch.tensor([pd.Timestamp("2019-12-16 00:06:00").value]),
    }

    model.eval()
    outputs = model(data)

    assert outputs["output"].shape == (1, 1, 1, 16, 16)
    assert (save_dir / "pred_20191216_000600.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (save_dir / "obs_20191216_000600.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    obs_rgba = np.asarray(Image.open(save_dir / "obs_20191216_000600.png").convert("RGBA"))
    assert np.any(np.all(obs_rgba == np.array([0, 0, 0, 255], dtype=np.uint8), axis=-1))


def test_radar_latent_forecast_forward_and_eval_shapes():
    model = RadarLatentForecast(
        in_frames=2,
        out_frames=3,
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        latent_image_size=(8, 8),
        embed_dim=8,
        depths=(1,),
        num_heads=2,
        patch_size=2,
        window_size=2,
        drop_rate=0.0,
        layer_type="swin",
        attn_type="v2",
        decode_loss_weight=0.1,
        strong_thresholds_dbz=(35.0,),
        precision="fp32",
        device="cpu",
    )
    inputs = torch.zeros(2, 2, 1, 16, 16)
    targets = torch.zeros(2, 3, 1, 16, 16)
    inputs[:, :, :, 4:8, 5:9] = 35.0
    targets[:, :, :, 5:9, 6:10] = 45.0
    data = {"inputs": inputs, "targets": targets}

    model.train()
    losses = model(data)
    assert set(losses) == {"loss_latent", "loss_dec_rec"}
    assert all(torch.isfinite(value) for value in losses.values())

    model.eval()
    outputs = model(data)
    assert outputs["output"].shape == (2, 3, 1, 16, 16)
    assert outputs["target"].shape == (2, 3, 1, 16, 16)
    assert outputs["echo_prob"].shape == (2, 3, 1, 16, 16)
    assert outputs["latent_output"].shape[:3] == (2, 3, 2)
    assert outputs["latent_target"].shape[:3] == (2, 3, 2)


def test_radar_flowcast_forward_and_eval_shapes():
    model = RadarFlowCast(
        in_frames=2,
        out_frames=2,
        flow_backbone="darknet_unet",
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        flow_sigma=0.01,
        euler_steps=2,
        sample_members=1,
        flow_base_channels=4,
        flow_ch_mults=(1, 2),
        flow_layers_per_block=1,
        flow_time_dim=8,
        strong_thresholds_dbz=(35.0,),
        precision="fp32",
        device="cpu",
    )
    inputs = torch.zeros(2, 2, 1, 16, 16)
    targets = torch.zeros(2, 2, 1, 16, 16)
    inputs[:, :, :, 4:8, 5:9] = 35.0
    targets[:, :, :, 5:9, 6:10] = 45.0
    data = {"inputs": inputs, "targets": targets}

    model.train()
    losses = model(data)
    assert set(losses) == {"loss_flow"}
    assert all(torch.isfinite(value) for value in losses.values())

    model.eval()
    outputs = model(data)
    assert outputs["output"].shape == (2, 2, 1, 16, 16)
    assert outputs["target"].shape == (2, 2, 1, 16, 16)
    assert outputs["echo_prob"].shape == (2, 2, 1, 16, 16)


def test_radar_latent_forecast_loads_and_freezes_vae_checkpoint(tmp_path):
    vae = RadarVAE(
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        precision="fp32",
        device="cpu",
    )
    with torch.no_grad():
        vae.to_mu.bias.fill_(1.25)
    checkpoint_path = tmp_path / "model_best.pth"
    torch.save({"model": vae.state_dict()}, checkpoint_path)

    model = RadarLatentForecast(
        in_frames=2,
        out_frames=3,
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        latent_image_size=(8, 8),
        embed_dim=8,
        depths=(1,),
        num_heads=2,
        patch_size=2,
        window_size=2,
        drop_rate=0.0,
        layer_type="swin",
        attn_type="v2",
        vae_ckpt=str(checkpoint_path),
        freeze_vae=True,
        precision="fp32",
        device="cpu",
    )

    assert torch.allclose(model.to_mu.bias, torch.full_like(model.to_mu.bias, 1.25))
    assert not any(param.requires_grad for name, param in model.named_parameters() if not name.startswith("latent_predictor."))
    assert any(param.requires_grad for name, param in model.named_parameters() if name.startswith("latent_predictor."))


def test_radar_latent_forecast_allows_missing_vae_checkpoint(tmp_path):
    model = RadarLatentForecast(
        in_frames=2,
        out_frames=3,
        base_channels=4,
        ch_mults=(1, 2),
        layers_per_block=1,
        latent_channels=2,
        time_dim=8,
        latent_image_size=(8, 8),
        embed_dim=8,
        depths=(1,),
        num_heads=2,
        patch_size=2,
        window_size=2,
        drop_rate=0.0,
        layer_type="swin",
        attn_type="v2",
        vae_ckpt=str(tmp_path / "missing_model_best.pth"),
        freeze_vae=True,
        precision="fp32",
        device="cpu",
    )

    assert any(param.requires_grad for name, param in model.named_parameters() if name.startswith("latent_predictor."))


def test_threshold_metrics_score_contingency_table():
    metrics = [
        ThreatScore(thresholds=(35.0,), test_times=(0,), neighborhood_size=1),
        FalseAlarmRate(thresholds=(35.0,), test_times=(0,), neighborhood_size=1),
        ProbabilityOfDetection(thresholds=(35.0,), test_times=(0,), neighborhood_size=1),
        FrequencyBias(thresholds=(35.0,), test_times=(0,), neighborhood_size=1),
    ]
    target = torch.zeros(1, 1, 1, 8, 8)
    output = torch.zeros_like(target)
    target[..., 2:4, 2:4] = 45.0
    output[..., 2:4, 2:4] = 45.0
    output[..., 6, 6] = 45.0

    results = {}
    for metric in metrics:
        metric.reset()
        metric.process({}, {"output": output, "target": target})
        metric_results = metric.evaluate()
        assert metric_results is not None
        results.update(metric_results)

    assert results["ThreatScore"]["Times"] == [0]
    assert results["ThreatScore"]["Threshold"] == [35.0]
    assert abs(results["ThreatScore"]["TS"][0] - 0.8) < 1.0e-6
    assert results["FalseAlarmRate"]["Times"] == [0]
    assert abs(results["FalseAlarmRate"]["FAR"][0] - 0.2) < 1.0e-6
    assert results["ProbabilityOfDetection"]["Times"] == [0]
    assert abs(results["ProbabilityOfDetection"]["POD"][0] - 1.0) < 1.0e-6
    assert results["FrequencyBias"]["Times"] == [0]
    assert abs(results["FrequencyBias"]["Bias"][0] - 1.25) < 1.0e-6


def test_threshold_metrics_require_test_times():
    with pytest.raises(TypeError, match="test_times"):
        ThreatScore(thresholds=(35.0,), neighborhood_size=1)


def test_threshold_metrics_use_test_times_with_frame_start():
    metric = ThreatScore(thresholds=(35.0,), neighborhood_size=1, test_times=(6, 12, 18))
    metric.reset()

    target = torch.zeros(1, 2, 1, 4, 4)
    output = torch.zeros_like(target)
    target[:, 0, 0, 1, 1] = 45.0
    output[:, 0, 0, 1, 1] = 45.0
    target[:, 1, 0, 1, 1] = 45.0
    output[:, 1, 0, 3, 3] = 45.0

    metric.process({}, {"output": output, "target": target, "frame_start": 1})
    results = metric.evaluate()
    assert results is not None
    assert results["ThreatScore"]["Times"] == [6, 12, 18]
    assert results["ThreatScore"]["Threshold"] == [35.0, 35.0, 35.0]
    assert results["ThreatScore"]["TS"][0] == 0.0
    assert abs(results["ThreatScore"]["TS"][1] - 1.0) < 1.0e-6
    assert results["ThreatScore"]["TS"][2] == 0.0
