from datetime import timedelta

import pandas as pd
import pytest
import torch
import torch.nn as nn

from xmetai.layers import FGA, SCA, DEConv, PEDadd_UViT, Perceiver3DDecoder, Perceiver3DEncoder
from xmetai.models.tianquan import TianQuan


def test_perceiver3d_encoder_supports_none_static_vars():
    encoder = Perceiver3DEncoder(
        surf_vars=("2m_temperature", "10m_wind_speed"),
        static_vars=None,
        atmos_vars=("geopotential", "wind_speed", "temperature", "relative_humidity", "specific_humidity"),
        atmos_levels=(500, 850),
        patch_size=2,
        latent_levels=2,
        embed_dim=16,
        num_heads=2,
        head_dim=8,
        drop_rate=0.0,
        depth=1,
        mlp_ratio=2.0,
        max_history_size=2,
    )

    b, t, h, w = 1, 2, 8, 8
    x_surf = torch.randn(b, t, 2, h, w)
    x_atmos = torch.randn(b, t, 5, 2, h, w)
    lat = torch.linspace(-90, 90, h)
    lon = torch.linspace(0, 360, w)
    hours = [pd.Timestamp("2024-01-01T00:00:00")]

    out = encoder(
        x_surf=x_surf,
        x_static=None,
        x_atmos=x_atmos,
        lead_time=timedelta(hours=24),
        lat=lat,
        lon=lon,
        hours=hours,
    )
    assert out.dim() == 3
    assert out.shape[0] == b


def test_perceiver3d_decoder_forward():
    decoder = Perceiver3DDecoder(
        surf_vars=("2m_temperature", "10m_wind_speed"),
        atmos_vars=("geopotential", "temperature"),
        atmos_levels=(500, 850),
        patch_size=2,
        embed_dim=16,
        depth=1,
        head_dim=8,
        num_heads=2,
        mlp_ratio=2.0,
        drop_rate=0.0,
    )

    surf_pred, atmos_pred = decoder(
        x=torch.randn(1, 8, 16),
        lat=torch.linspace(-90, 90, 4),
        lon=torch.linspace(0, 360, 4),
        patch_res=(2, 2, 2),
        lead_time=timedelta(hours=24),
    )

    assert surf_pred.shape == (1, 2, 4, 4)
    assert atmos_pred.shape == (1, 2, 2, 4, 4)


def test_tianquan_backbone_forward():
    backbone = PEDadd_UViT(embed_dim=16, depth=2, num_heads=4)
    x = torch.randn(2, 8, 16)

    out = backbone(x, use_diffusion=False)

    assert out.shape == x.shape


def test_tianquan_fusion_blocks_forward():
    def conv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    x = torch.randn(2, 8, 8, 8)
    y = torch.randn_like(x)

    assert SCA(conv, dim=8, kernel_size=3)(x).shape == x.shape
    assert DEConv(conv, dim=8, kernel_size=3)(x).shape == x.shape
    assert FGA(dim=8)(x, y).shape == x.shape


def test_tianquan_export_onnx_not_supported(monkeypatch):
    class _DummyClimate:
        pass

    monkeypatch.setattr("xmetai.models.tianquan.tianquan.xr.open_zarr", lambda *args, **kwargs: _DummyClimate())

    model = TianQuan(
        default_vars=["v1", "v2"],
        in_chans=2,
        out_chans=2,
        in_frames=2,
        out_frames=1,
        test_frames=[1],
        test_chans=[0],
        test_names=["v1"],
        step_range=[1],
        freq=24,
        img_size=(8, 16),
        patch_size=2,
        embed_dim=12,
        decoder_depth=1,
        num_heads=1,
        climate_zarr="/tmp/dummy.zarr",
    )

    with pytest.raises(NotImplementedError):
        model.export_onnx(
            inputs=torch.zeros(1, 2, 2, 8, 16),
            step=torch.tensor([0.0]),
            hour=torch.tensor([0.0]),
            doy=torch.tensor([0.0]),
        )
