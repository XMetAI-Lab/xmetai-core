"""Unit tests for PuYunModel (params aligned with configs/s2s_puyun.py)."""

import numpy as np
import torch

from xmetai.models.puyun import PuYunBackbone, PuYunModel

# configs/s2s_puyun.py
IMAGE_SIZE = (120, 240)
HEIGHT, WIDTH = IMAGE_SIZE
EMBED_DIM = 256
KERNEL_SIZE = 9
PATCH_SIZE = 2
CONST_CHANS = 2
IN_FRAMES = 2
# s2s_base default DEPTH_LIST=6,6,12,12; use shallow depths for fast unit tests.
DEPTHS = [2, 2, 2, 2]
IN_CHANS = 4
OUT_CHANS = 4


def _timestamp(bsz: int):
    return {
        "step": torch.full((bsz,), 1.0),
        "hour": torch.full((bsz,), 0.5),
        "doy": torch.full((bsz,), 0.25),
    }


def _mock_buffers():
    return {"const": np.zeros((CONST_CHANS, HEIGHT, WIDTH), dtype=np.float32)}


def _backbone_kwargs(**overrides):
    kwargs = dict(
        in_chans=IN_CHANS,
        out_chans=OUT_CHANS,
        in_frames=IN_FRAMES,
        out_frames=1,
        const_chans=CONST_CHANS,
        embed_dim=EMBED_DIM,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        depths=DEPTHS,
        kernel_size=KERNEL_SIZE,
        mlp_ratio=4.0,
        drop_rate=0.05,
        drop_path_rate=0.2,
        use_pixel_shuffle=True,
    )
    kwargs.update(overrides)
    return kwargs


def _model_kwargs(**overrides):
    kwargs = dict(
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        kernel_size=KERNEL_SIZE,
        mlp_ratio=4.0,
        drop_rate=0.05,
        drop_path_rate=0.2,
        loss_type="puyun",
        use_pixel_shuffle=True,
        in_chans=IN_CHANS,
        out_chans=OUT_CHANS,
        const_chans=CONST_CHANS,
        in_frames=IN_FRAMES,
        out_frames=1,
        device="cpu",
        precision="fp32",
        test_frames=[1],
        test_chans=list(range(OUT_CHANS)),
        test_names=[f"var{i}" for i in range(OUT_CHANS)],
        step_range=[1],
        buffers=_mock_buffers(),
    )
    kwargs.update(overrides)
    return kwargs


def test_puyun_backbone_forward():
    torch.manual_seed(0)
    bsz = 1
    backbone = PuYunBackbone(**_backbone_kwargs())
    inputs = torch.randn(bsz, IN_FRAMES, IN_CHANS, HEIGHT, WIDTH)
    const = torch.randn(bsz, CONST_CHANS, HEIGHT, WIDTH)

    output = backbone(inputs, const=const, timestamp=_timestamp(bsz))

    assert output.shape == (bsz, 1, OUT_CHANS, HEIGHT, WIDTH)


def test_puyun_forward_loss():
    torch.manual_seed(0)
    bsz = 2
    inputs = torch.randn(bsz, IN_FRAMES, IN_CHANS, HEIGHT, WIDTH)
    targets = torch.randn(bsz, 1, OUT_CHANS, HEIGHT, WIDTH)
    times = torch.tensor([946684800000000000, 946684800000000000])

    model = PuYunModel(**_model_kwargs())
    model.train()

    loss = model(
        {
            "inputs": inputs,
            "targets": targets,
            "times": times,
        }
    )

    assert "loss_rec" in loss


def test_puyun_test_cascade():
    bsz = 1
    inputs = torch.randn(bsz, IN_FRAMES, IN_CHANS, HEIGHT, WIDTH)
    times = torch.tensor([946684800000000000])

    model = PuYunModel(**_model_kwargs())
    model.eval()

    data = {
        "inputs": inputs,
        "times": times,
    }
    data = model.process_data(data)
    out = model.test_cascade(data)

    assert out.shape[1] == len(model.test_frames)
    assert out.shape[-2:] == (HEIGHT, WIDTH)
