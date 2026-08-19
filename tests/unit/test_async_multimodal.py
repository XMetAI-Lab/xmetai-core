"""AsyncMultiModalModel smoke test(封装:frames + availability + time_coords + context → tokens;需 CUDA + NATTEN)。

验证上层封装:L3 给 frames(不碰 positions),内部切片 + encode + AsyncMultiModal 组装 + backbone。
"""

from __future__ import annotations

import torch

from xmetai.layers.async_multimodal import AsyncMultiModalModel, FeatureEncoder, Layout, SlotSpec
from xmetai.layers.natten3d_unet import Natten3DBackbone

DIM = 32
C_NOISE = 5
C_COARSE = 6
H = W = 16
SEQ = 9


def _has_natten_cuda() -> bool:
    try:
        import natten  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def _make_model():
    slots = [
        SlotSpec("noise", 0),
        SlotSpec("noise", 1),
        SlotSpec("noise", 2),
        SlotSpec("coarse", 3),
        SlotSpec("noise", 3),
        SlotSpec("noise", 4),
        SlotSpec("noise", 5),
        SlotSpec("coarse", 6),
        SlotSpec("noise", 6),
    ]  # 未排序;Layout 内部按 time 排
    layout = Layout(slots)
    encoders = [
        FeatureEncoder("noise", C_NOISE, DIM, (2, 2), missing_token_learnable=False),
        FeatureEncoder("coarse", C_COARSE, DIM, (2, 2), missing_token_learnable=True),
    ]
    backbone = Natten3DBackbone(
        levels=[
            dict(width=DIM, depth=2, d_ff=64, num_heads=2, kernel_size_thw=[3, 3, 3], use_global_attn=False),
            dict(width=DIM * 2, depth=2, d_ff=128, num_heads=2, kernel_size_thw=[3, 3, 3], use_global_attn=False),
        ],
        cond_dim=DIM,
        num_modalities=layout.num_modalities,
        dim=DIM,
        cosine_sim_scale=True,
    )
    return layout, AsyncMultiModalModel(layout, encoders, backbone, DIM)


def _make_frames(B=2):
    return {
        "noise": torch.randn(B, 7, H, W, C_NOISE),  # 7 noise slot
        "coarse": torch.randn(B, 2, H, W, C_COARSE),  # 2 coarse slot
    }


class TestAsyncMultiModalModel:
    def test_forward_full_sequence_and_backward(self):
        if not _has_natten_cuda():
            print("SKIP (no cuda/natten)")
            return
        dev = "cuda"
        layout, model = _make_model()
        model.to(dev).train()
        B = 2
        frames = {m: f.to(dev) for m, f in _make_frames(B).items()}
        availability = torch.ones(B, SEQ, dtype=torch.bool, device=dev)
        time_coords = layout.default_time_coords(B, dev)
        context = torch.zeros(B, DIM, H // 2, W // 2, device=dev)
        out = model(frames, availability, time_coords, context)
        assert out.shape == (B, SEQ, H // 2, W // 2, DIM), f"got {out.shape}"
        assert torch.isfinite(out).all()
        out.float().sum().backward()

    def test_rejects_wrong_availability_shape(self):
        if not _has_natten_cuda():
            print("SKIP (no cuda/natten)")
            return
        dev = "cuda"
        layout, model = _make_model()
        model.to(dev)
        B = 2
        frames = {m: f.to(dev) for m, f in _make_frames(B).items()}
        time_coords = layout.default_time_coords(B, dev)
        context = torch.zeros(B, DIM, H // 2, W // 2, device=dev)
        bad_avail = torch.ones(B, SEQ - 1, dtype=torch.bool, device=dev)
        try:
            model(frames, bad_avail, time_coords, context)
            assert False, "应报 ValueError"
        except ValueError:
            pass
