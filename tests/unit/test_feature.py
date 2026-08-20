"""FeatureEncoder 单元测试(frames + availability → tokens;CPU,无需 NATTEN)。"""

from __future__ import annotations

import torch

from xmetai.layers.async_multimodal import FeatureEncoder

DIM = 32


class TestFeatureEncoder:
    def test_encode_shape(self):
        enc = FeatureEncoder("coarse", in_ch=6, dim=DIM, patch_size_hw=(2, 2), missing_token_learnable=True)
        frames = torch.randn(2, 3, 8, 8, 6)  # (B, K, H, W, C)
        avail = torch.ones(2, 3, dtype=torch.bool)
        out = enc(frames, avail)
        assert out.shape == (2, 3, 4, 4, DIM), f"got {out.shape}"

    def test_null_when_unavailable(self):
        enc = FeatureEncoder("coarse", in_ch=6, dim=DIM, missing_token_learnable=True)
        frames = torch.randn(2, 1, 8, 8, 6)
        out_avail = enc(frames, torch.ones(2, 1, dtype=torch.bool))
        out_null = enc(frames, torch.zeros(2, 1, dtype=torch.bool))
        assert not torch.allclose(out_avail, out_null)
        assert torch.allclose(out_null[0], out_null[1])  # null 在 batch 内一致

    def test_no_null_when_disabled(self):
        # missing_token_learnable=False:缺失位置用 zeros 覆盖(不漏 patch 垃圾)
        enc = FeatureEncoder("coarse", in_ch=6, dim=DIM, missing_token_learnable=False)
        frames = torch.randn(2, 1, 8, 8, 6)
        out_avail = enc(frames, torch.ones(2, 1, dtype=torch.bool))  # feat
        out_missing = enc(frames, torch.zeros(2, 1, dtype=torch.bool))  # zeros(覆盖)
        assert not torch.allclose(out_avail, out_missing)  # feat vs zeros
        assert torch.all(out_missing == 0)  # 缺失 = zeros
