"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-29 08:44:19
Description: Head module for decoding PLSF and SFC features.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from xmetai.layers import EarthConv2d, kaming_init

__all__ = ["PLSFHead"]


class _ResizeConv2d(nn.Module):
    """Longitude-periodic resize with channel mixing on the coarse grid."""

    def __init__(self, in_chans: int, out_chans: int, scale: int):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_chans, out_chans, 1, bias=False)
        self.spatial = EarthConv2d(out_chans, out_chans, 3, padding=1, groups=out_chans, bias=False)
        # Start with interpolation alone; learn spatial refinement without a
        # dense full-resolution channel-mixing convolution.
        nn.init.dirac_(self.spatial.conv.weight, groups=out_chans)

    def forward(self, h):
        h = self.proj(h)
        # Interpolation must mix the last and first longitude tokens too.
        h = F.pad(h, (1, 1, 0, 0), mode="circular")
        h = F.interpolate(h.float(), scale_factor=self.scale, mode="bilinear", align_corners=False).to(h.dtype)
        h = h[..., self.scale : -self.scale]
        return self.spatial(h)


class PLSFHead(nn.Module):
    def __init__(
        self,
        in_chans: int,
        pl_chans: int = 65,
        sfc_chans: int = 13,
        patch_size: tuple = (2, 2),
        feat_size: list = [120, 240],
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
        self.pl_chans = pl_chans
        self.sfc_chans = sfc_chans
        self.feat_size = feat_size

        if upsample_mode == "transpose":
            self.pl_head = kaming_init(nn.ConvTranspose2d(in_chans, pl_chans, patch_size[0], patch_size[1]))
            self.sf_head = kaming_init(nn.ConvTranspose2d(in_chans, sfc_chans, patch_size[0], patch_size[1]))
        elif upsample_mode == "resize":
            if patch_size[0] != patch_size[1] or patch_size[0] < 1 or min(pl_chans, sfc_chans) < 1:
                raise ValueError("Resize PLSFHead requires equal positive kernel/stride and nonempty PL/surface branches")
            self.pl_head = _ResizeConv2d(in_chans, pl_chans, patch_size[0])
            self.sf_head = _ResizeConv2d(in_chans, sfc_chans, patch_size[0])
        else:
            raise ValueError(f"Unknown PLSFHead upsample_mode: {upsample_mode!r}")

    def forward(self, h, need_rearrange=True):
        """
        input shape: [b, h*w, in_chans]
        for residual output shape: [b, h*w, (pl_chans + sfc_chans)*p1*p2]
        for residual hidden shape: [b, pl_chans + sfc_chans, h*p1, w*p2]
        """
        if need_rearrange:
            h = rearrange(h, "n (h w) c -> n c h w", h=self.feat_size[0])
        pl = self.pl_head(h)
        sf = self.sf_head(h)
        out = torch.cat([pl, sf], dim=1)
        if need_rearrange:
            out = rearrange(
                out,
                "n c (h p1) (w p2)-> n (h w) (c p1 p2)",
                h=self.feat_size[0],
                w=self.feat_size[1],
            )
        return out
