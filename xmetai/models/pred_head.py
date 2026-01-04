'''
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-29 08:44:19
Description: Head module for decoding PLSF and SFC features.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved. 
'''
import torch
from torch import nn
from einops import rearrange
from xmetai.layers import kaming_init

__all__ = ["PLSFHead"]


class PLSFHead(nn.Module):
    def __init__(
        self,
        in_chans: int,
        pl_chans: int = 65,
        sfc_chans: int = 13,
        patch_size: tuple = (2, 2),
        feat_size: list = [120, 240]
    ) -> None:
        super().__init__()
        self.pl_chans = pl_chans
        self.sfc_chans = sfc_chans
        self.feat_size = feat_size

        self.pl_head = kaming_init(nn.ConvTranspose2d(in_chans, pl_chans, patch_size[0], patch_size[1]))
        self.sf_head = kaming_init(nn.ConvTranspose2d(in_chans, sfc_chans, patch_size[0], patch_size[1]))

    def forward(self, h, need_rearrange=True):
        '''
        input shape: [b, h*w, in_chans]
        for residual output shape: [b, h*w, (pl_chans + sfc_chans)*p1*p2]
        for residual hidden shape: [b, pl_chans + sfc_chans, h*p1, w*p2]
        '''
        if need_rearrange:
            h = rearrange(h, 'n (h w) c -> n c h w', h=self.feat_size[0])
        pl = self.pl_head(h)
        sf = self.sf_head(h)
        out = torch.cat([pl, sf], dim=1)
        if need_rearrange:
            out = rearrange(out, 'n c (h p1) (w p2)-> n (h w) (c p1 p2)', h=self.feat_size[0], w=self.feat_size[1])
        return out
