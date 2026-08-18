from functools import partial

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.layers import AFNOBlock, AfnoPatchEmbed
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

from .base_predictor import BasePredictor

__all__ = ["AFNONet"]


@MODELS.register()
class AFNONet(BasePredictor):
    def __init__(
        self,
        embed_dim=768,
        img_size=(720, 1440),
        patch_size=(16, 16),
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans * self.in_frames, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(embed_dim, self.out_chans * self.patch_size[0] * self.patch_size[1], bias=False)

    def resize_input(self, input):
        x = rearrange(input, "n t c h w -> (n t) c h w")
        # 将 ListConfig 转换为普通 tuple[int]
        target_size = self.img_size
        if hasattr(target_size, "__len__"):
            target_size = tuple(int(s) for s in target_size)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x.float(), size=target_size, mode="bilinear", align_corners=False).to(input)
        return x

    def resize_output(self, output, input):
        output = rearrange(
            output,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        if output.shape[-2:] != input.shape[-2:]:
            output = F.interpolate(
                output.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)

        return output

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]
        targets = data["targets"]

        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")  # input: bx2x76x121x240, target: bx1x76x121x240
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames

        loss_dict = {}

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            self.reset_input(input)
            output = self.forward_step(input)
            step_losses = self.compute_step_loss(output, target)
            for k_loss, v_loss in step_losses.items():
                loss_dict[k_loss] = loss_dict.get(k_loss, 0) + v_loss

            self.logger.debug(f"step {k} loss_rec: {step_losses.get('loss_rec', 0)} with future frames {future_frames}.")
            # prepare next input with current output
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        for k_loss, v_loss in loss_dict.items():
            loss_dict[k_loss] = v_loss / future_frames
        return loss_dict

    def compute_step_loss(self, output, target):
        channel_weights = getattr(self, "channel_weights", None) if "channel_weights" in self._buffers else None
        weight = getattr(self, "weight", None)
        if channel_weights is not None and weight is not None:
            loss_weight = channel_weights * weight
        elif channel_weights is not None:
            loss_weight = channel_weights
        else:
            loss_weight = weight
        loss_rec = charbonnier_loss(output, target, weight=loss_weight)
        step_losses = {"loss_rec": loss_rec}
        # accumulate PINN losses per-step when enabled
        if self.pinn_enable:
            pinn_losses = self.compute_pinn_losses(output, target)
            for k, v in pinn_losses.items():
                step_losses[k] = v
        return step_losses

    def forward_step(self, input):

        input = input[:, -self.in_frames :]  # bxTxcxhxw
        B = input.shape[0]
        resized_input = self.resize_input(input)  # (b,t,c,h,w)
        resized_input = rearrange(resized_input, "(n t) c h w -> n (t c) h w", n=B)
        x = self.patch_embed(resized_input)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        x = self.head(x)
        output = self.resize_output(x, input)
        output = output.unsqueeze(1) + input[:, -1:]
        return output

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames
        assert sum(self.step_range) <= total_step

        chw = list(input.shape[-3:])
        outputs = input.new_zeros(input.shape[0], len(steps), *chw)  # batch_size, saved_steps, c,h,w
        self.logger.debug(f"output shape is {outputs.shape}")

        for num_step in self.step_range:
            curr_step = 0
            for step in range(num_step):
                self.reset_input(input)
                output = self.forward_step(input)
                input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                # save for output
                if step + 1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                    curr_step += 1

        return outputs
