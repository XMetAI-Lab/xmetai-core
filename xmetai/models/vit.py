'''
Author: DouZesheng && douzsh@gmail.com
Date: 2025-08-28 18:05:44
Description: Simple ViT Forecast model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved. 
'''
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from xmetai.builder import MODELS
from xmetai.loss import charbonnier_loss
from xmetai.layers import CubeEmbed, TimestepEmbed
try:
    from xmetai.layers import MambaVisionLayer
except Exception:
    MambaVisionLayer = None
    print("Warning: MambaVisionLayer is not available. Please install mamba-ssm.")
from .base_predictor import BasePredictor


@MODELS.register()
class ViTModel(BasePredictor):
    def __init__(
        self,
        embed_dim=256,
        depths=[6, 6, 12, 12],
        num_heads=8,
        patch_size=8,
        image_size=(120, 240),  # resize from 121x240->120x240 with input data
        drop_rate=0.1,
        mlp_ratio=4,
        use_mamba=True,
        freq=24,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.image_size = tuple(image_size)
        self.feat_size = [sz // patch_size for sz in image_size]
        self.freq = freq
        self.logger.name = __name__
        # Patch embedding
        self.patch_embed = CubeEmbed(
            in_chans=self.in_chans,
            out_chans=embed_dim,
            in_frames=self.in_frames,
            patch_size=patch_size,
            norm_layer=norm_layer,
            flatten=True,
        )
        self.step_embed = TimestepEmbed(embed_dim)
        self.hour_embed = TimestepEmbed(embed_dim)
        self.doy_embed = TimestepEmbed(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * 2)
        )

        # Transformer encoder with TransformerEncoderLayer
        blocks = []
        # 定义一个标准 TransformerEncoderLayer
        if not use_mamba or MambaVisionLayer is None:   # original ViT style
            self.logger.info("Using original ViT style transformer blocks.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*mlp_ratio,
                dropout=drop_rate,
                activation=F.gelu,
                batch_first=True    # 让输入是 (B, N, C)，和常见 ViT 格式一致
            )
            for depth in depths:
                blocks.append(
                    nn.TransformerEncoder(encoder_layer, num_layers=depth)
                )
                blocks.append(norm_layer(embed_dim))
        else:   # Mamba style
            self.logger.info("Using Mamba style transformer blocks.")
            for _, depth in enumerate(depths):
                block = MambaVisionLayer(dim=embed_dim,
                                         depth=depth,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=True,
                                         qk_scale=True,
                                         conv=False,
                                         drop=drop_rate,
                                         transformer_blocks=list(range(depth//2+1, depth)) if depth % 2 != 0 else list(range(depth//2, depth)),
                                         )
                blocks.append(block)
        self.encoder = nn.Sequential(*blocks)

        self.logger.info(f"Using 2fc head with embed_dim: {embed_dim}, out_chans: {self.out_chans}, patch_size: {patch_size}")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.out_chans * patch_size * patch_size)
        )
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 or name.endswith("weight"):
                nn.init.trunc_normal_(param, std=0.02)
            else:
                nn.init.zeros_(param)

    def forward(self, data):
        if not self.training:
            return self.test(data, freq=self.freq)

        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        targets = data["targets"]
        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")  # input: bx2x76x121x240, target: bx1x76x121x240
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames

        loss_rec = 0

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k:k+self.out_frames]
            self.reset_input(input)
            output = self.forward_step(input, times, step=t)
            loss_rec += charbonnier_loss(output, target)
            self.logger.debug(f"loss_rec: {loss_rec.item()} with future frames {future_frames}.")
            # prepare next input with current output
            input = torch.cat([input, output], dim=1)[:, -self.in_frames:]

        loss_dict = dict(loss_rec=loss_rec / future_frames)
        return loss_dict

    def forward_step(self, input, times=None, step=None, timestamp=None):
        # time embeddings.
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step, freq=self.freq)
        self.logger.debug(f"timestamp: {timestamp}")
        time_token = self.step_embed(timestamp['step']).unsqueeze(1)  # Bx1xC
        time_token = torch.concat([time_token, self.hour_embed(timestamp['hour']).unsqueeze(1)], dim=1)
        time_token = torch.concat([time_token, self.doy_embed(timestamp['doy']).unsqueeze(1)], dim=1)
        self.logger.debug(f"time embed shape: {time_token.shape}, min: {time_token.min().item()}, max: {time_token.max().item()}")  # Bx3xC

        # start forward calculation
        x = rearrange(input, "n t c h w -> n (t c) h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)
        h = self.patch_embed(x)  # B, h*w/p/p, C
        self.logger.debug(f"patch embed output shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")

        # apply time embeddings. h is B x N x C, time_token is Bx3xC
        time_token = self.time_mlp(time_token.sum(dim=1))
        gamma, beta = time_token.chunk(2, dim=-1)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        self.logger.debug(f"hidden shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")

        h = self.encoder(h)
        self.logger.debug(f"encoder output shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")
        output = self.head(h)  # B, N, C_out*P*P
        self.logger.debug(f"head output shape: {output.shape}")
        output = rearrange(output, "n (h w) (c p1 p2) -> n c (h p1) (w p2)",
                           h=self.feat_size[0],
                           p1=self.patch_size,
                           p2=self.patch_size,
                           )
        if output.shape[-2:] != input.shape[-2:]:
            output = F.interpolate(output.float(), size=input.shape[-2:], mode="bilinear", align_corners=False,).to(input)
        # decode output is only a residual, so we add the last input frame
        self.logger.debug(f"final output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")
        output = output.unsqueeze(1) + input[:, -1:]
        return output

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
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
                output = self.forward_step(input, times, step=step+self.start_frames)
                input = torch.cat([input, output], dim=1)[:, -self.in_frames:]
                # save for output
                if step+1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step:curr_step + 1, :self.out_chans] += output
                    curr_step += 1

        return outputs

    @torch.no_grad()
    def export_onnx(self, input, step, hour, doy):
        # normalize input with float32 precision
        normed_input = self.normalize(input)

        # convert all inputs to model dtype
        normed_input = normed_input.to(dtype=self.dtype)
        step = step.to(dtype=self.dtype)
        hour = hour.to(dtype=self.dtype)
        doy = doy.to(dtype=self.dtype)

        # forward model
        self.reset_input(normed_input)
        batch_size = input.shape[0]
        timestamp = {
            'step': step.repeat(batch_size),
            'hour': hour.repeat(batch_size),
            'doy': doy.repeat(batch_size),
        }
        output = self.forward_step(normed_input, timestamp=timestamp)

        # inv normalize output with float32 precision
        output = self.inv_normalize(output)
        output = torch.concat([input[:, -1:], output], dim=1)  # add the first input frame for output
        return output
