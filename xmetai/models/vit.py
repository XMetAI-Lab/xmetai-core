"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-08-28 18:05:44
Description: Simple ViT Forecast model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from xmetai.builder import MODELS
from xmetai.layers import set_lora_step
from xmetai.loss import charbonnier_loss

from .base_predictor import BasePredictor
from .utransformer import UTransformer


@MODELS.register()
class PerturbUMamba(BasePredictor):
    def __init__(
        self,
        embed_dim=256,
        depths=[6, 6, 12, 12],
        num_heads=8,
        patch_size=8,
        image_size=(120, 240),  # resize from 121x240->120x240 with input data
        window_size=8,
        drop_rate=0.1,
        mlp_ratio=4,
        layer_type="mamba_embed",
        const_chans=0,
        *args,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.image_size = tuple(image_size)
        self.const_chans = const_chans
        self.logger = logging.getLogger(__name__)

        # Perturbation settings (Mixup and Random Masking)
        self.pertub_mixup = kwargs.pop("pertub_mixup", 0.0)  # max mixup alpha
        self.pertub_mask = kwargs.pop("pertub_mask", 0.0)  # spatial dropout ratio
        self.use_pertub = kwargs.pop("use_pertub", False)  # enable at inference

        super().__init__(*args, **kwargs)

        # Uses UTransformer as the main backbone
        self.encoder = UTransformer(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            in_frames=self.in_frames,
            embed_dim=embed_dim,
            image_size=image_size,
            const_chans=const_chans,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            drop_mcmc=drop_rate,
            mlp_ratio=mlp_ratio,
            layer_type=layer_type,
            attn_type="v2",
            mask_type="h",
            head_type="2fc",
        )
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            # Respect parameters that shouldn't be reinitialized
            if getattr(param, "_no_reinit", False) or getattr(param, "_no_weight_decay", False):
                continue

            if "norm" in name.lower() or "ln" in name.lower() or "layernorm" in name.lower():
                if name.endswith("weight") and param.dim() == 1:
                    nn.init.ones_(param)
                elif name.endswith("bias") and param.dim() == 1:
                    nn.init.zeros_(param)
            else:
                if param.dim() > 1:
                    nn.init.trunc_normal_(param, std=0.02)
                elif name.endswith("bias") and param.dim() == 1:
                    nn.init.zeros_(param)

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        targets = data["targets"]
        const = data.get("const", None) if self.const_chans > 0 else None
        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")  # input: bx2x76x121x240, target: bx1x76x121x240
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames

        loss_dict = {}

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            self.reset_input(input)
            output = self.forward_step(input, const=const, times=times, step=t)
            step_losses = self.compute_step_loss(output, target, step=k)
            for k_loss, v_loss in step_losses.items():
                loss_dict[k_loss] = loss_dict.get(k_loss, 0) + v_loss

            self.logger.debug(f"step {k} loss_rec: {step_losses.get('loss_rec', 0)} with future frames {future_frames}.")
            # prepare next input with current output
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        for k_loss, v_loss in loss_dict.items():
            loss_dict[k_loss] = v_loss / future_frames
        return loss_dict

    def compute_step_loss(self, output, target, step):
        loss_rec = charbonnier_loss(output, target, weight=getattr(self, "weight", None))
        step_losses = {"loss_rec": loss_rec}
        # accumulate PINN losses per-step when enabled
        if self.pinn_enable:
            pinn_losses = self.compute_pinn_losses(output, target)
            for k, v in pinn_losses.items():
                step_losses[k] = v
        return step_losses

    def forward_step(self, input, const=None, times=None, step=None, timestamp=None):
        # Setup LoRA step for multi-step LoRA (cascade mode)
        if step is not None:
            # We defer lora step setting inside the underlying model
            set_lora_step(self.encoder, step)

        # time embeddings.
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step)
        self.logger.debug(f"timestamp: {timestamp}")

        # start forward calculation using utransformer capabilities
        input = input[:, -self.in_frames :]  # bxTxcxhxw
        batch_size = input.shape[0]
        resized_input = self.encoder.resize_input(input)  # resize to target size
        resized_input = rearrange(resized_input, "(n t) c h w -> n t c h w", n=batch_size)

        # Cube embedding
        h = self.encoder.patch_embed(resized_input, const)

        # timestamp embedding
        embed = self.encoder.get_embed(timestamp)

        # Iterate over transformer layers
        num_layers = len(self.encoder.layers)
        mid_layer = num_layers // 2
        for i, blk in enumerate(self.encoder.layers):
            # ---------- Inject Perturbations in the middle layer ----------
            if i == mid_layer and (self.training or self.use_pertub):
                # 1. Feature Space Mixup (Batch Mixup / Manifold Mixup)
                if self.pertub_mixup > 0 and batch_size > 1:
                    alpha = torch.rand(1, device=h.device) * self.pertub_mixup
                    indices = torch.randperm(batch_size, device=h.device)
                    h = h * (1 - alpha) + h[indices] * alpha

                # 2. Random Token Masking (Spatial Dropout)
                if self.pertub_mask > 0:
                    # Random mask on the sequence length dimension (N), shape: (B, N, 1)
                    mask = (torch.rand(batch_size, h.shape[1], 1, device=h.device) > self.pertub_mask).to(h.dtype)
                    h = h * mask / (1 - self.pertub_mask)
            # --------------------------------------------------------------

            h = getattr(self.encoder, f"drop{i}")(h, self.encoder.drop_mcmc)
            # handle time-embedding for layers correctly as UTransformer expects
            if "rotary" in self.encoder.attn_type:
                h = blk(
                    h,
                    time_embedding=embed,
                    freqs_cos=self.encoder.freqs_cos,
                    freqs_sin=self.encoder.freqs_sin,
                )
            else:
                h = blk(h, time_embedding=embed)

        if hasattr(self.encoder, "fpn") and self.encoder.fpn is not None:
            # Just in case we expand UTransformer later to use feature pyramids
            pass

        if hasattr(self.encoder, "norm") and self.encoder.norm is not None:
            h = self.encoder.norm(h, embed)

        output = self.encoder.head(h)  # B, N, C_out*P*P
        self.logger.debug(f"head output shape: {output.shape}")

        output = self.encoder.resize_output(output, input)

        # decode output is only a residual, so we add the last input frame
        self.logger.debug(f"final output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")
        output = output.unsqueeze(1) + input[:, -1:]
        return output

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None
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
                output = self.forward_step(input, const=const, times=times, step=step + self.start_frames)
                input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                # save for output
                if step + 1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
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

        batch_size = input.shape[0]
        const = None
        if self.const_chans > 0 and hasattr(self, "const"):
            const = self.const.to(input)
            const = torch.repeat_interleave(const.unsqueeze(0), repeats=batch_size, dim=0)

        # forward model
        self.reset_input(normed_input)
        timestamp = {
            "step": step.repeat(batch_size),
            "hour": hour.repeat(batch_size),
            "doy": doy.repeat(batch_size),
        }
        output = self.forward_step(normed_input, const=const, timestamp=timestamp)

        # inv normalize output with float32 precision
        output = self.inv_normalize(output)
        output = torch.concat([input[:, -1:], output], dim=1)  # add the first input frame for output
        return output
