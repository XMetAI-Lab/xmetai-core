"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-11-03 08:05:44
Description: ViT model with ensemble strategy.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.builder import MODELS
from xmetai.layers import DropPath, set_lora_step
from xmetai.loss import afcrps_loss_lite, charbonnier_loss

from .vit import ViTModel


def get_exponential_weights(steps: int, alpha: float) -> torch.Tensor:
    """
    Generate normalized exponential weights for sequence loss.

    Args:
        steps (int): number of timesteps (e.g., future frames).
        alpha (float): exponential factor (>1 = emphasize later, <1 = emphasize earlier).

    Returns:
        torch.Tensor: normalized weights summing to 1.
    """
    exponents = torch.arange(steps, dtype=torch.float32)
    raw = torch.pow(alpha, exponents)
    weights = raw / raw.sum()
    return weights


def noise_magnitude_loss(delta_h, min_std=0.1, lambda_mag=0.1):
    """Ensure that noise amplitude does not collapse."""
    std = delta_h.std(dim=-1)  # [B, pH*pW]
    loss = F.relu(min_std - std).pow(2).mean()
    return lambda_mag * loss


class EnsNoiseMLP(nn.Module):
    def __init__(self, embed_dim, init_scale=0.1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim),
            nn.Softsign(),
        )
        # learnable global scaling factor
        self.noise_scale = nn.Parameter(torch.full((1,), init_scale))
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization 适合对称激活（Tanh/Softsign）
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, noise):
        output = self.net(noise) * self.noise_scale.view(1, 1, 1)
        output = self.drop_path(output)
        return output


@MODELS.register()
class ViTEnsModel(ViTModel):
    def __init__(self, embed_dim=256, alpha=1.0, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.logger.name = __name__
        self.alpha = alpha
        self.ens_noise_mlp = EnsNoiseMLP(embed_dim)

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        targets = data["targets"]
        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")  # input: bx2x76x121x240, target: bx1x76x121x240
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames

        # get weights for each future frames
        rolling_weights = get_exponential_weights(steps=future_frames, alpha=self.alpha)
        self.logger.debug(f"rolling weights: {rolling_weights}")

        loss_rec = 0
        loss_crps = 0
        loss_noise = 0
        pinn_acc: dict[str, torch.Tensor] = {}

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            self.reset_input(input)
            output, delta_h = self.forward_step(input, times, step=t)
            if "channel_weights" in self._buffers:
                loss_rec += rolling_weights[k] * charbonnier_loss(output, target, weight=self.channel_weights * self.weight)
            else:
                loss_rec += rolling_weights[k] * charbonnier_loss(output, target, weight=self.weight)
            loss_crps += rolling_weights[k] * afcrps_loss_lite(output, target, weight=self.weight)
            loss_noise += noise_magnitude_loss(delta_h)
            if self.pinn_enable:
                pinn_losses = self.compute_pinn_losses(output, target)
                for k, v in pinn_losses.items():
                    pinn_acc[k] = pinn_acc.get(k, 0) + v

            # prepare next input with current output
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        loss_dict = dict(loss_rec=loss_rec, loss_crps=loss_crps, loss_noise=loss_noise)
        if len(pinn_acc) > 0:
            for k, v in pinn_acc.items():
                loss_dict[k] = v
        return loss_dict

    def forward_step(self, input, times=None, step=None, timestamp=None):
        # Setup LoRA step for multi-step LoRA (cascade mode)
        if step is not None:
            set_lora_step(self.encoder, step)
        # time embeddings.
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step)
        self.logger.debug(f"timestamp: {timestamp}")
        time_token = self.step_embed(timestamp["step"]).unsqueeze(1)  # Bx1xC
        time_token = torch.concat([time_token, self.hour_embed(timestamp["hour"]).unsqueeze(1)], dim=1)
        time_token = torch.concat([time_token, self.doy_embed(timestamp["doy"]).unsqueeze(1)], dim=1)
        self.logger.debug(f"time embed shape: {time_token.shape}, min: {time_token.min().item()}, max: {time_token.max().item()}")  # Bx3xC

        # start forward calculation
        x = rearrange(input, "n t c h w -> n (t c) h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)
        h = self.patch_embed(x)  # B, h*w/p/p, C
        self.logger.debug(f"patch embed output shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")

        # apply noise on h for ensemble
        noise = torch.randn_like(h)
        delta_h = self.ens_noise_mlp(noise)
        h = h + delta_h

        # apply time embeddings. h is B x N x C, time_token is Bx3xC
        time_token = self.time_mlp(time_token.sum(dim=1))
        gamma, beta = time_token.chunk(2, dim=-1)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        self.logger.debug(f"hidden shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")

        h = self.encoder(h)
        self.logger.debug(f"encoder output shape: {h.shape}, min: {h.min().item()}, max: {h.max().item()}")
        output = self.head(h)  # B, N, C_out*P*P
        self.logger.debug(f"head output shape: {output.shape}")
        output = rearrange(
            output,
            "n (h w) (c p1 p2) -> n c (h p1) (w p2)",
            h=self.feat_size[0],
            p1=self.patch_size,
            p2=self.patch_size,
        )
        if output.shape[-2:] != input.shape[-2:]:
            output = F.interpolate(
                output.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)
        # decode output is only a residual, so we add the last input frame
        self.logger.debug(f"final output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")
        output = output.unsqueeze(1) + input[:, -1:]
        if self.training:
            return output, delta_h
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
        self.logger.info(f"Start testing with {self.members} ensemble members and step_range {self.step_range} with time:{times}.")

        # for each member, we inference for step_range times. Each time we get num_step results.
        for member in range(self.members):
            self.logger.debug(f"inference with member:{member} for step_range:{self.step_range}.")
            new_input = input.clone()
            for num_step in self.step_range:
                curr_step = 0
                for step in range(num_step):
                    self.reset_input(new_input)
                    output = self.forward_step(new_input, times, step=step + self.start_frames)
                    new_input = torch.cat([new_input, output], dim=1)[:, -self.in_frames :]
                    # save for output
                    if step + 1 in steps:
                        self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                        outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                        curr_step += 1
        outputs /= self.members  # ensemble prediction with mean
        return outputs
