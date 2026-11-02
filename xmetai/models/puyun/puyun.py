"""
Author: Hujiahui && ltx1125919730@gmail.com
Date: 2026-06-22 18:05:44
Description: PuYun weather forecasting model.

This module implements a project-native version of PuYun: a fully
convolutional autoregressive predictor based on large-kernel attention.
Refer to: https://arxiv.org/abs/2409.02123
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.core.base_predictor import BasePredictor
from xmetai.layers.embed import TimestepEmbed
from xmetai.layers.helpers import to_2tuple
from xmetai.layers.puyun_blocks import ConvLayerNorm, PuYunBlock
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

__all__ = ["PuYunModel"]


class PuYunBackbone(nn.Module):
    """PuYun LKA-FCN backbone, adapted from https://arxiv.org/abs/2409.02123."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        in_frames: int = 2,
        out_frames: int = 1,
        const_chans: int = 0,
        embed_dim: int = 256,
        image_size: Tuple[int, int] = (120, 240),
        patch_size: int = 2,
        depths: List[int] = [6, 6, 6, 6],
        kernel_size: int = 9,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_pixel_shuffle: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.const_chans = const_chans
        self.embed_dim = embed_dim
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.depths = list(depths)
        self.use_pixel_shuffle = use_pixel_shuffle

        patch_size_2d = to_2tuple(patch_size)
        patch_in_chans = in_chans * in_frames + const_chans
        self.patch_embed = nn.Sequential(
            nn.Conv2d(patch_in_chans, embed_dim, kernel_size=patch_size_2d, stride=patch_size_2d),
            ConvLayerNorm(embed_dim),
        )

        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_offset = 0
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for depth in self.depths:
            blocks = []
            for i in range(depth):
                blocks.append(
                    PuYunBlock(
                        dim=embed_dim,
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        drop_rate=drop_rate,
                        drop_path=dpr[dpr_offset + i],
                    )
                )
            self.layers.append(nn.Sequential(*blocks))
            self.layer_norms.append(ConvLayerNorm(embed_dim))
            dpr_offset += depth

        merge_in_chans = embed_dim * (len(self.depths) + 1)
        merge_out_chans = out_chans * out_frames * patch_size * patch_size
        self.patch_merging = nn.Sequential(
            nn.Conv2d(merge_in_chans, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, merge_out_chans, kernel_size=1),
        )

        self.step_embed = TimestepEmbed(embed_dim)
        self.hour_embed = TimestepEmbed(embed_dim)
        self.doy_embed = TimestepEmbed(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def get_time_embed(self, timestamp: Dict[str, torch.Tensor]) -> torch.Tensor:
        embed = self.step_embed(timestamp["step"])
        if "hour" in timestamp:
            embed = embed + self.hour_embed(timestamp["hour"])
        if "doy" in timestamp:
            embed = embed + self.doy_embed(timestamp["doy"])
        return self.time_mlp(embed)

    def resize_input(self, input: torch.Tensor) -> torch.Tensor:
        x = rearrange(input, "n t c h w -> n (t c) h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)
        return x

    def resize_output(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        if self.use_pixel_shuffle:
            output = F.pixel_shuffle(output, self.patch_size)
        else:
            output = F.interpolate(output.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)

        if output.shape[-2:] != input.shape[-2:]:
            output = F.interpolate(
                output.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)

        output = rearrange(output, "n (t c) h w -> n t c h w", t=self.out_frames)
        return output

    def forward(
        self,
        input: torch.Tensor,
        const: Optional[torch.Tensor] = None,
        timestamp: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        input = input[:, -self.in_frames :]
        resized_input = self.resize_input(input)

        if const is not None:
            if const.ndim == 3:
                const = const.unsqueeze(0)
            if const.size(0) == 1 and resized_input.size(0) > 1:
                const = const.repeat(resized_input.size(0), 1, 1, 1)
            if const.shape[-2:] != self.image_size:
                const = F.interpolate(const.float(), size=self.image_size, mode="bilinear", align_corners=False).to(resized_input)
            resized_input = torch.cat([resized_input, const], dim=1)

        patch = self.patch_embed(resized_input)
        if timestamp is not None:
            time_embed = self.get_time_embed(timestamp).unsqueeze(-1).unsqueeze(-1)
            patch = patch + time_embed
        features = [patch]
        h = patch
        for layer, norm in zip(self.layers, self.layer_norms):
            h = layer(h)
            features.append(norm(h))

        h = torch.cat(features, dim=1)
        output = self.patch_merging(h)
        return self.resize_output(output, input)


@MODELS.register()
class PuYunModel(BasePredictor):
    """PuYun (https://arxiv.org/abs/2409.02123) adapted to XMetAI training and evaluation."""

    def __init__(
        self,
        embed_dim: int = 256,
        depths: List[int] = [6, 6, 6, 6],
        image_size: Tuple[int, int] = (120, 240),
        patch_size: int = 2,
        kernel_size: int = 9,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        loss_type: str = "charbonnier",
        use_pixel_shuffle: bool = True,
        const_chans: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.loss_type = loss_type
        self.const_chans = const_chans

        self.backbone = PuYunBackbone(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            in_frames=self.in_frames,
            out_frames=self.out_frames,
            const_chans=const_chans,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            depths=depths,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_pixel_shuffle=use_pixel_shuffle,
        )

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]
        targets = data["targets"]
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames
        loss_dict = {}

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            input = self.reset_input(input)
            output = self.forward_step(input, const=const, times=times, step=t)
            step_losses = self.compute_step_loss(output, target, step=k, future_frames=future_frames)
            for k_loss, v_loss in step_losses.items():
                loss_dict[k_loss] = loss_dict.get(k_loss, 0) + v_loss
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        for k_loss, v_loss in loss_dict.items():
            loss_dict[k_loss] = v_loss / future_frames
        return loss_dict

    def compute_step_loss(self, output, target, step, future_frames=1):
        weight = getattr(self, "weight", None)
        if "channel_weights" in self._buffers and weight is not None:
            weight = self.channel_weights * weight

        if self.loss_type == "mae" or (self.loss_type == "puyun" and future_frames <= 1):
            loss_rec = torch.abs(output - target)
            if weight is not None:
                loss_rec = loss_rec * weight
            loss_rec = loss_rec.mean()
        elif self.loss_type == "mse" or (self.loss_type == "puyun" and future_frames > 1):
            loss_rec = (output - target) ** 2
            if weight is not None:
                loss_rec = loss_rec * weight
            loss_rec = loss_rec.mean()
        else:
            loss_rec = charbonnier_loss(output, target, weight=weight)

        step_losses = {"loss_rec": loss_rec}
        if self.pinn_enable:
            pinn_losses = self.compute_pinn_losses(output, target)
            for k, v in pinn_losses.items():
                step_losses[k] = v
        return step_losses

    def forward_step(self, input, const=None, times=None, step=None, timestamp=None):
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step)
        self.logger.debug(f"timestamp: {timestamp}")
        output = self.backbone(input, const=const, timestamp=timestamp)
        output = output + input[:, -self.out_frames :, : self.out_chans]
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
        outputs = input.new_zeros(input.shape[0], len(steps), *chw)
        self.logger.debug(f"output shape is {outputs.shape}")

        for num_step in self.step_range:
            curr_step = 0
            for step in range(num_step):
                input = self.reset_input(input)
                output = self.forward_step(input, const=const, times=times, step=step + self.start_frames)
                input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                if step + 1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                    curr_step += 1

        return outputs

    @torch.no_grad()
    def export_onnx(self, data, step, hour, doy):
        self.logger.info(f"export model with device:{self.device} and dtype:{self.dtype}")
        data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))
        normed_input = self.normalize(data)
        normed_input = normed_input.to(dtype=self.dtype)
        if isinstance(step, torch.Tensor) and step.is_floating_point():
            step = step.to(dtype=self.dtype)
        if isinstance(hour, torch.Tensor) and hour.is_floating_point():
            hour = hour.to(dtype=self.dtype)
        if isinstance(doy, torch.Tensor) and doy.is_floating_point():
            doy = doy.to(dtype=self.dtype)

        const = self._buffers.get("const", None)
        if self.const_chans > 0 and const is not None:
            const = const.to(dtype=self.dtype, device=normed_input.device)
        else:
            const = None

        new_input = self.reset_input(normed_input)
        timestamp = dict(
            step=step.to(new_input),
            hour=hour.to(new_input),
            doy=doy.to(new_input),
        )

        output = self.forward_step(new_input, const=const, timestamp=timestamp)
        output = self.inv_normalize(output)
        return output
