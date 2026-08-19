"""
Author: XMetAI
Date: 2026-08-19
Description: FuXi-2.1 weather forecasting model.

A project-native, downscaled reimplementation of the FuXi-2.1 architecture
(flat single-trunk Transformer with Swin window attention + 1-D RoPE + adaLN
time conditioning + SwiGLU FFN and a variable-aware multi-head decoder),
trained on the project's own 1.5-degree dataset. Official 0.25-degree weights
are not loaded; the architecture is reproduced so the model trains from
scratch on any channel layout.

Refer to: https://huggingface.co/tpys/fuxi-2.1
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.layers.fuxi_blocks import FuxiConstEmbed, FuxiPatchEmbed, FuxiSwinBlock, FuxiTimeEmbed, FuxiVarHead
from xmetai.layers.helpers import to_2tuple
from xmetai.layers.norm import AdaptiveLN
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

from .base_predictor import BasePredictor

__all__ = ["Fuxi2", "split_var_groups"]

# Pressure-level variable short names (channel names look like "z500", "t850", "q925").
PL_NAMES = ("z", "t", "u", "v", "q")
# Derived / accumulated diagnostic channels (predicted but zeroed before feedback).
DERIVED_NAMES = ("ssr", "ssrd", "fdir", "ttr", "tp")

_PL_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")


def split_var_groups(in_names: List[str]) -> List[List[int]]:
    """Partition channel indices into ``[pressure, surface, derived]`` groups.

    - pressure: ``{z,t,u,v,q}<level>`` channels (e.g. ``z500``)
    - derived: accumulated radiation / precipitation channels (``ssr``, ``tp``, ...)
    - surface: everything else (``msl``, ``t2m``, ``sst``, ...)

    The grouping is driven entirely by ``in_names`` so that extending the
    dataset with more surface variables requires no model-code changes.
    """
    pl, sfc, der = [], [], []
    for i, name in enumerate(in_names):
        m = _PL_RE.match(name)
        if m and m.group(1) in PL_NAMES:
            pl.append(i)
        elif name in DERIVED_NAMES:
            der.append(i)
        else:
            sfc.append(i)
    return [pl, sfc, der]


class FuxiBackbone(nn.Module):
    """Patch embedding -> flat Swin trunk (adaLN + SwiGLU) -> variable-aware decoder."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        in_frames: int,
        const_chans: int,
        embed_dim: int,
        image_size: Tuple[int, int],
        patch_size: int,
        window_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        drop_rate: float,
        drop_path_rate: float,
        group_indices: List[List[int]],
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.in_frames = in_frames
        self.const_chans = const_chans
        self.embed_dim = embed_dim
        self.image_size = tuple(image_size)
        self.patch_size = patch_size

        patch_2d = to_2tuple(patch_size)
        self.feat_size = (self.image_size[0] // patch_2d[0], self.image_size[1] // patch_2d[1])

        self.patch_embed = FuxiPatchEmbed(in_chans, in_frames, embed_dim, patch_size)
        if const_chans > 0:
            self.const_embed = FuxiConstEmbed(const_chans, embed_dim, patch_size)
        else:
            self.const_embed = None
        self.joint_embed_layer = FuxiTimeEmbed(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                FuxiSwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    input_size=self.feat_size,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm_layer = AdaptiveLN(embed_dim)
        self.pred_layer = FuxiVarHead(embed_dim, group_indices, patch_size, self.feat_size)

    def resize_input(self, input: torch.Tensor) -> torch.Tensor:
        x = rearrange(input, "n t c h w -> n (t c) h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)
        return x

    def resize_const(self, const: torch.Tensor) -> torch.Tensor:
        if const.ndim == 3:
            const = const.unsqueeze(0)
        if const.shape[-2:] != self.image_size:
            const = F.interpolate(const.float(), size=self.image_size, mode="bilinear", align_corners=False).to(const)
        return const

    def forward(
        self,
        input: torch.Tensor,
        const: Optional[torch.Tensor] = None,
        timestamp: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        input = input[:, -self.in_frames :]
        x = self.resize_input(input)
        if const is not None:
            const = self.resize_const(const)

        tokens = self.patch_embed(x)
        if const is not None and self.const_embed is not None:
            tokens = tokens + self.const_embed(const)
        time_embed = self.joint_embed_layer(timestamp) if timestamp is not None else None
        for blk in self.layers:
            tokens = blk(tokens, time_embedding=time_embed)
        tokens = self.norm_layer(tokens, time_embed)

        output = self.pred_layer(tokens)  # (B, out_chans, image_size)

        if output.shape[-2:] != input.shape[-2:]:
            output = F.interpolate(output.float(), size=input.shape[-2:], mode="bilinear", align_corners=False).to(input)

        return output.unsqueeze(1)  # (B, 1, out_chans, H, W)


@MODELS.register()
class Fuxi2(BasePredictor):
    """Autoregressive FuXi-2.1 predictor with the project's training/eval API."""

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 12,
        window_size: int = 6,
        mlp_ratio: float = 8 / 3,
        patch_size: int = 2,
        image_size: Tuple[int, int] = (120, 240),
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        loss_type: str = "mse",
        const_chans: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.loss_type = loss_type
        self.const_chans = const_chans

        if self.in_names is None:
            raise ValueError(
                "Fuxi2 requires `in_names` to split channels into pressure/surface/derived groups; "
                "pass `in_names=...` in the model config."
            )
        group_indices = split_var_groups(self.in_names)
        self.group_indices = group_indices
        total = sum(len(g) for g in group_indices)
        if total != self.out_chans:
            raise ValueError(
                f"Fuxi2 group split covers {total} channels but out_chans={self.out_chans}; "
                f"pressure={len(group_indices[0])}, surface={len(group_indices[1])}, derived={len(group_indices[2])}."
            )
        self.logger.info(
            f"Fuxi2 variable groups: pressure={len(group_indices[0])}, "
            f"surface={len(group_indices[1])}, derived={len(group_indices[2])}"
        )

        # Named `decoder` to match the official FuXi-2.1 .pt2 state_dict (`decoder.*`).
        self.decoder = FuxiBackbone(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            in_frames=self.in_frames,
            const_chans=const_chans,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            group_indices=group_indices,
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
            self.reset_input(input)
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

        if self.loss_type == "mae":
            loss_rec = torch.abs(output - target)
            if weight is not None:
                loss_rec = loss_rec * weight
            loss_rec = loss_rec.mean()
        elif self.loss_type == "mse":
            loss_rec = (output - target) ** 2
            if weight is not None:
                loss_rec = loss_rec * weight
            loss_rec = loss_rec.mean()
        else:
            loss_rec = charbonnier_loss(output, target, weight=weight)

        step_losses = {"loss_rec": loss_rec}
        if self.pinn_enable:
            for k, v in self.compute_pinn_losses(output, target).items():
                step_losses[k] = v
        return step_losses

    def forward_step(self, input, const=None, times=None, step=None, timestamp=None):
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step)
        self.logger.debug(f"timestamp: {timestamp}")
        output = self.decoder(input, const=const, timestamp=timestamp)
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
                self.reset_input(input)
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
