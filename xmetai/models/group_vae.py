"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-03-06 12:00:00
Description: Level-wise VAE predictor with vertical cross-attention coupling.

Architecture (v2 -- reusing existing UTransformer-based encoder/decoder):
    - Per-level encoding via grouped GroupEncoder (UTransformer encoder)
    - LatentMixerPredictor: self-attention across spatial positions and pressure level groups
    - Per-level decoding via grouped GroupDecoder (UTransformer decoder with ConvTranspose2d head)

Level grouping (3 groups sharing encoder/decoder weights):
    - Group 0 (surface-coupled): 1000hPa only -- 16 input channels (5 PL + 11 surface)
    - Group 1 (lower-mid): 925, 850, 700, 600, 500, 400 hPa -- 5 channels each
    - Group 2 (upper): 300, 250, 200, 150, 100, 50 hPa -- 5 channels each

Channel layout (s2s 76-channel):
    PL vars: z,t,u,v,q  x 13 levels = 65 channels
    Surface vars: msl,t2m,sst,ttr,tp,ssr,ssrd,fdir,... = 11 channels

Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.builder import MODELS
from xmetai.layers import AdaptiveLN, EarthConvTranspose2d
from xmetai.loss import charbonnier_loss
from xmetai.models.base_predictor import BasePredictor
from xmetai.models.utransformer import UTransformer

__all__ = ["GroupVAEPredictor", "GroupEncoder", "GroupDecoder"]

LEVELS_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PL_VARS = ["z", "t", "u", "v", "q"]
NUM_PL_VARS = len(PL_VARS)
NUM_LEVELS = len(LEVELS_13)

DEFAULT_LEVEL_GROUPS = [[0], [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
RESIDUAL_VARS_24H = [
    # surface thermodynamics（强稳定）
    "sst",
    "t2m",
    "d2m",
    # near-surface slow evolution
    "t1000",
    "t925",
    # moisture（只保留最稳定的两层）
    "q1000",
    "q925",
    # stratosphere z（非常关键补充）
    "z50",
    "z100",
    "z150",
]
RESIDUAL_Z_LEVELS_6H = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700)
RESIDUAL_T_LEVELS_6H = (50, 100, 300, 400, 500, 600, 700, 850, 925, 1000)
RESIDUAL_VARS_6H = (
    [f"z{level}" for level in RESIDUAL_Z_LEVELS_6H]
    + [f"t{level}" for level in RESIDUAL_T_LEVELS_6H]
    + [
        # near-surface
        "t2m",
        "d2m",
        # moisture
        "q1000",
        # other slow variables
        "msl",
        "sst",
    ]
)


# 地表极其缓慢且强自相关的变量列表，作为独立的 Group 0（去高频噪声平滑）
SLOW_SFCS = ["sst"]


def build_level_channel_map(
    in_names: list[str],
    level_groups: list[list[int]],
    split_surface: bool = False,
) -> tuple[list[list[int]], list[int]]:
    """Build per-level and per-group channel index mappings.

    Args:
        in_names: list of input variable names
        level_groups: grouping of level indices, e.g. [[0], [1..6], [7..12]]
        split_surface: whether to split stable surface variables into a separate group

    Returns:
        group_slice_indices: list of channel indices for each group (for slicing input)
        group_reassemble_map: permutation indices mapping each original channel index
                              to its position in the concatenated grouped-channel order.
    """
    # Step 1: Build per-level channel indices
    level_indices: list[list[int]] = [[] for _ in range(NUM_LEVELS)]
    surface_fast_indices: list[int] = []
    surface_slow_indices: list[int] = []

    pl_set = set()
    for li, level in enumerate(LEVELS_13):
        for var in PL_VARS:
            name = f"{var}{level}"
            if name in in_names:
                idx = in_names.index(name)
                level_indices[li].append(idx)
                pl_set.add(idx)

    for i, name in enumerate(in_names):
        if i not in pl_set:
            if split_surface and name in SLOW_SFCS:
                surface_slow_indices.append(i)
            else:
                surface_fast_indices.append(i)

    # 1000hPa和地表快变量（降水、风速、辐射等）合在一起（原来的 level_indices[0]）
    level_indices[0] = level_indices[0] + surface_fast_indices

    # Step 2: Build per-group slice indices (concatenated channels from all levels in group)
    group_slice_indices: list[list[int]] = []

    # 如果开启了 split_surface，我们将独立出来的慢变量层作为 Group 0，插在最前面。
    if split_surface and len(surface_slow_indices) > 0:
        group_slice_indices.append(surface_slow_indices)

    for group in level_groups:
        group_chans = []
        for li in group:
            group_chans.extend(level_indices[li])
        group_slice_indices.append(group_chans)

    # Step 3: Build reassemble map, mapping sequential group output channels back to full channel indices.
    group_reassemble_map: list[int] = []
    group_mapping = sum(group_slice_indices, [])  # flatten list of lists
    group_reassemble_map = [group_mapping.index(i) for i in range(len(in_names))]  # map full channel index to group output index

    return group_slice_indices, group_reassemble_map


@MODELS.register()
class GroupEncoder(UTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.head
        self.norm_layer = AdaptiveLN(self.embed_dim)

    def forward(self, input, timestamp, const=None):
        input = input[:, -self.in_frames :]  # bxTxcxhxw
        batch_size = input.shape[0]
        resized_input = self.resize_input(input)
        resized_input = rearrange(resized_input, "(n t) c h w -> n t c h w", n=batch_size)
        h = self.patch_embed(resized_input, const)
        shortcut = h
        embed = self.get_embed(timestamp)
        for i, blk in enumerate(self.layers):
            h = getattr(self, f"drop{i}")(h, self.drop_mcmc)
            if "rotary" in self.attn_type:
                h = blk(
                    h,
                    time_embedding=embed,
                    freqs_cos=self.freqs_cos,
                    freqs_sin=self.freqs_sin,
                )
            else:
                h = blk(h, time_embedding=embed)
        h = self.norm_layer(h, embed) + shortcut
        return h


@MODELS.register()
class GroupDecoder(UTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_layer = AdaptiveLN(self.embed_dim)
        if hasattr(self, "patch_embed"):
            del self.patch_embed

    def forward(self, latent_h, timestamp=None, const=None):
        if timestamp is not None and "step" in timestamp:
            step_val = timestamp["step"]
            step_val = step_val[0] if getattr(step_val, "dim", lambda: 0)() > 0 else step_val
            from xmetai.layers import set_lora_step

            set_lora_step(self, int(step_val.item()))

        h = latent_h
        shortcut = h
        ih, iw = self.image_size
        embed = self.get_embed(timestamp)

        for i, blk in enumerate(self.layers):
            h = getattr(self, f"drop{i}")(h, self.drop_mcmc)
            if "rotary" in self.attn_type:
                h = blk(
                    h,
                    time_embedding=embed,
                    freqs_cos=self.freqs_cos,
                    freqs_sin=self.freqs_sin,
                )
            else:
                h = blk(h, time_embedding=embed)

        h = self.norm_layer(h, embed) + shortcut
        h = rearrange(h, "n (h w) c -> n c h w", h=self.feat_size[0])
        h = F.pad(h, (0, 0, 0, 1))

        pred = self.head(h, need_rearrange=False)
        pred = pred[:, :, :ih, :iw]
        if pred.shape[-2:] != [ih, iw]:
            pred = F.interpolate(
                pred.float(),
                size=(ih, iw),
                mode="bilinear",
                align_corners=False,
            ).to(latent_h)

        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        elif pred.dim() != 5:
            raise ValueError(f"Unexpected pred dim: {pred.shape}")

        return pred


class ConvHead(nn.Module):
    """ConvTranspose2d prediction head compatible with Decoder.

    Decoder.forward calls self.head(h, need_rearrange=False) where h is
    already in (N, C, H, W) spatial format. This wrapper accepts and ignores
    the need_rearrange kwarg so any head_type can be used with Decoder.
    """

    def __init__(self, in_dim: int, out_chans: int, patch_size: int = 2):
        super().__init__()
        self.proj = EarthConvTranspose2d(in_dim, out_chans, patch_size, stride=patch_size)

    def forward(self, h: torch.Tensor, need_rearrange: bool = True) -> torch.Tensor:
        return self.proj(h)


class LatentMixerPredictor(UTransformer):
    """Global attention coupling across spatial positions and pressure level groups.
    Uses UTransformer's layer construction and operates on flattened latent group tokens
    (Z: B, N, G, D -> B, N, G*D) to allow full spatial self-attention on the
    concatenated level features. The parent UTransformer handles layer/block building;
    we only care about self.layers in forward().

    Args:
        latent_dim: embedding dimension (matches VAE latent dimension)
        num_groups: number of pressure level groups (G dimension)
        num_heads: attention heads in transformer blocks
        num_layers: number of transformer block layers
        drop_mcmc: MC-Dropout rate
        attn_type: attention type ("v2", "rotary", etc.)
        norm_type: normalization type ("adaln", etc.)
        ffn_type: feed-forward type ("gegluffn", etc.)
        layer_type: block type ("swin", etc.)
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        window_size: int,
        latent_dim: list[int],
        num_heads: int = 4,
        num_layers: int = 2,
        drop_mcmc: float = 0.0,
        attn_type: str = "v2",
        mask_type: str = "h",
        norm_type: str = "adaln",
        ffn_type: str = "gegluffn",
        layer_type: str = "mamba_embed",
        use_projection: bool = True,
        **kwargs: Any,
    ) -> None:
        # Initialize parent UTransformer with minimal dummy parameters.
        # The critical parameters for layer construction are:
        #   - embed_dim: must match latent_dim
        #   - depths: controls number of blocks (num_layers)
        #   - num_heads, attn_type, norm_type, ffn_type, layer_type: block configs
        # Other spatial parameters (in_chans, image_size, etc.) are not used in forward().
        super().__init__(
            in_chans=1,  # dummy
            out_chans=1,  # dummy
            in_frames=1,  # dummy
            embed_dim=sum(latent_dim),  # critical: used for layer dimensions
            image_size=image_size,
            window_size=window_size,
            patch_size=1,  # dummy
            down_times=0,  # no downsampling
            depths=[num_layers],  # critical: controls number of blocks
            num_heads=num_heads,  # critical: for attention
            drop_mcmc=drop_mcmc,  # MC-Dropout rate
            attn_type=attn_type,  # critical: attention mechanism
            mask_type=mask_type,  # critical: masking mechanism
            head_type="none",  # no prediction head needed
            norm_type=norm_type,  # critical: normalization
            ffn_type=ffn_type,  # critical: feed-forward
            layer_type=layer_type,  # critical: block type
        )
        del self.patch_embed
        del self.head
        self.drop_mcmc = drop_mcmc
        self.latent_dim = latent_dim
        if use_projection:
            self.group_projs = torch.nn.ModuleList([torch.nn.Linear(sum(latent_dim), d) for d in latent_dim])
        else:
            self.group_projs = None

    def forward(self, h_states, timestamp=None):
        if len(h_states) != len(self.latent_dim):
            raise ValueError(f"LatentMixerPredictor expects num_groups={len(self.latent_dim)}, got {len(h_states)}")
        embed_dims = [h.shape[-1] for h in h_states]
        if embed_dims != self.latent_dim:
            raise ValueError(f"LatentMixerPredictor expects per-group latent_dim={self.latent_dim}, got {embed_dims}")
        h = torch.cat(h_states, dim=-1)  # (B, N, G*D)
        time_embedding = self.get_embed(timestamp)
        shortcut = h_states
        for i, blk in enumerate(self.layers):
            h = getattr(self, f"drop{i}")(h, self.drop_mcmc)
            self.logger.debug(f"LatentMixerPredictor block {i} input shape: {h.shape}, time_embedding shape: {time_embedding.shape}")
            if "rotary" in self.attn_type:
                h = blk(h, time_embedding=time_embedding, freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
            else:
                h = blk(h, time_embedding=time_embedding)
        if self.group_projs is not None:
            outputs = [proj(h) for proj in self.group_projs]
            outputs = [out + sc for out, sc in zip(outputs, shortcut)]  # residual connection in latent space
        else:
            h = h.split(self.latent_dim, dim=-1)  # split back into groups
            outputs = [out + sc for out, sc in zip(h, shortcut)]  # residual connection in latent space
        return outputs


@MODELS.register()
class GroupVAEPredictor(BasePredictor):
    def __init__(
        self,
        const_chans: int = 0,
        embed_dim: int | list[int] = 384,
        image_size: tuple[int, int] = (120, 240),
        window_size: int = 10,
        patch_size: int = 2,
        down_times: int = 0,
        encoder_depths: list[int] = [6, 6],
        decoder_depths: list[int] = [12],
        num_heads: int = 6,
        drop_mcmc: float = 0.05,
        attn_type: str = "v2",
        mask_type: str = "h",
        head_type: str = "2fc",
        norm_type: str = "adaln",
        ffn_type: str = "gegluffn",
        level_groups: list[list[int]] | None = None,
        split_surface: bool = False,
        vt_num_heads: int = 4,
        vt_num_layers: int = 12,
        vt_layer_type: str = "swin",
        vt_use_proj: bool = True,
        vae_type: str = "index",
        vae_residual_indices: list[str] | None = None,
        alpha: float = 1.0,
        loss_weight: tuple[float, float, float, float, float] = (0.2, 0.2, 1.0, 0.01, 0.3),
        rollout_max: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.logger.name = __name__

        self.alpha = alpha
        if len(loss_weight) != 5:
            raise ValueError("loss_weight must contain five floats: (vae_input, vae_target, pred_rec, latent_reg, latent_pred)")
        self.loss_weight = loss_weight
        self.const_chans = const_chans
        self.rollout_max = rollout_max

        if level_groups is None:
            level_groups = DEFAULT_LEVEL_GROUPS
        self.group_slice_indices, self.group_reassemble_map = build_level_channel_map(list(self.in_names), level_groups, split_surface)
        self.num_groups = len(self.group_slice_indices)

        if isinstance(embed_dim, int):
            self.embed_dims = [embed_dim] * self.num_groups
        else:
            if len(embed_dim) != self.num_groups:
                raise ValueError("Length of embed_dim list must match num_groups.")
            self.embed_dims = embed_dim

        self.vae_type = vae_type
        self.residual_vae_indices = self._build_residual_indices(vae_type, vae_residual_indices)
        self.logger.info(f"VAE residual mode: {self.vae_type}, indices: {self.residual_vae_indices}")
        self.group_reassemble_map = torch.tensor(self.group_reassemble_map, dtype=torch.long)
        self.logger.info(f"Group slice indices: {self.group_slice_indices}")

        common_encoder_kwargs = dict(
            image_size=image_size,
            in_frames=self.in_frames,
            const_chans=self.const_chans,
            window_size=window_size,
            patch_size=patch_size,
            down_times=down_times,
            depths=encoder_depths,
            num_heads=num_heads,
            drop_mcmc=drop_mcmc * 2,
            attn_type=attn_type,
            mask_type=mask_type,
            norm_type=norm_type,
            ffn_type=ffn_type,
        )
        common_decoder_kwargs = dict(
            image_size=image_size,
            window_size=window_size,
            patch_size=patch_size,
            down_times=down_times,
            depths=decoder_depths,
            num_heads=num_heads,
            attn_type=attn_type,
            mask_type=mask_type,
            head_type=head_type,
            norm_type=norm_type,
            ffn_type=ffn_type,
        )

        vae_encoder_list: list[GroupEncoder] = []
        posterior_encoder_list: list[GroupEncoder] = []
        decoder_list: list[GroupDecoder] = []

        for gi, group in enumerate(self.group_slice_indices):
            group_chans = len(self.group_slice_indices[gi])
            group_embed_dim = self.embed_dims[gi]

            vae_encoder = GroupEncoder(embed_dim=group_embed_dim, in_chans=group_chans, out_chans=group_chans, **common_encoder_kwargs)
            vae_encoder_list.append(vae_encoder)

            posterior_encoder = GroupEncoder(embed_dim=group_embed_dim, in_chans=group_chans, out_chans=group_chans, **common_encoder_kwargs)
            posterior_encoder_list.append(posterior_encoder)

            decoder = GroupDecoder(embed_dim=group_embed_dim, in_chans=group_chans, out_chans=group_chans, in_frames=self.in_frames, **common_decoder_kwargs)
            decoder.head = ConvHead(group_embed_dim, group_chans, patch_size=patch_size)
            decoder_list.append(decoder)

        self.vae_encoder_group = nn.ModuleList(vae_encoder_list)
        self.posterior_encoder_group = nn.ModuleList(posterior_encoder_list)
        self.decoder_group = nn.ModuleList(decoder_list)

        self.latent_mixer = LatentMixerPredictor(
            image_size=(image_size[0] // patch_size, image_size[1] // patch_size),
            window_size=window_size,
            latent_dim=self.embed_dims,
            num_heads=vt_num_heads,
            num_layers=vt_num_layers,
            attn_type=attn_type,
            mask_type=mask_type,
            norm_type=norm_type,
            layer_type=vt_layer_type,
            ffn_type=ffn_type,
            drop_mcmc=drop_mcmc,
            use_projection=vt_use_proj,
        )
        self.logger.info(
            f"Initialized GroupVAEPredictor with {self.num_groups} groups, embed_dims={self.embed_dims}, vae_type={self.vae_type}, residual_indices={self.residual_vae_indices}"
        )
        # set deterministic model member size to 1.
        self.members = 1
        self.logger.info(f"Group VAE with members={self.members}, alpha={self.alpha}, loss_weight={self.loss_weight}, rollout_max={self.rollout_max}")

    def _build_residual_indices(self, vae_type: str, vae_residual_indices: list[str] | None) -> list[int]:
        if vae_type == "absolute":
            return []
        if vae_type == "residual":
            return list(range(len(self.in_names)))
        if vae_type != "index":
            raise ValueError("vae_type must be one of {'residual', 'absolute', 'index'}.")

        if vae_residual_indices is None:
            if self.freq == 24:
                residual_channels = RESIDUAL_VARS_24H
            elif self.freq == 6:
                residual_channels = RESIDUAL_VARS_6H
            else:
                residual_channels = []
        else:
            residual_channels = vae_residual_indices
        self.logger.info(f"Building residual indices for vae_type='index' with channels: {residual_channels}")
        return [self.in_names.index(name) for name in residual_channels if name in self.in_names]

    def _apply_residual(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if len(self.residual_vae_indices) == 0:
            return pred
        out = pred.clone()
        out[:, :, self.residual_vae_indices] += ref[:, -1:, self.residual_vae_indices]
        return out

    def _slice_group(self, x: torch.Tensor, gi: int) -> torch.Tensor:
        indices = self.group_slice_indices[gi]
        return x[:, :, indices]

    def _reassemble(self, group_outputs: list[torch.Tensor], x_ref: torch.Tensor) -> torch.Tensor:
        full = torch.concatenate(group_outputs, dim=2)
        full = full[:, :, self.group_reassemble_map]
        return full

    @staticmethod
    def _resize_pred(pred: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if pred.shape[-2:] == torch.Size(target_hw):
            return pred
        B, T, C, _H, _W = pred.shape
        flat = pred.view(B * T, C, _H, _W)
        flat = F.interpolate(flat.float(), size=target_hw, mode="bilinear", align_corners=False).to(pred)
        return flat.view(B, T, C, *target_hw)

    def _encode_group(self, x_group: torch.Tensor, gi: int, timestamp: dict, is_prior: bool = True, const: torch.Tensor | None = None) -> torch.Tensor:
        encoder = self.vae_encoder_group[gi] if is_prior else self.posterior_encoder_group[gi]
        return encoder(x_group, timestamp, const=const)

    def _decode_group(self, h_state: torch.Tensor, gi: int, timestamp: dict, const: torch.Tensor | None = None) -> torch.Tensor:
        decoder = self.decoder_group[gi]
        return decoder(h_state, timestamp=timestamp, const=const)

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]
        targets = data["targets"]
        times = data["times"]
        const = data["const"] if self.const_chans > 0 else None

        future_step = targets.size(1)
        loss_steps, rolling_weights = self.build_loss_steps_and_weights(future_step, self.alpha, self.rollout_max)

        loss_vae_input = torch.zeros((), device=input.device)
        loss_vae_target = torch.zeros((), device=input.device)
        loss_pred_rec = torch.zeros((), device=input.device)
        loss_latent_pred_rec = torch.zeros((), device=input.device)
        loss_latent_reg = torch.zeros((), device=input.device)
        pinn_acc: dict[str, torch.Tensor] = {}

        true_input = input.clone()

        for t in range(self.start_frames, self.start_frames + future_step):
            k = t - self.start_frames
            loss_available = k in loss_steps
            target = targets[:, k : k + self.out_frames]

            input = self.reset_input(input)
            true_input = self.reset_input(true_input)

            timestamp = self.get_timestamp(input, times, t)
            timestamp_target = self.get_timestamp(input, times, t + 1)
            self.logger.debug(f"Time step {t}: input.shape={input.shape}, target.shape={target.shape}")

            new_target = torch.cat([true_input, target], dim=1)[:, -self.in_frames :]

            h_prior_states_by_group = []
            h_posterior_states_by_group = []
            vae_rec_input_outputs_by_group = []
            vae_rec_target_outputs_by_group = []

            for gi in range(self.num_groups):
                x_input_group = self._slice_group(input, gi)
                h_input = self._encode_group(x_input_group, gi, timestamp, is_prior=True, const=const)
                h_prior_states_by_group.append(h_input)
                vae_input_out = self._decode_group(h_input, gi, timestamp, const=const)
                vae_rec_input_outputs_by_group.append(vae_input_out)

                x_target_group = self._slice_group(new_target, gi)
                h_target = self._encode_group(x_target_group, gi, timestamp_target, is_prior=False, const=const)
                h_posterior_states_by_group.append(h_target)
                vae_target_out = self._decode_group(h_target, gi, timestamp_target, const=const)
                vae_rec_target_outputs_by_group.append(vae_target_out)

            Z_coupled = self.latent_mixer(h_prior_states_by_group, timestamp_target)

            if loss_available:
                # Constrain latent space to prevent norm explosion
                for h_prior, h_posterior in zip(h_prior_states_by_group, h_posterior_states_by_group):
                    loss_latent_reg += rolling_weights[k] * (torch.mean(h_prior**2) + torch.mean(h_posterior**2))

                vae_input_delta = self._reassemble(vae_rec_input_outputs_by_group, input)
                vae_input_delta = self._resize_pred(vae_input_delta, (int(input.shape[-2]), int(input.shape[-1])))

                # reference should be x_{t-1} based on formula x_t = F(h_t, x_{t-1})
                vae_input_base = true_input[:, -2:-1] if true_input.shape[1] > 1 else true_input[:, -1:]
                vae_input_recon = self._apply_residual(vae_input_delta, vae_input_base)
                vae_input_target = true_input[:, -1:]
                loss_vae_input += rolling_weights[k] * charbonnier_loss(vae_input_recon, vae_input_target, weight=self.weight)

                vae_target_delta = self._reassemble(vae_rec_target_outputs_by_group, new_target)
                vae_target_delta = self._resize_pred(vae_target_delta, (int(new_target.shape[-2]), int(new_target.shape[-1])))
                vae_target_recon = self._apply_residual(vae_target_delta, true_input)
                loss_vae_target += rolling_weights[k] * charbonnier_loss(vae_target_recon, target, weight=self.weight)

            group_outputs = []
            for gi in range(self.num_groups):
                if loss_available:
                    loss_latent_pred_rec += rolling_weights[k] * charbonnier_loss(Z_coupled[gi], h_posterior_states_by_group[gi].detach())
                out_group = self._decode_group(Z_coupled[gi], gi, timestamp_target, const=const)
                group_outputs.append(out_group)

            pred = self._reassemble(group_outputs, input)
            pred = self._resize_pred(pred, (int(input.shape[-2]), int(input.shape[-1])))
            output = self._apply_residual(pred, input)

            if loss_available:
                if "channel_weights" in self._buffers:
                    loss_pred_rec += rolling_weights[k] * charbonnier_loss(output, target, weight=self.channel_weights * self.weight)
                else:
                    loss_pred_rec += rolling_weights[k] * charbonnier_loss(output, target, weight=self.weight)

                if self.pinn_enable:
                    pinn_losses = self.compute_pinn_losses(output, target)
                    for loss_name, loss_val in pinn_losses.items():
                        pinn_acc[loss_name] = pinn_acc.get(loss_name, 0) + rolling_weights[k] * loss_val

            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
            true_input = torch.cat([true_input, target], dim=1)[:, -self.in_frames :]

        loss_dict = {
            "loss_pred_rec": self.loss_weight[2] * loss_pred_rec,
            "loss_vae_input": self.loss_weight[0] * loss_vae_input,
            "loss_vae_target": self.loss_weight[1] * loss_vae_target,
            "loss_latent_reg": self.loss_weight[3] * loss_latent_reg,
            "loss_latent_pred_rec": self.loss_weight[4] * loss_latent_pred_rec,
        }
        if len(pinn_acc) > 0:
            for loss_name, loss_val in pinn_acc.items():
                loss_dict[loss_name] = loss_val / future_step
        return loss_dict

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        const = data["const"] if self.const_chans > 0 else None
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames

        hw = list(input.shape[-2:])
        outputs = input.new_zeros(input.shape[0], len(steps), len(self.test_chans), *hw)

        for member in range(self.members):
            new_input = input.clone()
            new_input[~torch.isfinite(new_input)] = 0.0
            curr_step = 0

            for t in range(self.start_frames, total_step):
                new_input = self.reset_input(new_input)
                timestamp = self.get_timestamp(new_input, times, t)
                timestamp_target = self.get_timestamp(new_input, times, t + 1)

                h_states = []
                for gi in range(self.num_groups):
                    x_group = self._slice_group(new_input, gi)
                    h_state = self._encode_group(x_group, gi, timestamp, is_prior=True, const=const)
                    h_states.append(h_state)

                Z_coupled = self.latent_mixer(h_states, timestamp_target)

                group_outputs = []
                for gi in range(self.num_groups):
                    out_group = self._decode_group(Z_coupled[gi], gi, timestamp_target, const=const)
                    group_outputs.append(out_group)

                pred = self._reassemble(group_outputs, new_input)
                pred = self._resize_pred(pred, (int(new_input.shape[-2]), int(new_input.shape[-1])))
                output = self._apply_residual(pred, new_input)
                new_input = torch.cat([new_input, output], dim=1)[:, -self.in_frames :]

                if t + 1 in steps:
                    outputs[:, curr_step : curr_step + 1] += output[:, :, self.test_chans].detach()
                    curr_step += 1

        outputs /= self.members
        return outputs

    @torch.no_grad()
    def export_onnx(self, data, step, hour, doy):
        self.logger.info(f"export model with device:{self.device} and dtype:{self.dtype}")
        data[~torch.isfinite(data)] = 0.0
        normed_input = self.normalize(data)
        normed_input = normed_input.to(dtype=self.dtype)
        if isinstance(step, torch.Tensor) and step.is_floating_point():
            step = step.to(dtype=self.dtype)
        if isinstance(hour, torch.Tensor) and hour.is_floating_point():
            hour = hour.to(dtype=self.dtype)
        if isinstance(doy, torch.Tensor) and doy.is_floating_point():
            doy = doy.to(dtype=self.dtype)

        const = self._buffers.get("const", None)
        if const is not None:
            const = const.to(dtype=self.dtype)

        new_input = self.reset_input(normed_input)
        timestamp = dict(
            step=step.to(new_input),
            hour=hour.to(new_input),
            doy=doy.to(new_input),
        )

        freq = getattr(self, "freq", 6)
        next_hour_val = hour + freq / 24.0
        day_rollover = torch.floor(next_hour_val)

        timestamp_target = dict(
            step=(step + 1).to(new_input),
            hour=(next_hour_val % 1.0).to(new_input),
            doy=((doy + day_rollover * (1.0 / 365.0)) % 1.0).to(new_input),
        )

        h_states = []
        for gi in range(self.num_groups):
            x_group = self._slice_group(new_input, gi)
            h_state = self._encode_group(x_group, gi, timestamp, is_prior=True, const=const)
            h_states.append(h_state)

        Z_coupled = self.latent_mixer(h_states, timestamp_target)

        group_outputs = []
        for gi in range(self.num_groups):
            out_group = self._decode_group(Z_coupled[gi], gi, timestamp_target, const=const)
            group_outputs.append(out_group)

        pred = self._reassemble(group_outputs, new_input)
        pred = self._resize_pred(pred, (int(new_input.shape[-2]), int(new_input.shape[-1])))
        output = self._apply_residual(pred, new_input)

        output = self.inv_normalize(output)
        return output
