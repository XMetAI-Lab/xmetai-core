"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-03-06 12:00:00
Description: Model Architecture Designed by XMetAI.

See docs/tongyun/README.md for architecture and training details.

Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import math
from typing import Any, TypeAlias, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.core.base_predictor import BasePredictor, replace_channels
from xmetai.layers import AdaptiveLN, EarthConvTranspose2d, set_lora_step
from xmetai.layers.utransformer import UTransformer
from xmetai.loss import SphericalAMSELoss, charbonnier_loss, field_dynamics_consistency_loss
from xmetai.registries import MODELS
from xmetai.utils import get_event_storage
from xmetai.utils.events import has_event_storage

__all__ = [
    "TongYun",
]

LEVELS_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PL_VARS = ["z", "t", "u", "v", "q", "gh", "wz", "r"]


GROUP_VAR_MAP = {
    "residual_state": [f"{v}{level}" for v in ("z", "t") for level in LEVELS_13] + ["t2m", "t2m_min", "t2m_max", "d2m", "sst", "msl"],
    "abs_wind": [f"{v}{level}" for v in ("u", "v") for level in LEVELS_13]
    + ["u10", "v10", "10u", "10v", "u10m", "v10m", "u100", "v100", "100u", "100v", "u100m", "v100m", "ws10m", "ws100m"],
    "abs_scalar": [f"q{level}" for level in LEVELS_13] + ["q2m", "tcw", "tcwv", "tp", "ssr", "ssrd", "fdir", "ttr", "lcc", "mcc", "hcc", "tcc"],
}

GROUP_RECON_MODES = {
    "residual_state": "residual",
    "abs_wind": "absolute",
    "abs_scalar": "absolute",
}


def ensemble_schedule(max_members: int = 72, max_step: int = 65, freq: int = 6) -> list[int]:
    """Build lead-time-dependent ensemble size schedule.

    Uses the same exponential growth logic as `ens_latent_pertub`, but supports
    both 6-hour and 24-hour cadence by scaling lead time in hours via `t * freq`.
    """
    if freq not in (6, 24):
        raise ValueError(f"Only freq in (6, 24) is supported, got freq={freq}.")

    n0 = 4
    t0 = 36  # hours when nonlinear error starts to dominate
    lam = 0.015  # per-hour growth rate
    n_max = max_members
    ensemble_sizes: list[int] = []

    for t in range(1, max_step + 1):
        n_t = math.ceil(n0 * math.exp(lam * (t * freq - t0)))
        ensemble_sizes.append(min(max(n_t, 1), n_max))

    return ensemble_sizes


def build_feature_group_channel_map(
    in_names: list[str],
    group_var_map: dict[str, list[str]],
) -> tuple[list[list[int]], list[int], list[str]]:
    """Build channel groups from explicit variable-name groups.

    Variable names that appear in the map but are absent from ``in_names`` are
    ignored so the same map can be reused across nearby datasets. Variables
    present in ``in_names`` must be assigned exactly once.
    """
    name_to_idx = {name: idx for idx, name in enumerate(in_names)}
    assigned: dict[int, str] = {}
    group_slice_indices: list[list[int]] = []
    group_names = list(group_var_map.keys())

    for group_name, group_vars in group_var_map.items():
        group_indices: list[int] = []
        for var_name in dict.fromkeys(group_vars):
            if var_name not in name_to_idx:
                continue
            idx = name_to_idx[var_name]
            if idx in assigned:
                raise ValueError(f"Variable {var_name!r} is assigned to both {assigned[idx]!r} and {group_name!r}.")
            assigned[idx] = group_name
            group_indices.append(idx)
        if len(group_indices) == 0:
            raise ValueError(f"Feature group {group_name!r} has no channels in in_names.")
        group_slice_indices.append(group_indices)

    missing_names = [name for idx, name in enumerate(in_names) if idx not in assigned]
    if len(missing_names) > 0:
        raise ValueError(f"Feature groups must cover every input channel. Missing variables: {missing_names}")

    group_reassemble_map = _build_reassemble_map(group_slice_indices, len(in_names))
    return group_slice_indices, group_reassemble_map, group_names


def _build_reassemble_map(group_slice_indices: list[list[int]], num_channels: int) -> list[int]:
    flattened = [idx for group in group_slice_indices for idx in group]
    if len(flattened) != num_channels or len(set(flattened)) != num_channels:
        raise ValueError(
            f"group slices must partition all {num_channels} channels exactly once, got {len(flattened)} entries and {len(set(flattened))} unique indices"
        )

    reassemble_map = [0] * num_channels
    for group_pos, chan_idx in enumerate(flattened):
        reassemble_map[chan_idx] = group_pos
    return reassemble_map


_GroupShortcut: TypeAlias = torch.Tensor | dict[int, torch.Tensor]


@MODELS.register()
class GroupEncoder(UTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.head
        self.norm_layer = AdaptiveLN(self.embed_dim)
        self.local_shortcut = True

    def forward(self, input, timestamp, const=None):
        input = input[:, -self.in_frames :]  # bxTxcxhxw
        batch_size = input.shape[0]
        resized_input = self.resize_input(input)
        resized_input = rearrange(resized_input, "(n t) c h w -> n t c h w", n=batch_size)
        h = self.patch_embed(resized_input, const)
        shortcut = h
        if self.hierarchical and self.down_times:
            h, stages = self.forward_features(h, timestamp, return_stages=True, norm=self.norm_layer)
            return h, {index: stage for index, stage in enumerate(reversed(stages[:-1]), start=1)}
        h = self.forward_features(h, timestamp, norm=self.norm_layer)
        if self.local_shortcut:
            return h + shortcut
        return h, shortcut


@MODELS.register()
class GroupDecoder(UTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.up_times:
            # The parent only builds internal skips; decoder skips come from the external encoder.
            for index in range(1, self.num_layers):
                self.skip_projections[str(index)] = nn.Linear(2 * self.stage_dims[index], self.stage_dims[index])
        self.output_features = False
        self.norm_layer = AdaptiveLN(self.embed_dim)
        del self.patch_embed

    def _get_stage_sizes(self, down_factors, up_factors):
        if down_factors:
            raise ValueError("GroupDecoder supports upsampling stages only.")
        sizes = super()._get_stage_sizes(list(reversed(up_factors)), [])
        return list(reversed(sizes))

    def forward(self, latent_h, timestamp=None, const=None, patch_shortcut=None):
        ih, iw = self.image_size
        h = self.forward_features(latent_h, timestamp, stage_shortcuts=patch_shortcut if self.up_times else None, norm=self.norm_layer)
        if not self.up_times:
            h = h + (latent_h if patch_shortcut is None else patch_shortcut)
        h = rearrange(h, "n (h w) c -> n c h w", h=self.stage_sizes[-1][0])
        if self.output_features:
            return h
        h = F.pad(h, (0, 0, 0, 1))

        pred = self.head(h, need_rearrange=False)
        pred = pred[:, :, :ih, :iw]
        if pred.shape[-2:] != (ih, iw):
            pred = F.interpolate(
                pred.float(),
                size=(ih, iw),
                mode="bilinear",
                align_corners=False,
            ).to(latent_h)

        if pred.dim() == 4:
            pred = rearrange(pred, "n (t c) h w -> n t c h w", c=self.out_chans)
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
    """Shared latent dynamics across groups.

    Concatenate grouped latents (B, N, G, D -> B, N, G*D), apply the configured
    UTransformer blocks, and project back to the individual groups.

    Args:
        latent_dim: embedding dimension (matches VAE latent dimension)
        num_groups: number of latent groups (G dimension)
        num_heads: attention heads in transformer blocks
        num_layers: number of transformer block layers
        drop_mcmc: MC-Dropout rate
        attn_type: attention type ("v2", "rotary", etc.)
        norm_type: normalization type ("adaln", etc.)
        ffn_type: feed-forward type ("gegluffn", etc.)
        layer_type: block type ("swin", etc.)
        group_norm_mode: per-group normalization mode ("none", "norm", or "adaptive")
    """

    GROUP_NORM_MODES = {"none", "norm", "adaptive"}
    GROUP_NORM_WARMUP_ITERS = 1000
    GROUP_NORM_RAMP_ITERS = 4000

    def __init__(
        self,
        image_size: tuple[int, int],
        window_size: int | None,
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
        use_year_embed: bool = False,
        group_norm_mode: str = "none",
        **kwargs: Any,
    ) -> None:
        if group_norm_mode not in self.GROUP_NORM_MODES:
            raise ValueError(f"group_norm_mode must be one of {sorted(self.GROUP_NORM_MODES)}, got {group_norm_mode!r}.")

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
            use_year_embed=use_year_embed,
        )
        del self.patch_embed
        del self.head
        self.drop_mcmc = drop_mcmc
        self.latent_dim = latent_dim
        self.group_norm_mode = group_norm_mode
        if self.group_norm_mode == "adaptive":
            self.group_norm_logits = nn.Parameter(torch.zeros(len(latent_dim)))
            self.group_norm_logits._no_weight_decay = True
            # Keep old checkpoints functionally unchanged at zero, then save the
            # fine-tuning progress so export preserves the selected strength.
            self.register_buffer("group_norm_max", torch.tensor(0.0))
        if use_projection:
            self.group_projs = torch.nn.ModuleList([torch.nn.Linear(sum(latent_dim), d) for d in latent_dim])
        else:
            self.group_projs = None

    def _update_group_norm_max(self) -> None:
        if not self.training or not has_event_storage():
            return
        current_iter = get_event_storage().iter
        progress = max(0.0, min(1.0, (current_iter - self.GROUP_NORM_WARMUP_ITERS) / self.GROUP_NORM_RAMP_ITERS))
        self.group_norm_max.fill_(progress)

    def forward(self, h_states, timestamp=None):
        if len(h_states) != len(self.latent_dim):
            raise ValueError(f"LatentMixerPredictor expects num_groups={len(self.latent_dim)}, got {len(h_states)}")
        embed_dims = [h.shape[-1] for h in h_states]
        if embed_dims != self.latent_dim:
            raise ValueError(f"LatentMixerPredictor expects per-group latent_dim={self.latent_dim}, got {embed_dims}")
        if self.group_norm_mode == "none":
            mixer_states = h_states
        elif self.group_norm_mode == "norm":
            mixer_states = [F.layer_norm(x.float(), (d,), eps=1e-6).to(x) for x, d in zip(h_states, self.latent_dim)]
        else:
            self._update_group_norm_max()
            offsets = 2.0 * self.group_norm_logits.sigmoid() - 1.0
            strengths = self.group_norm_max + self.group_norm_max * (1.0 - self.group_norm_max) * offsets
            mixer_states = [
                torch.lerp(x, F.layer_norm(x.float(), (d,), eps=1e-6).to(x), strength.to(x)) for x, d, strength in zip(h_states, self.latent_dim, strengths)
            ]
        h = torch.cat(mixer_states, dim=-1)
        shortcut = h_states
        h = self.forward_features(h, timestamp)
        if self.group_projs is not None:
            outputs = [proj(h) for proj in self.group_projs]
            outputs = [out + sc for out, sc in zip(outputs, shortcut)]  # residual connection in latent space
        else:
            h = h.split(self.latent_dim, dim=-1)  # split back into groups
            outputs = [out + sc for out, sc in zip(h, shortcut)]  # residual connection in latent space
        return outputs


class _TongYunBase(BasePredictor):
    """Model Architecture Designed by XMetAI; see docs/tongyun/README.md."""

    def __init__(
        self,
        const_chans: int = 0,
        embed_dim: int | list[int] = 384,
        image_size: tuple[int, int] = (120, 240),
        window_size: int = 10,
        patch_size: int = 2,
        down_times: int | list[int] = 0,
        encoder_depths: list[int] = [6, 6],
        decoder_depths: list[int] = [12],
        num_heads: int = 6,
        drop_mcmc: float = 0.05,
        attn_type: str = "v2",
        mask_type: str = "h",
        head_type: str = "2fc",
        norm_type: str = "adaln",
        ffn_type: str = "gegluffn",
        layer_type: str = "swin",
        group_slice_indices: list[list[int]] | None = None,
        vt_attn_type: str = "std",
        vt_window_size: int | None = None,
        vt_num_heads: int = 4,
        vt_num_layers: int = 12,
        vt_layer_type: str = "swin",
        vt_use_proj: bool = True,
        vt_group_norm_mode: str = "none",
        patch_decoder_shortcut: bool = False,
        residual_vae_indices: list[int] | None = None,
        alpha: float = 1.0,
        loss_weight: tuple[float, float, float, float, float] = (0.2, 0.2, 1.0, 0.01, 0.3),
        latent_dyn_weight: float = 0.0,
        amse_weight: float = 0.0,
        amse_image_size: tuple[int, int] | None = None,
        rollout_max: int = 50,
        pretrain: bool = False,
        pretrain_mask_ratio: float = 0.15,
        pretrain_temporal_mix_prob: float = 0.5,
        forecast_stride: int = -1,
        kl_schedule_mode: str = "cosine",
        kl_min_weight: float = 0.0,
        kl_max_weight: float = 100.0,
        kl_cycle_iters: int = 40000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.logger.name = __name__

        self.alpha = alpha
        if len(loss_weight) != 5:
            raise ValueError("loss_weight must contain five floats: (vae_input, vae_target, pred_rec, latent_reg, latent_pred)")
        self.loss_weight = loss_weight
        self.amse_weight = float(amse_weight)
        if not math.isfinite(self.amse_weight) or self.amse_weight < 0:
            raise ValueError("amse_weight must be finite and nonnegative")
        # AMSE operates on the native output grid, not the encoder grid.
        amse_image_size = image_size if amse_image_size is None else amse_image_size
        self.amse_loss = SphericalAMSELoss(amse_image_size) if self.amse_weight > 0 else None
        self.latent_dyn_weight = float(latent_dyn_weight)
        self.compute_prior_vae = self.loss_weight[0] != 0
        self.compute_posterior_vae = self.loss_weight[1] != 0
        self.compute_pred_rec = self.loss_weight[2] != 0
        self.compute_latent_reg = self.loss_weight[3] != 0
        self.compute_latent_pred = self.compute_posterior_vae and self.loss_weight[4] != 0
        self.compute_latent_dyn = self.latent_dyn_weight != 0
        self.const_chans = const_chans
        self.rollout_max = rollout_max
        if self.in_frames < self.out_frames + 1:
            raise ValueError(
                f"Input frames (in_frames={self.in_frames}) must be >= out_frames+1 (out_frames={self.out_frames}) to build a reconstruction base window."
            )
        self.forecast_stride = int(forecast_stride)
        if self.forecast_stride <= 0 or self.forecast_stride > self.out_frames:
            self.forecast_stride = self.out_frames
            self.logger.warning(f"Invalid forecast_stride {forecast_stride} reset to out_frames {self.out_frames}.")
        self.kl_schedule_mode = kl_schedule_mode
        self.kl_min_weight = float(kl_min_weight)
        self.kl_max_weight = float(kl_max_weight)
        self.kl_cycle_iters = int(kl_cycle_iters)
        if self.kl_cycle_iters <= 0:
            raise ValueError(f"kl_cycle_iters must be positive, got {self.kl_cycle_iters}.")
        if pretrain:
            self.pretrain = True
            self.pretrain_mask_ratio = float(pretrain_mask_ratio)
            self.pretrain_temporal_mix_prob = float(pretrain_temporal_mix_prob)
            self.pretrain_mcr_weight = loss_weight[1]
            self.pretrain_aug_rec_weight = loss_weight[4]
            self.pretrain_latent_reg_weight = loss_weight[3]
        else:
            self.pretrain = False

        if group_slice_indices is None:
            raise ValueError("group_slice_indices must be provided by the concrete Group VAE subclass.")

        self.group_slice_indices = [list(group) for group in group_slice_indices]
        group_reassemble_map = _build_reassemble_map(self.group_slice_indices, len(self.in_names))
        self.num_groups = len(self.group_slice_indices)

        if isinstance(embed_dim, int):
            self.embed_dims = [embed_dim] * self.num_groups
        else:
            if len(embed_dim) != self.num_groups:
                raise ValueError("Length of embed_dim list must match num_groups.")
            self.embed_dims = embed_dim

        self.residual_vae_indices = [] if residual_vae_indices is None else list(residual_vae_indices)
        invalid_residual_indices = [idx for idx in self.residual_vae_indices if idx < 0 or idx >= len(self.in_names)]
        if len(invalid_residual_indices) > 0:
            raise ValueError(f"residual_vae_indices contains invalid channel indices: {invalid_residual_indices}")
        self.register_buffer("group_reassemble_map", torch.tensor(group_reassemble_map, dtype=torch.long), persistent=False)
        self.register_buffer("residual_vae_index", torch.tensor(self.residual_vae_indices, dtype=torch.long), persistent=False)
        for gi, group_indices in enumerate(self.group_slice_indices):
            self.register_buffer(f"group_indices_{gi}", torch.tensor(group_indices, dtype=torch.long), persistent=False)

        down_factors = [] if isinstance(down_times, int) else list(down_times)
        sampled = bool(down_factors)
        common_kwargs = dict(
            image_size=image_size,
            in_frames=self.in_frames,
            window_size=window_size,
            patch_size=patch_size,
            num_heads=num_heads,
            attn_type=attn_type,
            mask_type=mask_type,
            norm_type=norm_type,
            ffn_type=ffn_type,
            layer_type=layer_type,
        )
        encoder_kwargs = dict(common_kwargs, const_chans=self.const_chans, down_times=down_times, depths=encoder_depths, drop_mcmc=drop_mcmc * 2)
        decoder_kwargs = dict(
            common_kwargs,
            out_frames=self.out_frames,
            down_times=0 if sampled else down_times,
            up_times=list(reversed(down_factors)) if sampled else None,
            depths=decoder_depths,
            head_type=head_type,
        )

        vae_encoder_list: list[GroupEncoder] = []
        posterior_encoder_list: list[GroupEncoder] = []
        decoder_list: list[GroupDecoder] = []

        for gi, group in enumerate(self.group_slice_indices):
            group_chans = len(group)
            group_embed_dim = self.embed_dims[gi]

            vae_encoder = GroupEncoder(embed_dim=group_embed_dim, in_chans=group_chans, out_chans=group_chans, **encoder_kwargs)
            vae_encoder_list.append(vae_encoder)

            posterior_encoder = GroupEncoder(embed_dim=group_embed_dim, in_chans=group_chans, out_chans=group_chans, **encoder_kwargs)
            posterior_encoder_list.append(posterior_encoder)

            decoder = GroupDecoder(
                embed_dim=group_embed_dim,
                in_chans=group_chans,
                out_chans=group_chans,
                **decoder_kwargs,
            )
            decoder.head = ConvHead(group_embed_dim, group_chans * self.out_frames, patch_size=patch_size)
            decoder_list.append(decoder)

        self.vae_encoder_group = nn.ModuleList(vae_encoder_list)
        self.posterior_encoder_group = nn.ModuleList(posterior_encoder_list)
        self.decoder_group = nn.ModuleList(decoder_list)
        self.patch_decoder_shortcut = patch_decoder_shortcut or sampled
        for module in (*self.vae_encoder_group, *self.posterior_encoder_group):
            module.local_shortcut = not self.patch_decoder_shortcut

        self.latent_mixer = LatentMixerPredictor(
            image_size=tuple(self.vae_encoder_group[0].stage_sizes[-1]),
            window_size=vt_window_size,
            latent_dim=self.embed_dims,
            num_heads=vt_num_heads,
            num_layers=vt_num_layers,
            attn_type=vt_attn_type,
            mask_type=mask_type,
            norm_type=norm_type,
            layer_type=vt_layer_type,
            ffn_type=ffn_type,
            drop_mcmc=drop_mcmc,
            use_projection=vt_use_proj,
            group_norm_mode=vt_group_norm_mode,
        )
        self.logger.info(
            f"Initialized {self.__class__.__name__} with {self.num_groups} groups, embed_dims={self.embed_dims}, residual_indices={self.residual_vae_indices}"
        )

        # Build evaluation ensemble schedule once during initialization so test
        # and subclasses can reuse the same step-wise member plan directly.
        self.eval_total_step = max(self.test_frames) + self.start_frames
        if self.freq in (6, 24):
            self.ensemble_sizes = ensemble_schedule(
                max_members=self.members,
                max_step=self.eval_total_step,
                freq=self.freq,
            )
        else:
            self.ensemble_sizes = [self.members] * self.eval_total_step

        self.logger.info(
            f"Evaluation ensemble schedule initialized: freq={self.freq}, members={self.members}, "
            f"total_step={self.eval_total_step}, forecast_stride={self.forecast_stride}, ensemble_sizes={self.ensemble_sizes}"
        )
        self.logger.info(
            f"Group VAE with members={self.members}, alpha={self.alpha}, loss_weight={self.loss_weight}, "
            f"latent_dyn_weight={self.latent_dyn_weight}, rollout_max={self.rollout_max}"
        )
        if self.pretrain:
            self._configure_pretrain_trainable()
        elif not self._uses_posterior_encoder():
            self.freeze(("posterior_encoder_group",))

    def _uses_posterior_encoder(self) -> bool:
        """Whether forecast training uses posterior features; subclasses may add consumers."""
        return self.compute_posterior_vae

    def _configure_pretrain_trainable(self) -> None:
        """Train the posterior encoder and decoder.

        `forward_pretrain_stage` does not use the prior encoder or latent mixer. Freezing
        the unused branches avoids DDP unused-parameter reduction errors while
        keeping the complete reconstruction path trainable.
        """
        for param in self.parameters():
            param.requires_grad = False
        for module in (self.posterior_encoder_group, self.decoder_group):
            for param in module.parameters():
                param.requires_grad = True
        self.logger.info("Pretrain mode: posterior_encoder_group and decoder_group are trainable.")

    def _apply_residual(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.residual_vae_index.numel() == 0:
            return pred
        residual = pred.index_select(2, self.residual_vae_index) + ref.index_select(2, self.residual_vae_index)
        return replace_channels(pred, self.residual_vae_indices, residual)

    def _build_posterior_input(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reset forecast history while preserving target observations for the posterior encoder."""
        history = self.reset_input(history.clone())
        return torch.cat([history, target], dim=1)[:, -self.in_frames :]

    def _slice_group(self, x: torch.Tensor, gi: int) -> torch.Tensor:
        indices = getattr(self, f"group_indices_{gi}")
        return x.index_select(2, indices)

    def _prepare_group_input(self, x: torch.Tensor, gi: int) -> torch.Tensor:
        return self._slice_group(x, gi)

    def _reassemble(self, group_outputs: list[torch.Tensor], x_ref: torch.Tensor) -> torch.Tensor:
        return torch.cat(group_outputs, dim=2).index_select(2, self.group_reassemble_map)

    @staticmethod
    def _resize_pred(pred: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if pred.shape[-2:] == torch.Size(target_hw):
            return pred
        B, T, C, _H, _W = pred.shape
        flat = pred.view(B * T, C, _H, _W)
        flat = F.interpolate(flat.float(), size=target_hw, mode="bilinear", align_corners=False).to(pred)
        return flat.view(B, T, C, *target_hw)

    @final
    def _encode_group(
        self, x_group: torch.Tensor, gi: int, timestamp: dict, is_prior: bool = True, const: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, _GroupShortcut]:
        return (self.vae_encoder_group[gi] if is_prior else self.posterior_encoder_group[gi])(x_group, timestamp, const=const)

    def _set_decoder_lora_step(self, lora_step: int | None) -> None:
        if lora_step is None:
            return
        for decoder in self.decoder_group:
            set_lora_step(decoder, int(lora_step))

    def _is_group_frozen(self, gi: int, is_prior: bool = True) -> bool:
        """Check if a group encoder is frozen (has no trainable parameters)."""
        encoder = self.vae_encoder_group[gi] if is_prior else self.posterior_encoder_group[gi]
        return not any(p.requires_grad for p in encoder.parameters())

    @final
    def _encode_all_groups(
        self,
        x: torch.Tensor,
        timestamp: dict,
        const: torch.Tensor | None = None,
        is_prior: bool = True,
    ) -> list | tuple[list, list[_GroupShortcut]]:
        """Encode each group, returning shortcuts when decoder routing is enabled."""
        results_by_group = []
        shortcuts: list[_GroupShortcut] = []
        for gi in range(self.num_groups):
            x_group = self._prepare_group_input(x, gi)
            frozen = self._is_group_frozen(gi, is_prior=is_prior)
            with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                result = self._encode_group(x_group, gi, timestamp, is_prior=is_prior, const=const)
            if isinstance(result, tuple):
                result, shortcut = result
                shortcuts.append(shortcut)
            results_by_group.append(result)
        if self.patch_decoder_shortcut:
            return results_by_group, shortcuts
        return results_by_group

    @final
    def _decode_all_groups(
        self,
        h_states: list[torch.Tensor],
        timestamp: dict,
        const: torch.Tensor | None = None,
        lora_step: int | None = None,
        shortcuts: list[_GroupShortcut] | None = None,
    ) -> list[torch.Tensor]:
        """Decode groups with their matching encoder shortcuts."""
        self._set_decoder_lora_step(lora_step)
        outputs_by_group = []
        for gi, (decoder, h_state) in enumerate(zip(self.decoder_group, h_states, strict=True)):
            shortcut = None if shortcuts is None else shortcuts[gi]
            if not any(p.requires_grad for p in decoder.parameters()):
                h_state = h_state.detach()
                if shortcut is not None:
                    shortcut = {index: value.detach() for index, value in shortcut.items()} if isinstance(shortcut, dict) else shortcut.detach()
            outputs_by_group.append(decoder(h_state, timestamp=timestamp, const=const, patch_shortcut=shortcut))
        return outputs_by_group

    def _assemble_prediction(
        self,
        decoded_groups: list[torch.Tensor],
        reference_input: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Reassemble group outputs and apply residual reconstruction.

        Args:
            decoded_groups: List of decoded outputs from _decode_all_groups
            reference_input: Reference input for residual application (B, T, C, H, W)
            target_hw: Optional target height/width for resizing

        Returns:
            Full-field output with residual reconstruction applied (B, out_frames, C, H, W)
        """
        full_output = self._reassemble(decoded_groups, reference_input)
        if target_hw is None:
            target_hw = (int(reference_input.shape[-2]), int(reference_input.shape[-1]))
        full_output = self._resize_pred(full_output, target_hw)
        return self._apply_residual(full_output, reference_input[:, -1:])

    @final
    def _predict_next(self, input: torch.Tensor, times: Any, t: int, const: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.reset_input(input)
        timestamp = self.get_timestamp(input, times, t)
        timestamp_target = self.get_timestamp(input, times, t + self.out_frames)
        h_states = self._encode_all_groups(input, timestamp, const=const, is_prior=True)
        shortcuts = None
        if self.patch_decoder_shortcut:
            h_states, shortcuts = h_states
        z_coupled = self.latent_mixer(h_states, timestamp_target)
        group_outputs = self._decode_all_groups(z_coupled, timestamp_target, const=const, lora_step=t + self.out_frames, shortcuts=shortcuts)
        output = self._assemble_prediction(group_outputs, input)
        next_input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
        return output, next_input

    def _get_kl_weight(
        self,
        mode: str | None = None,
        max_weight: float | None = None,
        cycle_iters: int | None = None,
        min_weight: float | None = None,
    ) -> float:
        mode = self.kl_schedule_mode if mode is None else mode
        max_weight = self.kl_max_weight if max_weight is None else float(max_weight)
        min_weight = self.kl_min_weight if min_weight is None else float(min_weight)
        cycle_iters = self.kl_cycle_iters if cycle_iters is None else int(cycle_iters)
        try:
            current_iter = get_event_storage().iter
        except AssertionError:
            return float(max_weight)
        cycle_iters = max(1, int(cycle_iters))
        if mode == "cosine" and current_iter >= cycle_iters:
            cycle_pos = 1.0
        else:
            cycle_pos = (current_iter % cycle_iters) / cycle_iters
        weight_span = max(max_weight - min_weight, 0.0)
        if mode == "cosine":
            # Smooth monotonic warmup: min_weight -> max_weight.
            kl_weight = min_weight + 0.5 * weight_span * (1.0 - math.cos(math.pi * cycle_pos))
        elif mode == "cyclical":
            # Linear min_weight -> max_weight -> min_weight cycle.
            if cycle_pos < 0.5:
                kl_weight = min_weight + weight_span * (cycle_pos / 0.5)
            else:
                kl_weight = min_weight + weight_span * (1.0 - (cycle_pos - 0.5) / 0.5)
        else:
            raise ValueError(f"Unknown KL schedule mode: {mode}. Use 'cosine' or 'cyclical'.")

        if current_iter % cycle_iters == 0:
            self.logger.info(f"KL/Reg loss boundary: iter={current_iter}, mode={mode}, weight={float(kl_weight):.4f}")

        return float(kl_weight)

    def _pretrain_augment(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        temporal_mix_prob: float,
    ) -> torch.Tensor:
        """Build an augmented view for VAE pretraining."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got shape={x.shape}")

        aug = x.clone()
        bsz, t_len, n_chans, _, _ = aug.shape

        if mask_ratio > 0.0:
            mask_ratio = max(0.0, min(1.0, float(mask_ratio)))
            n_mask = max(1, int(round(n_chans * mask_ratio)))
            keep_prob = 1.0 - mask_ratio
            channel_gate = torch.rand(bsz, n_chans, device=aug.device) < keep_prob

            for bi in range(bsz):
                if not bool(channel_gate[bi].any()):
                    keep_idx = torch.randint(0, n_chans, (1,), device=aug.device)
                    channel_gate[bi, keep_idx] = True

            rand_perm = torch.stack([torch.randperm(n_chans, device=aug.device)[:n_mask] for _ in range(bsz)], dim=0)
            hard_mask = torch.ones(bsz, n_chans, device=aug.device, dtype=torch.bool)
            hard_mask.scatter_(1, rand_perm, False)
            channel_gate = channel_gate & hard_mask
            aug = aug * channel_gate[:, None, :, None, None].to(dtype=aug.dtype)

        if t_len > 1 and temporal_mix_prob > 0.0:
            temporal_mix_prob = max(0.0, min(1.0, float(temporal_mix_prob)))
            do_mix = (torch.rand(bsz, device=aug.device) < temporal_mix_prob).to(dtype=aug.dtype)
            do_mix = do_mix[:, None, None, None]
            mix_w = torch.empty(bsz, 1, 1, 1, device=aug.device, dtype=aug.dtype).uniform_(0.4, 0.9)
            mixed_last = mix_w * aug[:, -1] + (1.0 - mix_w) * aug[:, -2]
            aug[:, -1] = do_mix * mixed_last + (1.0 - do_mix) * aug[:, -1]

        return aug

    def _pretrain_diagnostic(self, output: torch.Tensor, kernel_size: int = 4) -> torch.Tensor:
        if output.ndim != 5:
            raise ValueError(f"Expected 5D output (B,T,C,H,W), got shape={tuple(output.shape)}")
        B, T, C, H, W = output.shape
        flat = output.view(B * T, C, H, W)
        pooled = F.avg_pool2d(flat, kernel_size=kernel_size, stride=kernel_size)
        Hp, Wp = pooled.shape[-2:]
        return pooled.view(B, T, C, Hp, Wp)

    def _pretrain_mcr_loss(
        self,
        h_states: list[torch.Tensor],
        reference_input: torch.Tensor,
        timestamp: dict,
        const: torch.Tensor | None = None,
        lora_step: int | None = None,
        pool_size: int = 4,
        small: float = 0.04,
        large: float = 0.08,
        alpha_amp: float = 0.05,
        min_scale: float = 0.08,
        shortcuts: list[_GroupShortcut] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if small <= 0.0 or large <= small:
            raise ValueError(f"MCR perturbation scales must satisfy 0 < small < large, got small={small}, large={large}.")

        h1_states = []
        h2_states = []
        for h in h_states:
            scale = h.detach().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(min_scale)
            direction = torch.randn_like(h)
            direction = direction / (direction.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-6)
            h1 = h + small * scale * direction
            h2 = h + large * scale * direction
            h1_states.append(h1)
            h2_states.append(h2)

        outputs0 = self._decode_all_groups(h_states, timestamp, const=const, lora_step=lora_step, shortcuts=shortcuts)
        outputs1 = self._decode_all_groups(h1_states, timestamp, const=const, lora_step=lora_step, shortcuts=shortcuts)
        outputs2 = self._decode_all_groups(h2_states, timestamp, const=const, lora_step=lora_step, shortcuts=shortcuts)

        x0 = self._assemble_prediction(outputs0, reference_input)
        x1 = self._assemble_prediction(outputs1, reference_input)
        x2 = self._assemble_prediction(outputs2, reference_input)

        f0 = self._pretrain_diagnostic(x0, kernel_size=pool_size)
        f1 = self._pretrain_diagnostic(x1, kernel_size=pool_size)
        f2 = self._pretrain_diagnostic(x2, kernel_size=pool_size)

        delta01 = f1 - f0.detach()
        delta12 = f2 - f1
        amp = delta01.pow(2).mean()
        curv = ((delta12 / (large - small)) - (delta01.detach() / small)).pow(2).mean() * (small * small)
        loss_mcr = alpha_amp * amp + curv
        return loss_mcr, x0

    def forward_pretrain_stage(self, data):
        """VAE-only pretraining stage.

        Uses the posterior path to reconstruct the tail of the input window.
        No latent mixer prediction path is involved.
        """
        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        const = data["const"] if self.const_chans > 0 else None

        recon_target = input[:, -self.out_frames :]
        input = self._build_posterior_input(input[:, : -self.out_frames], recon_target)
        recon_base = input[:, -self.out_frames - 1 : -1]
        timestamp = self.get_timestamp(input, times, self.start_frames)

        aug_input = self._pretrain_augment(input, mask_ratio=self.pretrain_mask_ratio, temporal_mix_prob=self.pretrain_temporal_mix_prob)

        # Encode clean and augmented inputs using the posterior path.
        h_clean_by_group = self._encode_all_groups(input, timestamp, const=const, is_prior=False)
        h_aug_by_group = self._encode_all_groups(aug_input, timestamp, const=const, is_prior=False)
        clean_shortcuts = aug_shortcuts = None
        if self.patch_decoder_shortcut:
            h_clean_by_group, clean_shortcuts = h_clean_by_group
            h_aug_by_group, aug_shortcuts = h_aug_by_group

        # Decode both views and reconstruct the clean tail frames.
        loss_mcr, recon_clean = self._pretrain_mcr_loss(
            h_clean_by_group,
            recon_base,
            timestamp,
            const=const,
            lora_step=self.start_frames,
            pool_size=8 if input.shape[-2] >= 256 else 4,
            shortcuts=clean_shortcuts,
        )
        recon_clean = recon_clean[:, -self.out_frames :]
        rec_aug_by_group = self._decode_all_groups(h_aug_by_group, timestamp, const=const, lora_step=self.start_frames, shortcuts=aug_shortcuts)
        recon_aug = self._assemble_prediction(rec_aug_by_group, recon_base)[:, -self.out_frames :]

        # Prepare weight for loss computation
        rec_weight = None
        base_weight = self._buffers.get("weight", None)
        channel_weight = self._buffers.get("channel_weights", None)
        if base_weight is not None:
            rec_weight = base_weight
            if channel_weight is not None:
                rec_weight = channel_weight * base_weight

        loss_rec_clean = charbonnier_loss(recon_clean, recon_target, weight=rec_weight)
        loss_rec_aug = charbonnier_loss(recon_aug, recon_target, weight=rec_weight)

        loss_latent_reg = input.new_zeros(())
        for h_clean, h_aug in zip(h_clean_by_group, h_aug_by_group):
            loss_latent_reg = loss_latent_reg + torch.mean(h_clean**2) + torch.mean(h_aug**2)
        loss_latent_reg = loss_latent_reg / self.num_groups

        return {
            "loss_pre_rec": loss_rec_clean + self.pretrain_aug_rec_weight * loss_rec_aug,
            "loss_pre_mcr": self.pretrain_mcr_weight * loss_mcr,
            "loss_pre_latent_reg": self.pretrain_latent_reg_weight * loss_latent_reg,
        }

    def forward(self, data, *, rollout_state=None, rollout_segment_size=None):
        if not self.training:
            return self.test(data)

        if self.pretrain:
            return self.forward_pretrain_stage(data)

        if rollout_state is None:
            targets = data["targets"][:, : data.get("future_frames", self.out_frames)]
            model_data = data.copy()
            del model_data["targets"]
            data = self.process_data(model_data)
            input = data["input"]
            future_step = targets.size(1)
            if future_step < self.out_frames:
                raise ValueError(f"Expected at least {self.out_frames} target frames, got {future_step}.")

            num_rollout_blocks = 1 + (future_step - self.out_frames) // self.forecast_stride
            segment_size = num_rollout_blocks if rollout_segment_size is None else int(rollout_segment_size)
            if segment_size <= 0:
                raise ValueError(f"rollout_segment_size must be positive, got {segment_size}.")

            num_segments = (num_rollout_blocks + segment_size - 1) // segment_size
            loss_steps, rolling_weights = self.build_loss_steps_and_weights(future_step, self.alpha, self.rollout_max)
            loss_step_set = set(loss_steps)
            loss_block_indices = {
                block_index
                for block_index in range(num_rollout_blocks)
                if not loss_step_set.isdisjoint(range(block_index * self.forecast_stride, block_index * self.forecast_stride + self.out_frames))
            }
            state = {
                "targets": targets,
                "input": input,
                "times": data["times"],
                "const": data["const"] if self.const_chans > 0 else None,
                "future_step": future_step,
                "segment_size": segment_size,
                "num_rollout_blocks": num_rollout_blocks,
                "num_segments": num_segments,
                "loss_block_indices": loss_block_indices,
                "rolling_weights": rolling_weights,
                "kl_weight": self._get_kl_weight(),
                "true_input": input.clone(),
                "segment_index": 0,
                "k": 0,
            }
        else:
            state = rollout_state

        input = state["input"]
        targets = state["targets"]
        times = state["times"]
        const = state["const"]
        future_step = state["future_step"]
        segment_size = state["segment_size"]
        num_rollout_blocks = state["num_rollout_blocks"]
        num_segments = state["num_segments"]
        loss_block_indices = state["loss_block_indices"]
        rolling_weights = state["rolling_weights"]
        kl_weight = state["kl_weight"]
        true_input = state["true_input"]
        segment_index = state["segment_index"]
        k = state["k"]

        # Traverse skipped segments without retaining a graph. Each call then
        # returns exactly one selected contiguous segment for backward.
        while segment_index < num_segments:
            first_block = segment_index * segment_size
            last_block = min(first_block + segment_size, num_rollout_blocks)
            if any(block_index in loss_block_indices for block_index in range(first_block, last_block)):
                break
            with torch.no_grad():
                for _ in range(first_block, last_block):
                    t = self.start_frames + k
                    target = targets[:, k : k + self.out_frames].to(input, non_blocking=True)
                    input = self.reset_input(input.detach())
                    true_input = self.reset_input(true_input)
                    _, input = self._predict_next(input, times, t, const=const)
                    input = input.detach()
                    true_input = torch.cat([true_input, target], dim=1)[:, -self.in_frames :]
                    k += self.forecast_stride
            segment_index += 1

        if segment_index >= num_segments:
            raise RuntimeError("No selected rollout loss block remains for backward.")

        first_block = segment_index * segment_size
        last_block = min(first_block + segment_size, num_rollout_blocks)

        loss_vae_input = input.new_zeros(())
        loss_vae_target = input.new_zeros(())
        loss_pred_rec = input.new_zeros(())
        loss_latent_pred_rec = input.new_zeros(())
        loss_latent_reg = input.new_zeros(())
        loss_latent_dyn_consistency = input.new_zeros(())
        loss_amse = input.new_zeros(())
        pinn_acc: dict[str, torch.Tensor] = {}
        precip_acc: dict[str, torch.Tensor] = {}

        for block_index in range(first_block, last_block):
            t = self.start_frames + k
            target = targets[:, k : k + self.out_frames].to(input, non_blocking=True)
            k += self.forecast_stride
            if block_index not in loss_block_indices:
                with torch.no_grad():
                    input = self.reset_input(input.detach())
                    true_input = self.reset_input(true_input)
                    _, input = self._predict_next(input, times, t, const=const)
                    input = input.detach()
                    true_input = torch.cat([true_input, target], dim=1)[:, -self.in_frames :]
                continue

            block_start = block_index * self.forecast_stride
            block_weight = rolling_weights[block_start : block_start + self.out_frames].mean()
            input = self.reset_input(input)
            true_input = self.reset_input(true_input)
            timestamp = self.get_timestamp(input, times, t)
            timestamp_target = self.get_timestamp(input, times, t + self.out_frames)

            h_prior_states = self._encode_all_groups(input, timestamp, const=const, is_prior=True)
            prior_shortcuts = None
            if self.patch_decoder_shortcut:
                h_prior_states, prior_shortcuts = h_prior_states
            # The prior always drives forecasting, independently of reconstruction.
            if self.compute_latent_reg:
                for h_prior in h_prior_states:
                    loss_latent_reg += block_weight * h_prior.square().mean() / self.num_groups

            if self.compute_prior_vae:
                vae_input_groups = self._decode_all_groups(h_prior_states, timestamp, const=const, lora_step=t, shortcuts=prior_shortcuts)
                vae_input_base = true_input[:, -self.out_frames - 1 : -1]
                vae_input_recon = self._assemble_prediction(vae_input_groups, vae_input_base)
                vae_input_target = true_input[:, -self.out_frames :]
                loss_vae_input += block_weight * charbonnier_loss(vae_input_recon, vae_input_target, weight=self.weight)

            z_coupled = self.latent_mixer(h_prior_states, timestamp_target)
            if self.compute_posterior_vae:
                new_target = self._build_posterior_input(true_input, target)
                h_posterior_states = self._encode_all_groups(new_target, timestamp_target, const=const, is_prior=False)
                posterior_shortcuts = None
                if self.patch_decoder_shortcut:
                    h_posterior_states, posterior_shortcuts = h_posterior_states
                vae_target_groups = self._decode_all_groups(
                    h_posterior_states, timestamp_target, const=const, lora_step=t + self.out_frames, shortcuts=posterior_shortcuts
                )
                vae_target_base = new_target[:, -self.out_frames - 1 : -1]
                vae_target_recon = self._assemble_prediction(vae_target_groups, vae_target_base)
                loss_vae_target += block_weight * charbonnier_loss(vae_target_recon, target, weight=self.weight)
                if self.compute_latent_reg:
                    for h_posterior in h_posterior_states:
                        loss_latent_reg += block_weight * h_posterior.square().mean() / self.num_groups
                if self.compute_latent_pred:
                    for predicted, posterior in zip(z_coupled, h_posterior_states):
                        loss_latent_pred_rec += block_weight * charbonnier_loss(predicted, posterior.detach())

            group_outputs = self._decode_all_groups(z_coupled, timestamp_target, const=const, lora_step=t + self.out_frames, shortcuts=prior_shortcuts)
            output = self._assemble_prediction(group_outputs, input)
            if self.amse_weight > 0:
                loss_amse += block_weight * self.amse_loss(output, target, channel_weights=self.channel_weights if "channel_weights" in self._buffers else None)
            if self.compute_latent_dyn:
                loss_latent_dyn_consistency += block_weight * field_dynamics_consistency_loss(
                    output=output,
                    target=target,
                    input=input.detach(),
                    target_input=true_input,
                    group_slice_indices=self.group_slice_indices,
                )
            if self.compute_pred_rec:
                field_weight = self.channel_weights * self.weight if "channel_weights" in self._buffers else self.weight
                loss_pred_rec += block_weight * charbonnier_loss(output, target, weight=field_weight)

            if self.pinn_enable:
                for loss_name, loss_value in self.compute_pinn_losses(output, target).items():
                    pinn_acc[loss_name] = pinn_acc.get(loss_name, 0) + block_weight * loss_value
            if self.precip_loss_enable:
                for loss_name, loss_value in self.compute_precip_losses(output, target).items():
                    precip_acc[loss_name] = precip_acc.get(loss_name, 0) + block_weight * loss_value

            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
            true_input = torch.cat([true_input, target], dim=1)[:, -self.in_frames :]

        segment_index += 1
        state.update(
            {
                "input": input.detach(),
                "true_input": true_input,
                "segment_index": segment_index,
                "k": k,
            }
        )
        done = not loss_block_indices.intersection(range(segment_index * segment_size, num_rollout_blocks))

        loss_dict = {}
        if self.compute_pred_rec:
            loss_dict["loss_pred_rec"] = self.loss_weight[2] * loss_pred_rec
        if self.compute_prior_vae:
            loss_dict["loss_vae_input"] = self.loss_weight[0] * loss_vae_input
        if self.compute_posterior_vae:
            loss_dict["loss_vae_target"] = self.loss_weight[1] * loss_vae_target
        if self.compute_latent_reg:
            loss_dict["loss_latent_reg"] = self.loss_weight[3] * loss_latent_reg * kl_weight
        if self.compute_latent_pred:
            loss_dict["loss_latent_pred_rec"] = self.loss_weight[4] * loss_latent_pred_rec
        if self.compute_latent_dyn:
            loss_dict["loss_latent_dyn_consistency"] = self.latent_dyn_weight * loss_latent_dyn_consistency
        if self.amse_weight > 0:
            loss_dict["loss_amse"] = self.amse_weight * loss_amse
        loss_dict.update(pinn_acc)
        loss_dict.update(precip_acc)
        if rollout_segment_size is None:
            if not done:
                raise RuntimeError("Unsegmented rollout did not consume all forecast steps.")
            return loss_dict
        return loss_dict, state, done

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        const = data["const"] if self.const_chans > 0 else None
        if self.pretrain:
            data["targets"] = input.new_zeros((input.shape[0], len(self.test_frames), len(self.test_chans), *input.shape[-2:]))
        total_step = max(self.test_frames) + self.start_frames
        test_frame_to_idx = {int(frame): idx for idx, frame in enumerate(self.test_frames)}
        self.logger.debug(f"Testing cascade: total_step={total_step}, stride={self.forecast_stride}, input.shape={input.shape}")

        hw = list(input.shape[-2:])
        # Match perturb evaluation precision: FP16 accumulation avoids BF16
        # ensemble-reduction error without the persistent cost of FP32.
        outputs = torch.zeros(input.shape[0], len(self.test_frames), len(self.test_chans), *hw, device=input.device, dtype=torch.float16)

        assert total_step <= len(self.ensemble_sizes), f"Total steps {total_step} exceeds precomputed ensemble schedule length {len(self.ensemble_sizes)}"
        ensemble_sizes = self.ensemble_sizes[:total_step]
        effective_members = max(ensemble_sizes)
        ensemble_counts = [0] * len(self.test_frames)

        self.logger.debug(f"Testing with dynamic ensemble size: freq={self.freq}, max_members={effective_members}.")

        for member_idx in range(effective_members):
            new_input = input.clone()
            new_input[~torch.isfinite(new_input)] = 0.0
            for t in range(self.start_frames, total_step, self.forecast_stride):
                if self.pretrain:
                    target = new_input[:, -self.out_frames :]
                    target_input = self._build_posterior_input(new_input[:, : -self.out_frames], target)
                    timestamp_target = self.get_timestamp(new_input, times, t)
                    h_states = self._encode_all_groups(target_input, timestamp_target, const=const, is_prior=False)
                    shortcuts = None
                    if self.patch_decoder_shortcut:
                        h_states, shortcuts = h_states
                    group_outputs = self._decode_all_groups(h_states, timestamp_target, const=const, lora_step=t, shortcuts=shortcuts)
                    target_base = target_input[:, -self.out_frames - 1 : -1]
                    output = self._assemble_prediction(group_outputs, target_base)
                    new_input = torch.cat([new_input[:, :-1], output[:, -1:]], dim=1)
                else:
                    output, new_input = self._predict_next(new_input, times, t, const=const)

                anchor = t - self.start_frames
                step_member_limit = ensemble_sizes[anchor] if anchor < len(ensemble_sizes) else effective_members
                for lead_idx in range(output.shape[1]):
                    lead = anchor + lead_idx + 1
                    out_idx = test_frame_to_idx.get(lead)
                    if out_idx is not None and member_idx < step_member_limit:
                        outputs[:, out_idx] += output[:, lead_idx, self.test_chans].detach()
                        if self.pretrain:
                            # Align each reconstruction position with the original input tail,
                            # including overlapping blocks and their ensemble averaging.
                            data["targets"][:, out_idx] += input[:, lead_idx - self.out_frames, self.test_chans]
                        ensemble_counts[out_idx] += 1
                self.logger.debug(f"Step {t}: output shape={output.shape}, updated outputs shape={outputs.shape}")

        denom = torch.tensor(ensemble_counts, device=outputs.device, dtype=outputs.dtype).view(1, -1, 1, 1, 1)
        denom = torch.clamp(denom, min=1.0)
        outputs = outputs / denom
        if self.pretrain:
            data["targets"] = data["targets"] / denom
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
        next_hour_val = hour + self.out_frames * freq / 24.0
        day_rollover = torch.floor(next_hour_val)

        timestamp_target = dict(
            step=(step + 1).to(new_input),
            hour=(next_hour_val % 1.0).to(new_input),
            doy=((doy + day_rollover * (1.0 / 365.0)) % 1.0).to(new_input),
        )

        h_states = self._encode_all_groups(new_input, timestamp, const=const, is_prior=True)
        shortcuts = None
        if self.patch_decoder_shortcut:
            h_states, shortcuts = h_states

        Z_coupled = self.latent_mixer(h_states, timestamp_target)
        group_outputs = self._decode_all_groups(Z_coupled, timestamp_target, const=const, shortcuts=shortcuts)
        output = self._assemble_prediction(group_outputs, new_input)

        output = self.inv_normalize(output)
        return output


@MODELS.register()
class TongYun(_TongYunBase):
    """Model Architecture Designed by XMetAI.

    See docs/tongyun/README.md for architecture and training details.
    """

    def __init__(
        self,
        group_var_map: dict[str, list[str]] | None = None,
        group_recon_modes: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if group_var_map is None:
            group_var_map = GROUP_VAR_MAP
        if group_recon_modes is None:
            group_recon_modes = GROUP_RECON_MODES
        in_names = list(kwargs.get("in_names", []))
        if len(in_names) == 0:
            raise ValueError("TongYun requires in_names to build feature channel groups.")
        group_slice_indices, _, group_names = build_feature_group_channel_map(in_names, group_var_map)
        group_modes = [group_recon_modes.get(group_name, "absolute") for group_name in group_names]
        invalid_modes = [mode for mode in group_modes if mode not in {"absolute", "residual"}]
        if len(invalid_modes) > 0:
            raise ValueError(f"group_recon_modes must use only 'absolute' or 'residual', got {invalid_modes}")
        residual_vae_indices = [idx for group_indices, mode in zip(group_slice_indices, group_modes) if mode == "residual" for idx in group_indices]
        super().__init__(
            group_slice_indices=group_slice_indices,
            residual_vae_indices=residual_vae_indices,
            **kwargs,
        )
        self.group_modes = group_modes
        self.logger.info(f"TongYun groups: {self._format_group_info(group_slice_indices, group_names, group_modes)}")
        self.logger.info(f"TongYun residual indices: {self.residual_vae_indices}")

    def _format_group_info(self, group_slice_indices: list[list[int]], group_names: list[str], group_modes: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "name": group_name,
                "mode": group_mode,
                "channels": len(group),
                "indices": group,
                "vars": [self.in_names[idx] for idx in group],
            }
            for group_name, group_mode, group in zip(group_names, group_modes, group_slice_indices)
        ]

    def _prepare_group_input(self, x: torch.Tensor, gi: int) -> torch.Tensor:
        x_group = self._slice_group(x, gi)
        if x_group.shape[1] < 2:
            raise ValueError(f"TongYun requires at least two input frames, got shape={tuple(x_group.shape)}")

        mode = self.group_modes[gi]
        if mode == "residual":
            if self.out_frames == 1:
                return torch.cat([x_group[:, :1], x_group[:, 1:] - x_group[:, :1]], dim=1)
            curr = x_group[:, -1:]
            return torch.cat([curr, curr - torch.flip(x_group[:, :-1], dims=[1])], dim=1)
        if mode == "absolute":
            return x_group
        raise ValueError(f"Unexpected feature group reconstruction mode: {mode}")
