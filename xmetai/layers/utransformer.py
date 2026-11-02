import logging
from collections.abc import Sequence
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.layers import (
    CubeEmbed,
    GatedDeltaNet2HybridLayer,
    GatedDeltaNet2Layer,
    LayerNorm,
    MambaVisionLayer,
    MCDropout,
    StepEncoding,
    SwinLayer,
    TimestepEmbed,
    precompute_freqs_cis,
)
from xmetai.layers.pred_head import PLSFHead

__all__ = ["UTransformer"]


class UTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        embed_dim: int | Sequence[int],
        image_size: Sequence[int],
        in_frames: int,
        out_frames: int = 1,
        const_chans: int = 0,
        window_size: int | Sequence[int] | None = 8,
        patch_size: int = 4,
        stride_size: int = -1,
        down_times: int | Sequence[int] = 0,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: int = 8,
        drop_mcmc: float = 0,
        mlp_ratio: float = 4,
        attn_type: str = "v1",
        mask_type: str = "hw",
        head_type: str = "none",
        norm_type: str = "adaln",
        ffn_type: str = "gegluffn",
        patch_norm: bool = True,
        layer_type: str | Sequence[str] = "swin",
        embed_type: str = "all",
        use_year_embed: bool = False,
        up_times: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if norm_type == "adaln":
            norm_layer = partial(LayerNorm, eps=1e-6)
        else:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.const_chans = const_chans
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.embed_dim = embed_dim
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [size // patch_size for size in self.image_size]

        self.num_layers = len(depths)
        self.down_times = down_times
        self.up_times = up_times
        self.hierarchical = not isinstance(down_times, int) or up_times is not None
        if isinstance(embed_dim, int):
            self.stage_dims = [embed_dim] * max(self.num_layers, 1)
        else:
            if not self.hierarchical or len(embed_dim) != len(depths):
                raise ValueError("An embed_dim sequence requires sampling sequences and one width per depth stage.")
            self.stage_dims = list(embed_dim)
        input_dim = self.stage_dims[0]
        output_dim = self.stage_dims[-1]
        self.skip_sources: dict[int, int] = {}
        self.skip_projections = nn.ModuleDict()
        self.stage_sizes = [self.feat_size] * max(self.num_layers, 1)
        if self.hierarchical:
            self._build_sampling(down_times, up_times, stride_size)
        self.norm_type = norm_type
        self.embed_type = embed_type
        self.use_year_embed = use_year_embed
        self.attn_type = attn_type
        self.drop_mcmc = drop_mcmc

        if isinstance(layer_type, str):
            self.layer_types = [layer_type] * self.num_layers
        else:
            if len(layer_type) != self.num_layers:
                raise ValueError(f"layer_type list length ({len(layer_type)}) must match depths length ({self.num_layers})")
            self.layer_types = list(layer_type)

        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            out_chans=input_dim,
            const_chans=const_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            stride_size=stride_size,
            norm_layer=norm_layer if patch_norm else None,
            flatten=True,
        )

        if self.embed_type == "all":
            self.step_embed = TimestepEmbed(input_dim)
            self.hour_embed = TimestepEmbed(input_dim)
            self.doy_embed = TimestepEmbed(input_dim)
            if use_year_embed:
                self.year_embed = TimestepEmbed(input_dim)
        else:
            # for only step embedding.
            self.time_embed = nn.Sequential(
                StepEncoding(12),
                nn.Linear(12, input_dim),
                nn.SiLU(),
                nn.Linear(input_dim, input_dim),
            )

        self.embed_projections = None
        if not isinstance(embed_dim, int):
            self.embed_projections = nn.ModuleList(
                [nn.Identity() if stage_dim == input_dim else nn.Linear(input_dim, stage_dim) for stage_dim in self.stage_dims]
            )

        input_size = self.stage_sizes[0]
        if isinstance(down_times, int) and up_times is None:
            stride = patch_size if stride_size < 1 else stride_size
            input_size = [size // (stride * 2**down_times) for size in self.image_size]
        self.logger.info(
            f"Input size for {self.__class__.__name__}: {input_size}, patch size: {patch_size}, stride size: {stride_size}, down times: {down_times}, up times: {up_times}"
        )

        if "rotary" in attn_type and not self.hierarchical:
            freqs_cos, freqs_sin = precompute_freqs_cis(input_dim // num_heads, int(np.prod(input_size)))
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        layers = []
        for i in range(self.num_layers):
            stage_dim = self.stage_dims[i]
            if self.hierarchical:
                input_size = self.stage_sizes[i]
                if "rotary" in attn_type:
                    stage_cos, stage_sin = precompute_freqs_cis(stage_dim // num_heads, int(np.prod(input_size)))
                    self.register_buffer(f"freqs_cos_{i}", stage_cos, persistent=False)
                    self.register_buffer(f"freqs_sin_{i}", stage_sin, persistent=False)
            current_layer_type = self.layer_types[i]
            layer_lower = current_layer_type.lower()
            if layer_lower.startswith("gdn2"):
                if layer_lower not in {"gdn2", "gdn2_embed", "gdn2_swin", "gdn2_swin_embed"}:
                    raise ValueError(
                        f"Unsupported GatedDeltaNet-2 layer_type={current_layer_type!r}; " "use 'gdn2', 'gdn2_embed', 'gdn2_swin', or 'gdn2_swin_embed'."
                    )
                with_embedding = "embed" in layer_lower
                if "swin" in layer_lower:
                    self.logger.info(f"Using hybrid GatedDeltaNet-2 + Swin blocks with embedding={with_embedding}.")
                    swin_window = input_size if window_size is None else window_size
                    layer = GatedDeltaNet2HybridLayer(
                        dim=stage_dim,
                        depth=depths[i],
                        num_heads=num_heads,
                        input_size=(input_size[0], input_size[1]),
                        window_size=swin_window if isinstance(swin_window, int) else (swin_window[0], swin_window[1]),
                        mlp_ratio=mlp_ratio,
                        drop=drop_mcmc,
                        with_embedding=with_embedding,
                        attn_type=attn_type,
                        mask_type=mask_type,
                        ffn_type=ffn_type,
                    )
                else:
                    self.logger.info(f"Using GatedDeltaNet-2 style transformer blocks with embedding={with_embedding}.")
                    layer = GatedDeltaNet2Layer(
                        dim=stage_dim,
                        depth=depths[i],
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_mcmc,
                        with_embedding=with_embedding,
                    )
            elif layer_lower.find("mamba") > -1:
                with_embedding = "embed" in layer_lower
                mamba_version = "mamba3" if "mamba3" in layer_lower else ("mamba2" if "mamba2" in layer_lower else "mamba")
                mamba_type = f"{mamba_version}_met" if "_met" in layer_lower else mamba_version

                self.logger.info(f"Using {mamba_type} style transformer blocks with embedding={with_embedding}.")
                layer = MambaVisionLayer(
                    mamba_type=mamba_type,
                    dim=stage_dim,
                    depth=depths[i],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=True,
                    drop=drop_mcmc,
                    transformer_blocks=(list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i]))),
                    with_embedding=with_embedding,
                    attn_type=attn_type,
                    window_size=((window_size, window_size) if isinstance(window_size, int) else window_size),
                    input_size=(tuple(input_size) if isinstance(input_size, (list, tuple)) else input_size),
                )
            else:
                self.logger.info("Using Swin style transformer blocks by default.")
                layer = SwinLayer(
                    stage_dim,
                    stage_dim,
                    input_size,
                    window_size,
                    depth=depths[i],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_type=attn_type,
                    mask_type=mask_type,
                    norm_type=norm_type,
                    ffn_type=ffn_type,
                )
            layers.append(layer)
            if self.norm_type != "adaln":
                self.add_module(f"norm{i}", norm_layer(stage_dim))
            self.add_module(f"drop{i}", MCDropout())

        self.layers = nn.ModuleList(layers)

        if norm_type == "adaln" or self.num_layers == 0:
            self.fpn = None
        else:
            self.fpn = nn.Sequential(
                nn.Linear(sum(self.stage_dims), output_dim),
                nn.GELU(),
            )

        if head_type == "1fc":
            self.logger.info(f"Using 1fc head with embed_dim: {embed_dim}, out_chans: {out_chans}, patch_size: {patch_size}")
            self.head = nn.Linear(output_dim, out_chans * out_frames * patch_size * patch_size)
        elif head_type == "2fc":
            self.logger.info(f"Using 2fc head with embed_dim: {embed_dim}, out_chans: {out_chans}, patch_size: {patch_size}")
            self.head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, out_chans * out_frames * patch_size * patch_size),
            )
        elif head_type == "plsf":
            pl_chans = 65
            self.logger.info(f"Using PLSF head with embed_dim: {embed_dim}, pl_chans: {pl_chans}, patch_size: {patch_size}")
            self.head = PLSFHead(
                output_dim,
                pl_chans=pl_chans,
                sfc_chans=in_chans - pl_chans,
                patch_size=(patch_size, patch_size),
                feat_size=self.stage_sizes[-1],
            )
        else:
            self.logger.info(f"Using head type: {head_type} for no head.")
            self.head = None

        self.logger.info(
            f"Initialized {self.__class__.__name__} with args: "
            # Data dimensions
            f"in_chans={in_chans}, out_chans={out_chans}, const_chans={const_chans}, in_frames={in_frames}, out_frames={out_frames}, "
            # Spatial and patch configurations
            f"image_size={image_size}, patch_size={patch_size}, window_size={window_size}, stride_size={stride_size}, down_times={down_times}, up_times={up_times}, "
            # Architecture and layers
            f"embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, mlp_ratio={mlp_ratio}, layer_type={self.layer_types}, "
            # Attention and heads
            f"attn_type={attn_type}, mask_type={mask_type}, head_type={head_type}, drop_mcmc={drop_mcmc}, "
            # Normalization and FFN
            f"norm_type={norm_type}, embed_type={self.embed_type}, patch_norm={patch_norm}, ffn_type={ffn_type}."
        )

    def _build_sampling(self, down_times, up_times, stride_size):
        if isinstance(down_times, int) and down_times != 0:
            raise ValueError("Explicit up_times requires down_times=0 or a sequence of factors.")
        down_factors = [] if isinstance(down_times, int) else list(down_times)
        up_factors = list(up_times or ())
        factors = down_factors + up_factors
        if any(not isinstance(factor, int) or factor < 2 for factor in factors):
            raise ValueError("Sampling factors must be integers >= 2.")
        if self.num_layers != len(factors) + 1:
            raise ValueError("Sampling sequences require len(depths) == len(down_times) + len(up_times) + 1.")
        if stride_size > 0:
            raise ValueError("Sampling sequences require the default patch embedding stride.")

        self.stage_sizes = self._get_stage_sizes(down_factors, up_factors)
        if any(len(size) != 2 or min(size) < 1 for size in self.stage_sizes):
            raise ValueError("Stage grids must contain two positive dimensions.")
        self.resamplers = nn.ModuleList()
        for index, factor in enumerate(factors):
            source = self.skip_sources.get(index + 1)
            if source is not None:
                self.skip_projections[str(index + 1)] = nn.Linear(self.stage_dims[index + 1] + self.stage_dims[source], self.stage_dims[index + 1])
            sampler = nn.Conv2d if index < len(down_factors) else nn.ConvTranspose2d
            self.resamplers.append(sampler(self.stage_dims[index], self.stage_dims[index + 1], factor + 1, stride=factor, padding=1))

    def _get_stage_sizes(self, down_factors, up_factors):
        sizes = [self.feat_size]
        down_stages = []
        for factor in down_factors:
            down_stages.append((factor, len(sizes) - 1))
            sizes.append([(size + 1) // factor for size in sizes[-1]])
        for factor in up_factors:
            if down_stages and down_stages[-1][0] == factor:
                source = down_stages.pop()[1]
                self.skip_sources[len(sizes)] = source
                sizes.append(sizes[source])
            else:
                down_stages.clear()
                sizes.append([size * factor for size in sizes[-1]])
        return sizes

    def forward_features(self, h, timestamp, *, return_stages=False, stage_shortcuts=None, norm=None):
        """Run stages, resample between them, and fuse shortcuts before each block."""
        embed = self.get_embed(timestamp)
        stage_embed = embed
        stage_outputs = []
        skip_features = {}
        stage_shortcuts = {} if stage_shortcuts is None else stage_shortcuts
        skip_stages = set(self.skip_sources.values())
        for stage_index, block in enumerate(self.layers):
            if self.hierarchical and stage_index:
                height, width = self.stage_sizes[stage_index - 1]
                grid = rearrange(h, "b (h w) c -> b c h w", h=height, w=width)
                resampler = self.resamplers[stage_index - 1]
                if isinstance(resampler, nn.ConvTranspose2d):
                    grid = resampler(grid, output_size=self.stage_sizes[stage_index])
                else:
                    grid = resampler(grid)
                h = rearrange(grid, "b c h w -> b (h w) c")
            source_stage = self.skip_sources.get(stage_index)
            shortcut = skip_features.pop(source_stage) if source_stage is not None else stage_shortcuts.get(stage_index)
            if shortcut is not None:
                h = self.skip_projections[str(stage_index)](torch.cat([h, shortcut], dim=-1))
            stage_embed = self.embed_projections[stage_index](embed) if self.embed_projections is not None else embed
            h = getattr(self, f"drop{stage_index}")(h, self.drop_mcmc)
            if "rotary" in self.attn_type:
                cos_name = f"freqs_cos_{stage_index}" if self.hierarchical else "freqs_cos"
                sin_name = f"freqs_sin_{stage_index}" if self.hierarchical else "freqs_sin"
                h = block(h, time_embedding=stage_embed, freqs_cos=getattr(self, cos_name), freqs_sin=getattr(self, sin_name))
            else:
                h = block(h, time_embedding=stage_embed)
            if stage_index in skip_stages:
                skip_features[stage_index] = h
            if return_stages:
                stage_outputs.append(h)
        norm = getattr(self, "norm", None) if norm is None else norm
        if norm is not None:
            h = norm(h, stage_embed)
        return (h, stage_outputs) if return_stages else h

    def resize_input(self, input):
        x = rearrange(input, "n t c h w -> (n t) c h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)
        return x

    def resize_output(self, output, input, unsqueeze=False):
        output = rearrange(
            output,
            "n (h w) (p1 p2 c) -> n c (h p1) (w p2)",
            h=self.stage_sizes[-1][0],
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

        if unsqueeze:
            output = rearrange(output, "n (t c) h w -> n t c h w", t=self.out_frames)
        return output

    def get_embed(self, timestamp):
        if self.embed_type == "all":
            embed = self.step_embed(timestamp["step"])
            if "hour" in timestamp:
                embed += self.hour_embed(timestamp["hour"])
            if "doy" in timestamp:
                embed += self.doy_embed(timestamp["doy"])
            if self.use_year_embed:
                embed += self.year_embed(timestamp["year"])
            return embed
        else:
            return self.time_embed(timestamp)

    def forward(self, input):
        return NotImplemented
