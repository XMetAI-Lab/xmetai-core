import logging
from functools import partial

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.layers import (
    HAS_MAMBA,
    CubeEmbed,
    LayerNorm,
    MCDropout,
    SwinLayer,
    TimestepEmbed,
    precompute_freqs_cis,
)

if HAS_MAMBA:
    from xmetai.layers import MambaVisionLayer

from .pred_head import PLSFHead

__all__ = ["UTransformer"]


class DeltaNetLayer(nn.Module):
    def __init__(
        self,
        mode,
        hidden_size,
        num_heads,
        depth,
        expand_k,
        expand_v,
        use_gate=False,
        use_short_conv=False,
        conv_size=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        from fla.layers import DeltaNet

        for i in range(depth):
            blk = DeltaNet(
                mode=mode,
                hidden_size=hidden_size,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                use_gate=use_gate,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                layer_idx=i,
            )
            self.blocks.append(blk)

    def forward(self, h, time_embedding=None, freqs_cos=None, freqs_sin=None):
        past_key_values = None
        for blk in self.blocks:
            h, _, past_key_values = blk(h, past_key_values=past_key_values)
        return h


class UTransformer(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        in_frames,
        embed_dim,
        image_size,
        const_chans=0,
        window_size=8,
        patch_size=4,
        stride_size=-1,
        down_times=0,
        depths=[6, 6, 6, 6],
        num_heads=8,
        drop_mcmc=0,
        mlp_ratio=4,
        attn_type="v1",
        mask_type="hw",
        head_type="none",
        norm_type="adaln",
        ffn_type="gegluffn",
        patch_norm=True,
        layer_type="swin",
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if norm_type == "adaln":
            norm_layer = partial(LayerNorm, eps=1e-6)
        else:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.in_chans = in_chans
        self.in_frames = in_frames
        self.out_chans = out_chans
        self.const_chans = const_chans
        self.embed_dim = embed_dim
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times
        self.norm_type = norm_type
        self.attn_type = attn_type
        self.drop_mcmc = drop_mcmc

        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            out_chans=embed_dim,
            const_chans=const_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            stride_size=stride_size,
            norm_layer=norm_layer if patch_norm else None,
            flatten=True,
        )

        if self.norm_type == "adaln":
            self.step_embed = TimestepEmbed(embed_dim)
            self.hour_embed = TimestepEmbed(embed_dim)
            self.doy_embed = TimestepEmbed(embed_dim)
        else:
            # for only step embedding.
            self.time_embed = nn.Sequential(
                nn.Linear(12, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        if stride_size < 1:
            input_size = [sz // (patch_size * (2**down_times)) for sz in image_size]
        else:
            input_size = [sz // (stride_size * (2**down_times)) for sz in image_size]
        self.logger.info(
            f"Input size for {self.__class__.__name__}: {input_size}, patch size: {patch_size}, stride size: {stride_size}, down times: {down_times}"
        )

        if "rotary" in attn_type:
            freqs_cos, freqs_sin = precompute_freqs_cis(embed_dim // num_heads, np.prod(input_size))
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        layers = []
        for i in range(self.num_layers):
            if layer_type.lower().find("mamba") > -1:
                if not HAS_MAMBA:
                    raise ImportError("MambaVisionLayer requires mamba-ssm. " "Install it with: pip install '.[gpu]' or pip install mamba-ssm")
                layer_lower = layer_type.lower()
                with_embedding = "embed" in layer_lower
                mamba_version = "mamba3" if "mamba3" in layer_lower else ("mamba2" if "mamba2" in layer_lower else "mamba")
                mamba_type = f"{mamba_version}_met" if "_met" in layer_lower else mamba_version

                self.logger.info(f"Using {mamba_type} style transformer blocks with embedding={with_embedding}.")
                layer = MambaVisionLayer(
                    mamba_type=mamba_type,
                    dim=embed_dim,
                    depth=depths[i],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=True,
                    conv=False,
                    drop=drop_mcmc,
                    transformer_blocks=(list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i]))),
                    with_embedding=with_embedding,
                    attn_type=attn_type,
                    window_size=((window_size, window_size) if isinstance(window_size, int) else window_size),
                    input_size=(tuple(input_size) if isinstance(input_size, (list, tuple)) else input_size),
                )
            elif layer_type.lower().find("deltanet") > -1:
                self.logger.info("Using DeltaNet style transformer blocks.")
                layer = DeltaNetLayer(
                    mode="chunk",
                    hidden_size=embed_dim,
                    num_heads=num_heads,
                    expand_k=mlp_ratio,
                    expand_v=mlp_ratio,
                    use_gate=True,
                    use_short_conv=True,
                    depth=depths[i],
                )
            else:
                self.logger.info("Using Swin style transformer blocks by default.")
                layer = SwinLayer(
                    embed_dim,
                    embed_dim,
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
                self.add_module(f"norm{i}", norm_layer(embed_dim))
            self.add_module(f"drop{i}", MCDropout())

        self.layers = nn.ModuleList(layers)

        if norm_type == "adaln":
            self.fpn = None
        else:
            self.fpn = nn.Sequential(
                nn.Linear(embed_dim * self.num_layers, embed_dim),
                nn.GELU(),
            )

        if head_type == "1fc":
            self.logger.info(f"Using 1fc head with embed_dim: {embed_dim}, out_chans: {out_chans}, patch_size: {patch_size}")
            self.head = nn.Linear(embed_dim, out_chans * patch_size * patch_size)
        elif head_type == "2fc":
            self.logger.info(f"Using 2fc head with embed_dim: {embed_dim}, out_chans: {out_chans}, patch_size: {patch_size}")
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, out_chans * patch_size * patch_size),
            )
        elif head_type == "plsf":
            pl_chans = 65
            self.logger.info(f"Using PLSF head with embed_dim: {embed_dim}, pl_chans: {pl_chans}, patch_size: {patch_size}")
            self.head = PLSFHead(
                embed_dim,
                pl_chans=pl_chans,
                sfc_chans=in_chans - pl_chans,
                patch_size=(patch_size, patch_size),
                feat_size=self.feat_size,
            )
        else:
            self.logger.error(f"Unknown head type: {head_type}.")
            self.head = None

    def resize_input(self, input):
        x = rearrange(input, "n t c h w -> (n t) c h w")

        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)

        return x

    def resize_output(self, output, input):
        output = rearrange(
            output,
            "n (h w) (p1 p2 c) -> n c (h p1) (w p2)",
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

        return output

    def get_embed(self, timestamp):
        embed = self.step_embed(timestamp["step"])
        if "hour" in timestamp:
            embed += self.hour_embed(timestamp["hour"])
        if "doy" in timestamp:
            embed += self.doy_embed(timestamp["doy"])
        return embed

    def forward(self, input):
        return NotImplemented
