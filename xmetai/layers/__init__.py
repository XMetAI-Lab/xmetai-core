from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .activation import GEGLU, TruncExp, trunc_exp
from .activation import __all__ as _activation_all
from .afnonet_blocks import AFNO2D, Block
from .afnonet_blocks import Block as AFNOBlock
from .afnonet_blocks import PatchEmbed as AfnoPatchEmbed
from .afnonet_blocks import __all__ as _afnonet_blocks_all
from .attention import Attention, CrossAttention, GroupWindowAttention, WindowAttention, WindowAttentionV2
from .attention import __all__ as _attention_all
from .drop import DropBlock2d, DropPath, MCDropout, drop_block_2d, drop_block_fast_2d, drop_path
from .drop import __all__ as _drop_all
from .earth_conv import EarthConv2d, EarthConvTranspose2d, EarthDeconv2d, earth_pad
from .earth_conv import __all__ as _earth_conv_all
from .embed import CubeEmbed, FieldEmbed, GeoEmbed, PatchEmbed, TimestepEmbed, sincos_embedding
from .embed import __all__ as _embed_all
from .encoding import StepEncoding, TimeEncoding
from .encoding import __all__ as _encoding_all
from .graphcast_blocks import GraphCastDecoder, GraphCastEncoder, GraphCastLayer
from .graphcast_blocks import __all__ as _graphcast_blocks_all
from .helpers import GroupNorm32, append_dims, default, exists, kaming_init, normalization, to_1tuple, to_2tuple, to_3tuple, to_4tuple, to_ntuple, zero_module
from .helpers import __all__ as _helpers_all
from .lora import LinearLoRA, apply_lora_to_model, merge_lora_weights, set_lora_step, unmerge_lora_weights
from .lora import __all__ as _lora_all
from .mlp import GeGLUFFN, Mlp
from .mlp import __all__ as _mlp_all
from .norm import AdaptiveGroupNorm, AdaptiveGroupNorm2D, AdaptiveInstanceNorm, AdaptiveLN, FrozenBatchNorm2d, LayerNorm, PixelNorm, get_norm
from .norm import __all__ as _norm_all
from .rotary_attention import RotaryAttention, apply_rotary_emb, precompute_freqs_cis, precompute_freqs_cis_2d, repeat_kv, reshape_for_broadcast
from .rotary_attention import __all__ as _rotary_attention_all
from .swin import SwinBlock, SwinLayer, window_partition, window_reverse
from .swin import __all__ as _swin_all
from .weight_init import __all__ as _weight_init_all
from .weight_init import (
    c2_msra_fill,
    c2_xavier_fill,
    constant_fill,
    ddpm_copy,
    ddpm_fill,
    lecun_normal_,
    trunc_normal_,
    variance_scaling_,
    vit_fill,
    xavier_fill,
)
from .wrappers import Conv2d, ModulatedConv2d, PeriodicConv2d, aligned_bilinear, cat, cross_entropy, nonzero_tuple
from .wrappers import __all__ as _wrappers_all

HAS_MAMBA = False
if find_spec("mamba_ssm") is not None:
    HAS_MAMBA = True

MambaVisionLayer: Any = None
if HAS_MAMBA:
    _mamba_vision = import_module(".mamba_vision", package=__name__)
    MambaVisionLayer = _mamba_vision.MambaVisionLayer

_mamba_vision_all = ["MambaVisionLayer"]

__all__ = [
    "HAS_MAMBA",
    *_activation_all,
    *[name for name in _afnonet_blocks_all if name != "PatchEmbed"],
    "AFNOBlock",
    "AfnoPatchEmbed",
    *_attention_all,
    *_drop_all,
    *_earth_conv_all,
    *_embed_all,
    *_encoding_all,
    *_graphcast_blocks_all,
    *_helpers_all,
    *_lora_all,
    *_mlp_all,
    *_norm_all,
    *_rotary_attention_all,
    *_swin_all,
    *_weight_init_all,
    *_wrappers_all,
    *_mamba_vision_all,
]

del (
    _activation_all,
    _afnonet_blocks_all,
    _attention_all,
    _drop_all,
    _earth_conv_all,
    _embed_all,
    _encoding_all,
    _graphcast_blocks_all,
    _helpers_all,
    _lora_all,
    _mlp_all,
    _norm_all,
    _rotary_attention_all,
    _swin_all,
    _weight_init_all,
    _wrappers_all,
    _mamba_vision_all,
    import_module,
    find_spec,
)
if HAS_MAMBA:
    del _mamba_vision
