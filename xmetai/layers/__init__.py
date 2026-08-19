from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .activation import GEGLU, SquaredReLU, SwiGLU, trunc_exp
from .afnonet_blocks import Block as AFNOBlock
from .afnonet_blocks import PatchEmbed as AfnoPatchEmbed
from .darknet_unet import DarknetConvUNet2D
from .drop import DropPath, MCDropout
from .earth_conv import EarthConv2d, EarthConvTranspose2d
from .embed import CubeEmbed, TimestepEmbed
from .encoding import CyclicTimesteps, StepEncoding, TimestepProjection
from .gated_deltanet2 import GatedDeltaNet2Block, GatedDeltaNet2HybridLayer, GatedDeltaNet2Layer, GatedDeltaNet2Mixer
from .graphcast_blocks import GraphCastDecoder, GraphCastEncoder, GraphCastLayer
from .helpers import kaming_init, to_1tuple
from .lora import apply_lora_to_model, merge_lora_weights, set_lora_step
from .norm import AdaptiveLN, AdaRMSNorm, LayerNorm, RMSNorm
from .puyun_blocks import ConvLayerNorm, PuYunBlock
from .rotary_attention import precompute_freqs_cis, precompute_freqs_cis_3d, precompute_freqs_cis_3d_remapped
from .swin import SwinLayer
from .tianquan_blocks import (
    FGA,
    SCA,
    DEConv,
    PEDadd_UViT,
    Perceiver3DDecoder,
    Perceiver3DEncoder,
)

HAS_MAMBA = find_spec("mamba_ssm") is not None
HAS_NATTEN = find_spec("natten") is not None
HAS_TORCH_HARMONICS = find_spec("torch_harmonics") is not None

MambaVisionLayer: Any = None
if HAS_MAMBA:
    _mamba_vision = import_module(".mamba_vision", package=__name__)
    MambaVisionLayer = _mamba_vision.MambaVisionLayer

EarthSpectralConv2d: Any = None
if HAS_TORCH_HARMONICS:
    from .earth_spectral_conv import EarthSpectralConv2d as _EarthSpectralConv2d

    EarthSpectralConv2d = _EarthSpectralConv2d

NattenSelfAttention3D: Any = None
NattenSelfAttention3DRemap: Any = None
GlobalSelfAttention3D: Any = None
GlobalSelfAttention3DRemap: Any = None
TransformerLayer3D: Any = None
if HAS_NATTEN:
    _natten = import_module(".natten", package=__name__)
    NattenSelfAttention3D = _natten.NattenSelfAttention3D
    GlobalSelfAttention3D = _natten.GlobalSelfAttention3D
    TransformerLayer3D = _natten.TransformerLayer3D

__all__ = [
    "AFNOBlock",
    "AdaRMSNorm",
    "AdaptiveLN",
    "AfnoPatchEmbed",
    "ConvLayerNorm",
    "CubeEmbed",
    "CyclicTimesteps",
    "DarknetConvUNet2D",
    "DropPath",
    "EarthConv2d",
    "EarthConvTranspose2d",
    "EarthSpectralConv2d",
    "GEGLU",
    "GatedDeltaNet2Block",
    "GatedDeltaNet2HybridLayer",
    "GatedDeltaNet2Layer",
    "GatedDeltaNet2Mixer",
    "GlobalSelfAttention3D",
    "GraphCastDecoder",
    "GraphCastEncoder",
    "GraphCastLayer",
    "HAS_MAMBA",
    "HAS_NATTEN",
    "HAS_TORCH_HARMONICS",
    "LayerNorm",
    "MCDropout",
    "MambaVisionLayer",
    "NattenSelfAttention3D",
    "PuYunBlock",
    "RMSNorm",
    "StepEncoding",
    "SquaredReLU",
    "SwinLayer",
    "SwiGLU",
    "TimestepEmbed",
    "Perceiver3DEncoder",
    "Perceiver3DDecoder",
    "PEDadd_UViT",
    "SCA",
    "DEConv",
    "FGA",
    "TimestepProjection",
    "TransformerLayer3D",
    "apply_lora_to_model",
    "kaming_init",
    "merge_lora_weights",
    "precompute_freqs_cis",
    "precompute_freqs_cis_3d",
    "precompute_freqs_cis_3d_remapped",
    "set_lora_step",
    "to_1tuple",
    "trunc_exp",
]

del import_module, find_spec
if HAS_MAMBA:
    del _mamba_vision
if HAS_TORCH_HARMONICS:
    del _EarthSpectralConv2d
if HAS_NATTEN:
    del _natten
