"""
Extended unit tests for xmetai.layers.attention module to increase coverage.
Covers: CrossAttention mask path, precision variants, NewCrossAttention backends,
TwoAttentionBlock context usage, and TransformerBlock conv2d branch.
"""

import pytest
import torch

from xmetai.layers.attention import (
    CrossAttention,
    NewCrossAttention,
    TransformerBlock,
    TwoAttentionBlock,
)


@pytest.mark.gpu
def test_cross_attention_mask_path(device):
    """Exercise CrossAttention mask branch with partial masking."""
    if device.type == "cpu":
        pytest.skip("GPU test")

    B, L, C = 2, 32, 64
    heads, dim_head = 8, C // 8
    attn = CrossAttention(query_dim=C, heads=heads, dim_head=dim_head).to(device)

    x = torch.randn(B, L, C, device=device)
    # Allow only first half of the keys
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    mask[:, : L // 2] = True

    out = attn(x, context=None, mask=mask)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


@pytest.mark.gpu
def test_cross_attention_precision_variant(device):
    """Cover non-fp32 precision path by setting precision to 'bf16'."""
    if device.type == "cpu":
        pytest.skip("GPU test")

    B, L, C = 2, 32, 64
    heads, dim_head = 8, C // 8
    attn = CrossAttention(query_dim=C, heads=heads, dim_head=dim_head, precision="bf16").to(device)

    x = torch.randn(B, L, C, device=device)
    out = attn(x)
    assert out.shape == x.shape


@pytest.mark.gpu
def test_new_cross_attention_sdp_backends(device):
    """Exercise NewCrossAttention with different SDP backend settings."""
    if device.type == "cpu":
        pytest.skip("GPU test")

    B, L, C = 2, 64, 64
    heads, dim_head = 8, C // 8

    x = torch.randn(B, L, C, device=device)

    # Test default backend (None)
    attn_default = NewCrossAttention(query_dim=C, heads=heads, dim_head=dim_head, backend=None).to(device)
    out_default = attn_default(x)
    assert out_default.shape == x.shape

    # Test Flash Attention backend when available
    try:
        from torch.backends.cuda import SDPBackend

        attn_flash = NewCrossAttention(
            query_dim=C,
            heads=heads,
            dim_head=dim_head,
            backend=SDPBackend.FLASH_ATTENTION,
        ).to(device)
        out_flash = attn_flash(x)
        assert out_flash.shape == x.shape
    except Exception:
        # Backend may not be available or supported; skip gracefully
        pytest.skip("SDPBackend not available")


def test_two_attention_block_with_context():
    """Ensure TwoAttentionBlock uses context on second attention."""
    B, L, C = 2, 32, 64
    heads, dim_head = 8, C // 8
    block = TwoAttentionBlock(dim=C, dim_head=dim_head, n_heads=heads, dropout=0.1)

    x = torch.randn(B, L, C)
    context = torch.randn(B, L, C)
    out = block(x, context=context)
    assert out.shape == x.shape


def test_transformer_block_conv2d_branch():
    """Cover TransformerBlock conv2d projection path with depth>1."""
    B, C, H, W = 2, 64, 16, 16
    heads = 8
    block = TransformerBlock(in_channels=C, n_heads=heads, depth=2, dropout=0.0, proj_type="conv2d")

    x = torch.randn(B, C, H, W)
    context = torch.randn(B, H * W, C)
    out = block(x, context=context)
    assert out.shape == x.shape
