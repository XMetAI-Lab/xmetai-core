"""
Extended unit tests for xmetai.layers.embed to increase coverage.
Covers: CubeEmbed stride/pad branch, flatten=False, FieldEmbed conv2d/conv3d,
GeoEmbed forward, and sincos_embedding periodic path.
"""

import torch

from xmetai.layers.embed import CubeEmbed, FieldEmbed, GeoEmbed, sincos_embedding


def test_cube_embed_stride_and_pad_flatten_false():
    """Exercise CubeEmbed stride_size with manual pad and flatten=False."""
    B, T, C_in, H, W = 2, 2, 4, 16, 16
    out_chans = 8
    patch_size = 2
    stride_size = 2  # triggers manual pad path

    x = torch.randn(B, T, C_in, H, W)
    embed = CubeEmbed(
        in_chans=C_in,
        out_chans=out_chans,
        in_frames=T,
        patch_size=patch_size,
        stride_size=stride_size,
        flatten=False,
    )

    y = embed(x)
    # Expected spatial dims after conv with kernel=stride=2
    expected_h = (H + 1 - patch_size) // stride_size + 1
    expected_w = (W + 2 - patch_size) // stride_size + 1
    assert y.shape == (B, out_chans, expected_h, expected_w)


def test_cube_embed_const_addition_no_flatten():
    """Exercise const_proj addition branch without flattening."""
    B, T, H, W = 2, 2, 16, 16
    dyn_chans, const_chans = 5, 3
    out_chans = 8

    x_dynamic = torch.randn(B, T, dyn_chans, H, W)
    x_const = torch.randn(B, const_chans, H, W)

    embed = CubeEmbed(
        in_chans=dyn_chans,
        out_chans=out_chans,
        const_chans=const_chans,
        in_frames=T,
        patch_size=2,
        # Do not set stride_size explicitly to avoid manual pad path
        flatten=False,
    )

    merged = x_dynamic.reshape(B, T * dyn_chans, H, W)
    y = embed(merged, const=x_const)
    assert y.ndim == 4 and y.shape[0] == B and y.shape[1] == out_chans


def test_field_embed_conv2d_with_const_flatten_true():
    """Cover FieldEmbed conv2d path with const and flatten=True."""
    B, T, C, H, W = 2, 2, 10, 16, 16
    pl_chans = 4
    const_chans = 2
    embed_dim = 12

    x = torch.randn(B, T, C, H, W)
    const = torch.randn(const_chans, H, W)

    embed = FieldEmbed(
        in_chans=C,
        pl_chans=pl_chans,
        const_chans=const_chans,
        embed_dim=embed_dim,
        in_frames=T,
        patch_size=4,
        flatten=True,
        proj_type="conv2d",
    )

    y = embed(x, const=const)
    expected_len = (H // 4) * (W // 4)
    assert y.shape == (B, expected_len, embed_dim)


def test_field_embed_conv3d_flatten_true():
    """Cover FieldEmbed conv3d path without const to avoid dim mismatch."""
    B, T, C, H, W = 2, 2, 10, 8, 8
    pl_chans = 2  # ensures sf_proj in_channels matches after conv3d branch
    const_chans = 0
    embed_dim = 12

    x = torch.randn(B, T, C, H, W)

    embed = FieldEmbed(
        in_chans=C,
        pl_chans=pl_chans,
        const_chans=const_chans,
        embed_dim=embed_dim,
        in_frames=T,
        patch_size=2,
        flatten=True,
        proj_type="conv3d",
    )

    y = embed(x)
    expected_len = (H // 2) * (W // 2)
    assert y.shape == (B, expected_len, embed_dim)


def test_geo_embed_forward_flatten_true():
    """Exercise GeoEmbed forward with flatten=True."""
    H, W = 12, 12
    hidden_size = 24
    ge = GeoEmbed(hidden_size=hidden_size, frequency=36, patch_size=3, flatten=True)

    z = torch.randn(H, W)
    y = ge(z)
    expected_len = (H // 3) * (W // 3)
    assert y.shape == (1, expected_len, hidden_size)


def test_sincos_embedding_periodic_path():
    """Cover periodic path for sincos_embedding."""
    x = torch.tensor([0.0, 0.5, 1.0])  # normalized periodic inputs
    emb = sincos_embedding(x, embed_dim=8, is_periodic=True)
    assert emb.shape == (x.numel(), 8)
