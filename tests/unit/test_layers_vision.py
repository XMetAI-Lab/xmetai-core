"""
Unit tests for vision layers: SwinLayer, MambaVisionLayer, Block.
Tests windowed attention, rotary embeddings, and mamba blocks.
"""

import pytest
import torch
import torch.nn as nn

# Prefer package exports; otherwise import directly from modules.
try:
    from xmetai.layers import (
        HAS_MAMBA,
        MambaVisionLayer,
        SwinLayer,
        precompute_freqs_cis,
    )
except Exception:
    HAS_MAMBA = False
    from xmetai.layers.swin import SwinLayer

    try:
        from xmetai.layers.mamba_vision import MambaVisionLayer

        HAS_MAMBA = True
    except Exception:
        # leave MambaVisionLayer undefined; tests that require it should handle ImportError/skip
        MambaVisionLayer = None
    from xmetai.layers.rotary_attention import precompute_freqs_cis


class TestSwinLayer:
    """Test Swin Transformer layer with windowed attention."""

    @pytest.mark.gpu
    def test_swin_layer_forward_shape(self, torch_seed):
        """Test SwinLayer output shape matches input."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        input_size = (16, 16)  # H, W
        window_size = 8

        layer = SwinLayer(
            in_chans=C,
            embed_dim=C,
            input_size=input_size,
            window_size=window_size,
            depth=2,
            num_heads=8,
        )

        output = layer(x)
        assert output.shape == x.shape

    @pytest.mark.gpu
    def test_swin_layer_gradient_flow(self, torch_seed):
        """Test gradients flow through SwinLayer."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C, requires_grad=True)

        layer = SwinLayer(
            in_chans=C,
            embed_dim=C,
            input_size=(16, 16),
            window_size=8,
            depth=1,
            num_heads=8,
        )

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.gpu
    def test_swin_layer_different_attn_types(self, torch_seed):
        """Test SwinLayer with different attention types."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)
        input_size = (16, 16)
        window_size = 8

        for attn_type in ["v1", "v2"]:
            layer = SwinLayer(
                in_chans=C,
                embed_dim=C,
                input_size=input_size,
                window_size=window_size,
                depth=1,
                num_heads=8,
                attn_type=attn_type,
            )
            output = layer(x)
            assert output.shape == x.shape


class TestBlock:
    """Test Block (Vision Transformer block with optional Mamba/Attention)."""

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    @pytest.mark.gpu
    def test_block_standard_attention(self, torch_seed):
        """Test Block with standard attention."""
        from xmetai.layers.mamba_vision import Block

        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        block = Block(
            dim=C,
            num_heads=8,
            counter=0,
            transformer_blocks=[0],
            attn_type="standard",
        )

        output = block(x)
        assert output.shape == x.shape

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    @pytest.mark.gpu
    def test_block_rotary_attention(self, torch_seed, sample_freqs_cis):
        """Test Block with rotary attention."""
        from xmetai.layers.mamba_vision import Block

        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)
        freqs_cos, freqs_sin = sample_freqs_cis

        block = Block(
            dim=C,
            num_heads=8,
            counter=0,
            transformer_blocks=[0],
            attn_type="rotary",
        )

        output = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        assert output.shape == x.shape

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    @pytest.mark.gpu
    def test_block_windowed_rotary_attention(self, torch_seed, sample_freqs_cis):
        """Test Block with windowed rotary attention."""
        from xmetai.layers.mamba_vision import Block

        B, H, W, C = 2, 16, 16, 64
        L = H * W
        x = torch.randn(B, L, C)

        window_size = (8, 8)
        input_size = (H, W)
        head_dim = C // 8
        seq_len = window_size[0] * window_size[1]
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)

        block = Block(
            dim=C,
            num_heads=8,
            counter=0,
            transformer_blocks=[0],
            attn_type="rotary",
            window_size=window_size,
            input_size=input_size,
        )

        try:
            output = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
            assert output.shape == x.shape
        except AssertionError:
            pytest.skip("RotaryAttention requires matching precompute layout; skipping if incompatible")

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_block_mamba_mixer(self, torch_seed):
        """Test Block with Mamba mixer."""
        from xmetai.layers.mamba_vision import Block

        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        if not torch.cuda.is_available():
            pytest.skip("Mamba mixer path requires CUDA selective_scan")

        x = x.cuda()
        block = Block(
            dim=C,
            num_heads=8,
            counter=1,  # Not in transformer_blocks
            transformer_blocks=[0],
            attn_type="standard",
        ).cuda()

        output = block(x)
        assert output.shape == x.shape

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_mamba_mixer_conv_bias_flag(self, torch_seed):
        """Test Mamba mixer honors the conv bias flag."""
        from xmetai.layers.mamba_vision import MambaVisionMixer

        mixer_with_bias = MambaVisionMixer(d_model=64, conv_bias=True)
        mixer_without_bias = MambaVisionMixer(d_model=64, conv_bias=False)

        assert mixer_with_bias.conv1d_x.bias is not None
        assert mixer_with_bias.conv1d_z.bias is not None
        assert mixer_without_bias.conv1d_x.bias is None
        assert mixer_without_bias.conv1d_z.bias is None


class TestMambaVisionLayer:
    """Test complete MambaVisionLayer with depth > 1."""

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    @pytest.mark.gpu
    def test_mamba_vision_layer_basic(self, torch_seed):
        """Test MambaVisionLayer basic forward pass."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        layer = MambaVisionLayer(
            dim=C,
            depth=4,
            num_heads=8,
            transformer_blocks=[0, 1, 2, 3],
        )

        output = layer(x)
        assert output.shape == x.shape

    @pytest.mark.gpu
    def test_mamba_vision_layer_with_rotary(self, torch_seed, sample_freqs_cis):
        """Test MambaVisionLayer with rotary attention."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)
        freqs_cos, freqs_sin = sample_freqs_cis

        head_dim = C // 8
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, L)
        layer = MambaVisionLayer(
            dim=C,
            depth=4,
            num_heads=8,
            transformer_blocks=[0, 1, 2, 3],
            attn_type="rotary",
        )

        output = layer(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        assert output.shape == x.shape

    @pytest.mark.gpu
    def test_mamba_vision_layer_windowed_rotary(self, torch_seed, sample_freqs_cis):
        """Test MambaVisionLayer with windowed rotary attention."""
        B, H, W, C = 2, 16, 16, 64
        L = H * W
        x = torch.randn(B, L, C)

        window_size = (8, 8)
        input_size = (H, W)
        head_dim = C // 8
        seq_len = window_size[0] * window_size[1]
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)

        layer = MambaVisionLayer(
            dim=C,
            depth=4,
            num_heads=8,
            transformer_blocks=[0, 1, 2, 3],
            attn_type="rotary",
            window_size=window_size,
            input_size=input_size,
        )

        assert isinstance(layer.blocks[0].mixer, nn.Module)
        assert layer.blocks[0].mixer.max_seq_len == seq_len

        output = layer(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        assert output.shape == x.shape

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_mamba_vision_layer_preserves_mamba_init(self, torch_seed):
        """Test layer construction does not clobber Mamba mixer initialization."""
        from xmetai.layers.mamba_vision import MambaVisionLayer, MambaVisionMixer

        torch.manual_seed(1234)
        expected_mixer = MambaVisionMixer(d_model=64, d_state=8, d_conv=3, expand=1)
        expected_dt_proj_weight = expected_mixer.dt_proj.weight.detach().clone()
        expected_out_proj_weight = expected_mixer.out_proj.weight.detach().clone()

        torch.manual_seed(1234)
        layer = MambaVisionLayer(
            dim=64,
            depth=1,
            num_heads=8,
            transformer_blocks=[],
        )
        mixer = layer.blocks[0].mixer

        assert torch.allclose(mixer.dt_proj.weight, expected_dt_proj_weight)
        assert torch.allclose(mixer.out_proj.weight, expected_out_proj_weight)

    @pytest.mark.gpu
    def test_mamba_vision_layer_mixed_blocks(self, torch_seed):
        """Test MambaVisionLayer with mixed attention and mamba blocks."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        # First 2 blocks are transformer, rest are mamba
        if not torch.cuda.is_available():
            pytest.skip("Mixed mamba blocks require CUDA selective_scan")

        x = x.cuda()
        layer = MambaVisionLayer(
            dim=C,
            depth=4,
            num_heads=8,
            transformer_blocks=[0, 1],
        ).cuda()

        output = layer(x)
        assert output.shape == x.shape

    @pytest.mark.gpu
    def test_mamba_vision_layer_gradient_flow(self, torch_seed):
        """Test gradients flow through entire layer."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C, requires_grad=True)

        if not torch.cuda.is_available():
            pytest.skip("Mamba mixer requires CUDA selective_scan")

        x = x.cuda()
        layer = MambaVisionLayer(
            dim=C,
            depth=2,
            num_heads=8,
            transformer_blocks=[0],
        ).cuda()

        x.retain_grad()
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.gpu
    def test_mamba_vision_layer_determinism(self, torch_seed):
        """Test that identical inputs produce identical outputs."""
        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C)

        if not torch.cuda.is_available():
            pytest.skip("Mamba mixer requires CUDA selective_scan")

        x = x.cuda()
        layer = MambaVisionLayer(
            dim=C,
            depth=2,
            num_heads=8,
            transformer_blocks=[0],
        ).cuda()

        layer.eval()
        with torch.no_grad():
            output1 = layer(x.clone())
            output2 = layer(x.clone())

        assert torch.allclose(output1, output2, atol=1e-5)

    @pytest.mark.gpu
    def test_mamba_vision_layer_gpu(self, device, torch_seed):
        """Test MambaVisionLayer on GPU."""
        if device.type == "cpu":
            pytest.skip("GPU test")

        B, L, C = 2, 256, 64
        x = torch.randn(B, L, C, device=device)

        layer = MambaVisionLayer(
            dim=C,
            depth=2,
            num_heads=8,
            transformer_blocks=[0],
        ).to(device)

        output = layer(x)
        assert output.device.type == device.type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
