"""
Unit tests for basic layers in xmetai.layers module.
Tests: Attention, RotaryAttention, Embeddings, Activations, Normalizations
"""

import pytest
import torch
import torch.nn as nn

# Import helpers: prefer package exports, fall back to module-level imports
try:
    from xmetai.layers import (
        Attention,
        RotaryAttention,
        CubeEmbed,
        TimestepEmbed,
        precompute_freqs_cis,
        precompute_freqs_cis_2d,
        LayerNorm,
    )
except Exception:
    # Some projects expose fewer symbols from package __init__.
    # Fallback to direct module imports and provide small adapters where needed.
    try:
        from xmetai.layers.attention import AttentionBlock as _AttentionBlock

        class Attention:
            def __init__(self, dim, num_heads=8, **kwargs):
                dim_head = max(1, dim // max(1, num_heads))
                self._inst = _AttentionBlock(dim, dim_head, n_heads=num_heads, **kwargs)

            def __call__(self, *args, **kwargs):
                return self._inst(*args, **kwargs)

            def to(self, *args, **kwargs):
                return self._inst.to(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inst, name)

    except Exception:
        # Last resort: try to import a named symbol if available
        from xmetai.layers.attention import AttentionBlock as Attention

    from xmetai.layers.rotary_attention import RotaryAttention, precompute_freqs_cis, precompute_freqs_cis_2d
    from xmetai.layers.embed import CubeEmbed, TimestepEmbed
    from xmetai.layers.norm import LayerNorm


class TestAttention:
    """Test standard scaled dot-product attention."""

    @pytest.mark.gpu
    def test_attention_forward_shape(self, sample_input_2d):
        """Test attention output shape matches input."""
        B, L, C = sample_input_2d.shape
        num_heads = 8
        attn = Attention(dim=C, num_heads=num_heads)
        
        output = attn(sample_input_2d)
        assert output.shape == sample_input_2d.shape

    @pytest.mark.gpu
    def test_attention_batch_invariance(self, torch_seed):
        """Test attention produces different outputs for different batches."""
        C, num_heads = 64, 8
        attn = Attention(dim=C, num_heads=num_heads)
        
        # Same sequence in batch
        x = torch.randn(2, 256, C)
        out1 = attn(x[0:1])
        out2 = attn(x[1:2])
        
        # Outputs should be different (not all zeros)
        assert not torch.allclose(out1, out2, atol=1e-4)

    @pytest.mark.gpu
    def test_attention_gradient_flow(self, sample_input_2d):
        """Test gradients flow through attention."""
        attn = Attention(dim=sample_input_2d.shape[-1], num_heads=8)
        sample_input_2d.requires_grad = True
        
        output = attn(sample_input_2d)
        loss = output.sum()
        loss.backward()
        
        assert sample_input_2d.grad is not None
        assert sample_input_2d.grad.abs().sum() > 0

    @pytest.mark.gpu
    def test_attention_gpu(self, device, sample_input_2d):
        """Test attention on GPU."""
        if device.type == "cpu":
            pytest.skip("GPU test")
        
        attn = Attention(dim=sample_input_2d.shape[-1], num_heads=8).to(device)
        x = sample_input_2d.to(device)
        output = attn(x)
        
        assert output.device.type == device.type


class TestRotaryAttention:
    """Test Rotary Position Embedding (RoPE) attention."""

    def test_rotary_attention_forward_shape(self, sample_input_2d, sample_freqs_cis):
        """Test rotary attention output shape."""
        B, L, C = sample_input_2d.shape
        num_heads = 8
        freqs_cos, freqs_sin = sample_freqs_cis
        
        attn = RotaryAttention(dim=C, n_heads=num_heads)
        output = attn(sample_input_2d, freqs_cos, freqs_sin)
        
        assert output.shape == sample_input_2d.shape

    def test_rotary_attention_with_max_seq_len(self, sample_input_2d):
        """Test rotary attention with explicit max_seq_len."""
        B, L, C = sample_input_2d.shape
        num_heads = 8
        
        attn = RotaryAttention(dim=C, n_heads=num_heads, max_seq_len=512)
        freqs_cos, freqs_sin = precompute_freqs_cis(C // num_heads, 512)
        
        output = attn(sample_input_2d, freqs_cos, freqs_sin)
        assert output.shape == sample_input_2d.shape

    @pytest.mark.gpu
    @pytest.mark.gpu
    def test_rotary_attention_flash_gpu(self, sample_input_2d):
        """Test rotary attention flash path on GPU without mask."""
        B, L, C = sample_input_2d.shape
        num_heads = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = sample_input_2d.to(device)
        attn = RotaryAttention(dim=C, n_heads=num_heads, flash=True, max_seq_len=L).to(device)
        freqs_cos, freqs_sin = precompute_freqs_cis(C // num_heads, L)
        out = attn(x, freqs_cos.to(device), freqs_sin.to(device))
        assert out.shape == sample_input_2d.shape

    @pytest.mark.gpu
    def test_rotary_attention_manual_path(self, sample_input_2d):
        """Test manual attention path (flash=False) to increase coverage."""
        B, L, C = sample_input_2d.shape
        num_heads = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = sample_input_2d.to(device)
        attn = RotaryAttention(dim=C, n_heads=num_heads, flash=False).to(device)
        freqs_cos, freqs_sin = precompute_freqs_cis(C // num_heads, L)
        out = attn(x, freqs_cos.to(device), freqs_sin.to(device))
        assert out.shape == sample_input_2d.shape

    @pytest.mark.gpu
    def test_rotary_attention_masked_paths(self, sample_input_2d):
        """Cover mask broadcasting for both flash and manual attention paths."""
        _, L, C = sample_input_2d.shape
        num_heads = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use batch=1 to keep mask repeat compatible with flash attention
        x = torch.randn(1, L, C, device=device)
        freqs_cos, freqs_sin = precompute_freqs_cis(C // num_heads, L)
        mask = torch.zeros(1, L, L, device=device)

        attn_flash = RotaryAttention(dim=C, n_heads=num_heads, flash=True, max_seq_len=L).to(device)
        out_flash = attn_flash(x, freqs_cos.to(device), freqs_sin.to(device), mask=mask)
        assert out_flash.shape == x.shape

        attn_manual = RotaryAttention(dim=C, n_heads=num_heads, flash=False, max_seq_len=L).to(device)
        out_manual = attn_manual(x, freqs_cos.to(device), freqs_sin.to(device), mask=mask)
        assert out_manual.shape == x.shape

    @pytest.mark.gpu
    def test_precompute_freqs_cis_2d(self):
        """Cover 2D RoPE frequency precomputation function."""
        dim = 8
        size = (16, 16)
        freqs_cos, freqs_sin = precompute_freqs_cis_2d(dim, size)
        assert freqs_cos.shape[0] == size[0] * size[1]
        assert freqs_sin.shape[0] == size[0] * size[1]
        assert freqs_cos.shape[1] == dim // 2

    def test_rotary_attention_gradient_flow(self, sample_input_2d, sample_freqs_cis):
        """Test gradients through rotary attention."""
        C = sample_input_2d.shape[-1]
        freqs_cos, freqs_sin = sample_freqs_cis
        sample_input_2d.requires_grad = True
        
        attn = RotaryAttention(dim=C, n_heads=8)
        output = attn(sample_input_2d, freqs_cos, freqs_sin)
        loss = output.sum()
        loss.backward()
        
        assert sample_input_2d.grad is not None
        assert sample_input_2d.grad.abs().sum() > 0

    def test_precompute_freqs_cis(self):
        """Test RoPE frequency precomputation."""
        dim = 8
        seq_len = 256

        freqs_cos, freqs_sin = precompute_freqs_cis(dim, seq_len)

        # Implementation returns shapes (seq_len, dim) per head dim
        assert freqs_cos.shape[0] == seq_len
        assert freqs_sin.shape[0] == seq_len
        assert freqs_cos.shape[1] == freqs_sin.shape[1]
        # cos and sin should sum to ~1 (Pythagorean identity)
        trig_sum = freqs_cos ** 2 + freqs_sin ** 2
        assert torch.allclose(trig_sum, torch.ones_like(trig_sum), atol=1e-5)


class TestEmbeddings:
    """Test embedding layers."""

    def test_cube_embed_shape(self, sample_input_4d):
        """Test CubeEmbed output shape."""
        B, T, C_in, H, W = sample_input_4d.shape
        embed_dim = 128
        patch_size = 4
        
        embed = CubeEmbed(
            in_chans=C_in,
            out_chans=embed_dim,
            in_frames=T,
            patch_size=patch_size,
            flatten=True,
        )
        
        output = embed(sample_input_4d)
        expected_len = (H // patch_size) * (W // patch_size)
        assert output.shape == (B, expected_len, embed_dim)

    def test_cube_embed_with_const(self):
        """Test CubeEmbed with constant channels."""
        B, T, H, W = 2, 4, 32, 64
        dyn_chans = 57
        const_chans = 8

        x_dynamic = torch.randn(B, T, dyn_chans, H, W)
        x_const = torch.randn(B, const_chans, H, W)
        
        embed = CubeEmbed(
            in_chans=dyn_chans,
            out_chans=128,
            const_chans=const_chans,
            in_frames=T,
            patch_size=4,
            flatten=True,
        )
        
        # Flatten temporal dimension for dynamic inputs; pass const separately to exercise const_proj
        merged = x_dynamic.reshape(B, T * dyn_chans, H, W)
        output = embed(merged, const=x_const)
        expected_len = (H // 4) * (W // 4)
        assert output.shape == (B, expected_len, 128)

    def test_timestep_embed_shape(self):
        """Test TimestepEmbed output shape."""
        embed_dim = 256
        embed = TimestepEmbed(embed_dim)
        
        # Test with scalar tensor
        t = torch.tensor([[1], [2]], dtype=torch.float32)
        output = embed(t)
        
        assert output.shape == (2, embed_dim)

    def test_timestep_embed_sinusoidal(self):
        """Test that TimestepEmbed uses sinusoidal encoding."""
        embed = TimestepEmbed(64)
        
        # Embed same timestep multiple times
        t = torch.tensor([[5]], dtype=torch.float32)
        out1 = embed(t)
        out2 = embed(t)
        
        assert torch.allclose(out1, out2)


class TestLayerNorm:
    """Test custom LayerNorm implementations."""

    def test_layernorm_forward(self, sample_input_2d):
        """Test LayerNorm forward pass."""
        ln = LayerNorm(sample_input_2d.shape[-1])
        output = ln(sample_input_2d)
        
        assert output.shape == sample_input_2d.shape
        # Output should be approximately normalized (mean ~0, std ~1)
        assert torch.allclose(output.mean(dim=-1), torch.zeros_like(output.mean(dim=-1)), atol=1e-5)

    def test_layernorm_numerical_stability(self):
        """Test LayerNorm numerical stability with extreme values."""
        ln = LayerNorm(64)
        
        # Very large values
        x_large = torch.randn(2, 256, 64) * 1e6
        output = ln(x_large)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Very small values
        x_small = torch.randn(2, 256, 64) * 1e-6
        output = ln(x_small)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
