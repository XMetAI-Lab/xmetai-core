"""
Unit tests for models in xmetai.models module.
Tests: UTransformer, ViT, base predictor, and model utilities.
"""

import sys
import types

import pytest
import torch
import torch.nn as nn

from xmetai.layers.gated_deltanet2 import GatedDeltaNet2HybridLayer, GatedDeltaNet2Layer

try:
    from xmetai.layers.utransformer import UTransformer
except Exception as e:
    pytest.skip(
        f"UTransformer import failed; skipping model tests: {e}",
        allow_module_level=True,
    )


class TestUTransformer:
    """Test UTransformer weather prediction model."""

    @staticmethod
    def install_fake_fla_gdn2(monkeypatch):
        class FakeGatedDeltaNet2(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.kwargs = kwargs

            def forward(self, hidden_states, **kwargs):
                return hidden_states, None, None

        fla_module = types.ModuleType("fla")
        layers_module = types.ModuleType("fla.layers")
        layers_module.GatedDeltaNet2 = FakeGatedDeltaNet2
        monkeypatch.setitem(sys.modules, "fla", fla_module)
        monkeypatch.setitem(sys.modules, "fla.layers", layers_module)

    @pytest.mark.gpu
    def test_utransformer_initialization(self, torch_seed):
        """Test UTransformer can be initialized with standard config."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[2, 2, 2, 2],
            num_heads=8,
            layer_type="swin",
        )

        assert isinstance(model, nn.Module)
        assert model.embed_dim == 128
        assert model.in_chans == 65

    @pytest.mark.gpu
    def test_utransformer_with_mamba_vision(self, torch_seed):
        """Test UTransformer with MambaVisionLayer."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[2, 2],
            num_heads=8,
            layer_type="mamba",
        )

        assert isinstance(model, nn.Module)
        assert len(model.layers) == 2

    @pytest.mark.gpu
    def test_utransformer_patch_embed(self, torch_seed):
        """Test UTransformer patch embedding."""
        B, T, C, H, W = 2, 2, 65, 32, 64
        x = torch.randn(B, T, C, H, W)

        model = UTransformer(
            in_chans=C,
            out_chans=C,
            in_frames=T,
            embed_dim=128,
            image_size=(H, W),
            window_size=8,
            patch_size=4,
            depths=[1],
            num_heads=8,
            layer_type="swin",
        )

        # Test patch embedding
        x_embed = model.patch_embed(x)

        # Expected: (B, (H*W)/(patch_size^2), embed_dim)
        expected_len = (H // 4) * (W // 4)
        assert x_embed.shape == (B, expected_len, 128)

    @pytest.mark.gpu
    def test_utransformer_with_rotary_attention(self, torch_seed):
        """Test UTransformer with rotary attention config."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[2],
            num_heads=8,
            attn_type="rotary",
            layer_type="mamba",
        )

        # Check RoPE buffers registered
        assert hasattr(model, "freqs_cos")
        assert hasattr(model, "freqs_sin")
        assert model.freqs_cos is not None
        assert model.freqs_sin is not None
        assert model.freqs_cos.shape[0] == (32 // 4) * (64 // 4)
        assert model.freqs_sin.shape[0] == (32 // 4) * (64 // 4)

    @pytest.mark.gpu
    def test_utransformer_dropout(self, torch_seed):
        """Test UTransformer with dropout."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[2],
            num_heads=8,
            drop_mcmc=0.1,
            layer_type="swin",
        )

        assert model.drop_mcmc == 0.1

    @pytest.mark.gpu
    def test_utransformer_head_types(self, torch_seed):
        """Test UTransformer with different head types."""
        in_chans, out_chans = 65, 65

        for head_type in ["1fc", "2fc"]:
            model = UTransformer(
                in_chans=in_chans,
                out_chans=out_chans,
                in_frames=2,
                embed_dim=128,
                image_size=(32, 64),
                window_size=8,
                patch_size=4,
                depths=[1],
                num_heads=8,
                head_type=head_type,
                layer_type="swin",
            )

            assert model.head is not None

    @pytest.mark.gpu
    def test_utransformer_layer_types(self, torch_seed):
        """Test UTransformer with different layer types."""
        for layer_type in ["swin", "mamba"]:
            try:
                model = UTransformer(
                    in_chans=65,
                    out_chans=65,
                    in_frames=2,
                    embed_dim=128,
                    image_size=(32, 64),
                    window_size=8,
                    patch_size=4,
                    depths=[1],
                    num_heads=8,
                    layer_type=layer_type,
                )
                assert isinstance(model, nn.Module)
            except ImportError:
                # Some layers may not be available
                pytest.skip(f"{layer_type} layer not available")

    def test_utransformer_with_gdn2_layer(self, torch_seed, monkeypatch):
        """Test UTransformer selects and runs GatedDeltaNet-2 layers."""
        self.install_fake_fla_gdn2(monkeypatch)
        model = UTransformer(
            in_chans=4,
            out_chans=4,
            in_frames=2,
            embed_dim=32,
            image_size=(8, 8),
            window_size=4,
            patch_size=2,
            depths=[1],
            num_heads=4,
            head_type="1fc",
            layer_type="gdn2_embed",
        )
        x = torch.randn(2, 2, 4, 8, 8)
        timestamp = {"step": torch.zeros(2), "hour": torch.zeros(2), "doy": torch.zeros(2)}

        h = model.patch_embed(x)
        h = model.layers[0](h, time_embedding=model.get_embed(timestamp))
        output = model.resize_output(model.head(h), x, unsqueeze=True)

        assert isinstance(model.layers[0], GatedDeltaNet2Layer)
        assert output.shape == (2, 1, 4, 8, 8)

    def test_utransformer_with_gdn2_swin_layer(self, torch_seed, monkeypatch):
        """Test UTransformer selects hybrid GDN2 + Swin window-attention layers."""
        self.install_fake_fla_gdn2(monkeypatch)
        model = UTransformer(
            in_chans=4,
            out_chans=4,
            in_frames=2,
            embed_dim=32,
            image_size=(8, 8),
            window_size=2,
            patch_size=2,
            depths=[2],
            num_heads=4,
            head_type="1fc",
            layer_type="gdn2_swin_embed",
        )
        x = torch.randn(2, 2, 4, 8, 8)
        timestamp = {"step": torch.zeros(2), "hour": torch.zeros(2), "doy": torch.zeros(2)}

        h = model.patch_embed(x)
        h = model.layers[0](h, time_embedding=model.get_embed(timestamp))
        output = model.resize_output(model.head(h), x, unsqueeze=True)

        assert isinstance(model.layers[0], GatedDeltaNet2HybridLayer)
        assert output.shape == (2, 1, 4, 8, 8)

    def test_utransformer_rejects_gdn2_met_layer_type(self, torch_seed):
        """GDN2 intentionally exposes only the standard unidirectional layer."""
        with pytest.raises(ValueError, match="gdn2_swin"):
            UTransformer(
                in_chans=4,
                out_chans=4,
                in_frames=2,
                embed_dim=32,
                image_size=(8, 8),
                window_size=4,
                patch_size=2,
                depths=[1],
                num_heads=4,
                layer_type="gdn2_met",
            )

    @pytest.mark.gpu
    def test_utransformer_different_input_sizes(self, torch_seed):
        """Test UTransformer with various image sizes."""
        for image_size in [(16, 32), (32, 64), (48, 96)]:
            model = UTransformer(
                in_chans=65,
                out_chans=65,
                in_frames=2,
                embed_dim=128,
                image_size=image_size,
                window_size=8,
                patch_size=4,
                depths=[1],
                num_heads=8,
                layer_type="swin",
            )

            assert model.image_size == tuple(image_size)

    @pytest.mark.gpu
    def test_utransformer_different_patch_sizes(self, torch_seed):
        """Test UTransformer with different patch sizes."""
        for patch_size in [2, 4, 8]:
            model = UTransformer(
                in_chans=65,
                out_chans=65,
                in_frames=2,
                embed_dim=128,
                image_size=(32, 64),
                window_size=8,
                patch_size=patch_size,
                depths=[1],
                num_heads=8,
                layer_type="swin",
            )

            assert model.patch_size == patch_size

    @pytest.mark.gpu
    def test_utransformer_feature_size_calculation(self, torch_seed):
        """Test feature size is correctly calculated."""
        H, W = 32, 64
        patch_size = 4

        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(H, W),
            window_size=8,
            patch_size=patch_size,
            depths=[1],
            num_heads=8,
            layer_type="swin",
        )

        assert model.feat_size == [H // patch_size, W // patch_size]

    def test_utransformer_timestep_embedding(self, torch_seed):
        """Test timestep embedding initialization."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[1],
            num_heads=8,
            norm_type="adaln",
            layer_type="swin",
        )

        # Check timestep embeddings exist
        assert hasattr(model, "step_embed")
        assert hasattr(model, "hour_embed")
        assert hasattr(model, "doy_embed")

    def test_utransformer_freezing_params(self, torch_seed):
        """Test that model parameters can be frozen/unfrozen."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[1],
            num_heads=8,
            layer_type="swin",
        )

        # Freeze patch embedding
        for param in model.patch_embed.parameters():
            param.requires_grad = False

        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_count > 0

    def test_utransformer_parameter_count(self, torch_seed):
        """Test parameter count for small model."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=64,
            image_size=(16, 32),
            window_size=8,
            patch_size=4,
            depths=[1, 1],
            num_heads=4,
            layer_type="swin",
        )

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 100_000_000  # Should be reasonable size

    @pytest.mark.gpu
    def test_utransformer_gpu(self, device, torch_seed):
        """Test UTransformer on GPU."""
        if device.type == "cpu":
            pytest.skip("GPU test")

        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=128,
            image_size=(32, 64),
            window_size=8,
            patch_size=4,
            depths=[1],
            num_heads=8,
            layer_type="swin",
        ).to(device)

        # Check all parameters are on GPU
        for param in model.parameters():
            assert param.device.type == device.type


class TestModelUtilities:
    """Test model-related utilities."""

    def test_parameter_count(self, torch_seed):
        """Test parameter counting."""
        model = UTransformer(
            in_chans=65,
            out_chans=65,
            in_frames=2,
            embed_dim=64,
            image_size=(16, 32),
            window_size=8,
            patch_size=4,
            depths=[1],
            num_heads=4,
            layer_type="swin",
        )

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert trainable == total
        assert total > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
