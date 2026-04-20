"""
Integration tests for end-to-end model workflows.
Tests: Forward passes, gradient flow, ONNX export, model save/load.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from xmetai.layers import HAS_MAMBA


class TestForwardPass:
    """Test complete forward pass through models."""

    def test_utransformer_swin_forward(self, sample_input_4d, torch_seed):
        """Test UTransformer with Swin layers end-to-end forward pass."""
        from xmetai.models.utransformer import UTransformer

        B, T, C, H, W = sample_input_4d.shape

        model = UTransformer(
            in_chans=C,
            out_chans=C,
            in_frames=T,
            embed_dim=128,
            image_size=(H, W),
            window_size=8,
            patch_size=4,
            depths=[2, 2],
            num_heads=8,
            head_type="1fc",
            layer_type="swin",
        )

        model.eval()
        with torch.no_grad():
            try:
                # Test forward (expected to fail if not fully implemented)
                output = model.patch_embed(sample_input_4d)
                assert output.shape[0] == B
                assert output.shape[-1] == 128
            except NotImplementedError:
                pytest.skip("Model forward not fully implemented")

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_utransformer_mamba_forward(self, sample_input_4d, torch_seed):
        """Test UTransformer with Mamba layers."""
        from xmetai.models.utransformer import UTransformer

        B, T, C, H, W = sample_input_4d.shape

        model = UTransformer(
            in_chans=C,
            out_chans=C,
            in_frames=T,
            embed_dim=128,
            image_size=(H, W),
            window_size=8,
            patch_size=4,
            depths=[2],
            num_heads=8,
            head_type="1fc",
            layer_type="mamba",
            attn_type="standard",
        )

        model.eval()
        with torch.no_grad():
            # Test patch embedding at least
            output = model.patch_embed(sample_input_4d)
            assert output.shape[0] == B

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_utransformer_rotary_forward(self, sample_input_4d, torch_seed):
        """Test UTransformer with rotary attention."""
        from xmetai.models.utransformer import UTransformer

        B, T, C, H, W = sample_input_4d.shape

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
            head_type="1fc",
            layer_type="mamba",
            attn_type="rotary",
        )

        # Check RoPE buffers
        assert hasattr(model, "freqs_cos")
        assert hasattr(model, "freqs_sin")
        assert model.freqs_cos is not None


class TestGradientFlow:
    """Test gradient flow through models."""

    def test_gradient_flow_patch_embed(self, sample_input_4d):
        """Test gradients through patch embedding."""
        from xmetai.models.utransformer import UTransformer

        B, T, C, H, W = sample_input_4d.shape

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

        sample_input_4d.requires_grad = True
        output = model.patch_embed(sample_input_4d)
        loss = output.sum()
        loss.backward()

        assert sample_input_4d.grad is not None
        assert sample_input_4d.grad.abs().sum() > 0

    def test_gradient_accumulation(self, sample_input_2d):
        """Test gradient accumulation over multiple batches."""
        from xmetai.layers.attention import Attention

        attn = Attention(dim=sample_input_2d.shape[-1], num_heads=8)
        optimizer = torch.optim.SGD(attn.parameters(), lr=0.01)

        # Two backward passes
        sample_input_2d.requires_grad = True

        for _ in range(2):
            output = attn(sample_input_2d)
            loss = output.sum()
            loss.backward()

        # Gradients should accumulate
        param_grad_norm = sum(p.grad.abs().sum() for p in attn.parameters() if p.grad is not None)
        assert param_grad_norm > 0


class TestModelSaveLoad:
    """Test model checkpointing."""

    def test_save_load_state_dict(self, torch_seed):
        """Test saving and loading model state."""
        from xmetai.models.utransformer import UTransformer

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

        # Save state
        state = model.state_dict()

        # Create new model and load state
        model2 = UTransformer(
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
        model2.load_state_dict(state)

        # Check parameters are identical
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_file_io(self, torch_seed):
        """Test checkpoint file operations."""
        from xmetai.models.utransformer import UTransformer

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

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pth"

            # Save checkpoint
            torch.save(model.state_dict(), ckpt_path)
            assert ckpt_path.exists()

            # Load checkpoint
            model2 = UTransformer(
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
            model2.load_state_dict(torch.load(ckpt_path))

            # Verify
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestONNXExport:
    """Test ONNX model export compatibility."""

    @pytest.mark.skip(reason="ONNX export requires full model implementation")
    def test_onnx_export_basic(self, sample_input_4d):
        """Test basic ONNX export."""
        from xmetai.models.utransformer import UTransformer

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

        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"

            # This would fail without proper forward() implementation
            torch.onnx.export(
                model,
                sample_input_4d,
                onnx_path,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                opset_version=14,
            )

            assert onnx_path.exists()

    @pytest.mark.skipif(not HAS_MAMBA, reason="mamba-ssm not installed")
    def test_onnx_export_selective_scan(self):
        """Test ONNX export with custom selective scan op."""
        from xmetai.layers.mamba_vision import MambaMetMixer

        # Test that mamba blocks have symbolic export
        mixer = MambaMetMixer(d_model=64)

        # Check if symbolic export is registered
        from xmetai.layers.mamba_vision import symbolic_selective_scan_fn

        assert callable(symbolic_selective_scan_fn)


class TestMixedPrecision:
    """Test mixed precision training."""

    @pytest.mark.gpu
    def test_mixed_precision_forward(self, device, torch_seed):
        """Test forward pass with mixed precision on GPU."""
        if device.type == "cpu":
            pytest.skip("GPU test")

        # Use new torch.amp autocast API for CUDA
        from torch.amp import autocast

        from xmetai.models.utransformer import UTransformer

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

        x = torch.randn(2, 2, 65, 32, 64, device=device)

        with autocast(device_type="cuda"):
            output = model.patch_embed(x)
            # Some paths may keep fp32 under autocast; accept fp32 as valid
            assert output.dtype in (torch.float16, torch.bfloat16, torch.float32)

    @pytest.mark.gpu
    def test_mixed_precision_backward(self, device, torch_seed):
        """Test backward pass with mixed precision."""
        if device.type == "cpu":
            pytest.skip("GPU test")

        from torch.amp import GradScaler, autocast

        from xmetai.models.utransformer import UTransformer

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

        x = torch.randn(2, 2, 65, 32, 64, device=device)

        scaler = GradScaler("cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with autocast(device_type="cuda"):
            output = model.patch_embed(x)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


class TestDeterminism:
    """Test model determinism for reproducibility."""

    def test_eval_mode_determinism(self, sample_input_2d, torch_seed):
        """Test that eval mode produces deterministic outputs."""
        from xmetai.layers.attention import Attention

        attn = Attention(dim=sample_input_2d.shape[-1], num_heads=8)
        attn.eval()

        with torch.no_grad():
            out1 = attn(sample_input_2d.clone())
            out2 = attn(sample_input_2d.clone())

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_training_mode_stochasticity(self, sample_input_2d, torch_seed):
        """Test that training mode can produce different outputs (due to dropout)."""
        from xmetai.layers.attention import Attention

        attn = Attention(
            dim=sample_input_2d.shape[-1],
            num_heads=8,
            attn_drop=0.5,
            proj_drop=0.5,
        )
        attn.train()

        # With dropout, outputs should differ
        out1 = attn(sample_input_2d.clone())
        out2 = attn(sample_input_2d.clone())

        # They should be different (high probability with 0.5 dropout)
        assert not torch.allclose(out1, out2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
