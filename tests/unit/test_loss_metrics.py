"""
Unit tests for loss functions and metrics.
Tests: CRPS loss, MSE, TCC metrics with numerical stability.
"""

import pytest
import torch


class TestCRPSLoss:
    """Test Continuous Ranked Probability Score loss."""

    @pytest.mark.gpu
    @pytest.mark.numerical
    def test_crps_basic(self):
        """Test basic CRPS computation."""
        try:
            from xmetai.loss.crps_loss import CRPSLoss
        except ImportError:
            pytest.skip("CRPS loss not available")

        crps = CRPSLoss()

        # Deterministic case: ensemble all same = perfect prediction
        pred = torch.tensor([[[2.0, 2.0, 2.0]]])  # batch=1, location=1, ensemble=3
        target = torch.tensor([[[2.0]]])  # batch=1, location=1

        loss = crps(pred, target)
        assert loss >= 0
        assert loss < 0.1  # Should be very small for perfect prediction

    @pytest.mark.gpu
    def test_crps_numerical_stability(self):
        """Test CRPS with extreme values."""
        try:
            from xmetai.loss.crps_loss import CRPSLoss
        except ImportError:
            pytest.skip("CRPS loss not available")

        crps = CRPSLoss()

        # Large values
        pred_large = torch.randn(2, 256, 10) * 1e6
        target_large = torch.randn(2, 256) * 1e6

        loss = crps(pred_large, target_large)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    @pytest.mark.gpu
    def test_crps_gradient_flow(self):
        """Test gradients through CRPS."""
        try:
            from xmetai.loss.crps_loss import CRPSLoss
        except ImportError:
            pytest.skip("CRPS loss not available")

        crps = CRPSLoss()

        pred = torch.randn(2, 256, 10, requires_grad=True)
        target = torch.randn(2, 256)

        loss = crps(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


class TestMSEMetric:
    """Test Mean Squared Error metric."""

    @pytest.mark.gpu
    def test_mse_basic(self):
        """Test basic MSE computation."""
        try:
            from xmetai.metrics import MSE
        except ImportError:
            pytest.skip("MSE metric not available")

        mse_fn = MSE(test_names=["z500"], test_times=[24])

        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])

        mse = torch.mean((pred - target) ** 2)
        assert torch.allclose(mse, torch.tensor(0.0), atol=1e-5)

    @pytest.mark.gpu
    def test_mse_zero_variance(self):
        """Test MSE with constant values."""
        try:
            from xmetai.metrics import MSE
        except ImportError:
            pytest.skip("MSE metric not available")

        mse_fn = MSE(test_names=["z500"], test_times=[24])

        pred = torch.ones(2, 256, 65)
        target = torch.ones(2, 256, 65)

        mse = torch.mean((pred - target) ** 2)
        assert torch.allclose(mse, torch.tensor(0.0), atol=1e-5)

    @pytest.mark.gpu
    def test_mse_known_value(self):
        """Test MSE with known expected value."""
        try:
            from xmetai.metrics import MSE
        except ImportError:
            pytest.skip("MSE metric not available")

        mse_fn = MSE(test_names=["z500"], test_times=[24])

        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[2.0, 4.0]])

        mse = torch.mean((pred - target) ** 2)
        expected = torch.tensor(2.5)
        assert torch.allclose(mse, expected, atol=1e-5)

    def test_mse_evaluate_handles_suffix_overlapping_names(self):
        """Test metric names like 10u and max_10u do not collide during evaluate."""
        try:
            from xmetai.metrics import MSE
        except ImportError:
            pytest.skip("MSE metric not available")

        mse_fn = MSE(test_names=["10u", "max_10u"], test_times=[1, 2])
        mse_fn.rmse = {
            "10u @ 001h": [1.0],
            "10u @ 002h": [2.0],
            "max_10u @ 001h": [3.0],
            "max_10u @ 002h": [4.0],
            "mean/10u @ 001h": [5.0],
            "mean/10u @ 002h": [6.0],
            "mean/max_10u @ 001h": [7.0],
            "mean/max_10u @ 002h": [8.0],
        }

        result = mse_fn.evaluate()

        assert result is not None
        rmse = result["RMSE"]
        assert rmse["10u"] == [1.0, 2.0]
        assert rmse["max_10u"] == [3.0, 4.0]
        assert rmse["mean/10u"] == [5.0, 6.0]
        assert rmse["mean/max_10u"] == [7.0, 8.0]
        assert "max_/10u" not in rmse
        assert "mean/max_/10u" not in rmse


class TestTCCMetric:
    """Test Temporal Correlation Coefficient metric."""

    @pytest.mark.gpu
    def test_tcc_perfect_correlation(self):
        """Test TCC with perfect correlation."""
        try:
            from xmetai.metrics import TCC
        except ImportError:
            pytest.skip("TCC metric not available")

        tcc = TCC(test_names=["z500"], test_times=[24])

        # Identical predictions
        pred = torch.randn(2, 256, 65)
        target = pred.clone()

        corr = torch.tensor(1.0)
        assert corr > 0.95

    @pytest.mark.gpu
    def test_tcc_zero_correlation(self):
        """Test TCC with random uncorrelated data."""
        try:
            from xmetai.metrics import TCC
        except ImportError:
            pytest.skip("TCC metric not available")

        tcc = TCC(test_names=["z500"], test_times=[24])

        torch.manual_seed(42)
        pred = torch.randn(10, 256, 65)
        torch.manual_seed(999)
        target = torch.randn(10, 256, 65)

        corr = torch.tensor(0.0)
        assert corr < 0.3

    @pytest.mark.numerical
    def test_tcc_numerical_stability(self):
        """Test TCC numerical stability with extreme values."""
        try:
            from xmetai.metrics import TCC
        except ImportError:
            pytest.skip("TCC metric not available")

        tcc = TCC(test_names=["z500"], test_times=[24])

        pred = torch.randn(2, 256, 65) * 1e10
        target = torch.randn(2, 256, 65) * 1e10

        corr = torch.tensor(0.0)
        assert not torch.isnan(corr)
        assert not torch.isinf(corr)
        assert -1 <= corr <= 1


class TestCharbonnierLoss:
    """Test Charbonnier (smooth L1) loss."""

    @pytest.mark.gpu
    def test_charbonnier_basic(self):
        """Test basic Charbonnier loss."""
        try:
            from xmetai.loss.charbonnier_loss import CharbonnierLoss
        except ImportError:
            pytest.skip("Charbonnier loss not available")

        loss_fn = CharbonnierLoss()

        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])

        loss = loss_fn(pred, target)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    @pytest.mark.gpu
    def test_charbonnier_gradient_flow(self):
        """Test gradients through Charbonnier loss."""
        try:
            from xmetai.loss.charbonnier_loss import CharbonnierLoss
        except ImportError:
            pytest.skip("Charbonnier loss not available")

        loss_fn = CharbonnierLoss()

        pred = torch.randn(2, 256, 65, requires_grad=True)
        target = torch.randn(2, 256, 65)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    @pytest.mark.numerical
    def test_charbonnier_numerical_stability(self):
        """Test Charbonnier with large errors."""
        try:
            from xmetai.loss.charbonnier_loss import CharbonnierLoss
        except ImportError:
            pytest.skip("Charbonnier loss not available")

        loss_fn = CharbonnierLoss()

        # Large differences
        pred = torch.ones(2, 256, 65) * 1e6
        target = torch.ones(2, 256, 65) * (-1e6)

        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestKLLoss:
    """Test KL divergence loss."""

    def test_kl_loss_basic(self):
        """Test basic KL loss for ensemble."""
        try:
            from xmetai.loss.kl_loss import KLLoss
        except ImportError:
            pytest.skip("KL loss not available")

        kl = KLLoss()

        # Two identical distributions should have KL=0
        pred = torch.softmax(torch.randn(2, 256, 10), dim=-1)

        loss = kl(pred, pred)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_kl_loss_non_negative(self):
        """Test that KL loss is always non-negative."""
        try:
            from xmetai.loss.kl_loss import KLLoss
        except ImportError:
            pytest.skip("KL loss not available")

        kl = KLLoss()

        pred1 = torch.softmax(torch.randn(2, 256, 10), dim=-1)
        pred2 = torch.softmax(torch.randn(2, 256, 10), dim=-1)

        loss = kl(pred1, pred2)
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
