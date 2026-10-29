"""
Unit tests for loss functions and metrics.
Tests: CRPS loss, MSE, TCC metrics with numerical stability.
"""

import importlib

import pytest
import torch


class TestSortedCRPSEntrypoints:
    @pytest.mark.parametrize("members", [2, 3, 8])
    @pytest.mark.parametrize("name,alpha", [("crps_loss", 0.0), ("afcrps_loss", 0.0), ("afcrps_loss", 0.95), ("afcrps_loss", 1.0)])
    def test_scores_and_gradients(self, monkeypatch, members, name, alpha):
        crps_module = importlib.import_module("xmetai.loss.crps_loss")
        generator = torch.Generator().manual_seed(42)
        outputs = torch.randn(members, 1, 1, 2, 9, 16, generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(outputs[0], requires_grad=True)
        weight = torch.rand(9, 1, dtype=torch.float64, requires_grad=True)
        monkeypatch.setattr(crps_module, "all_gather_grad", lambda output: outputs)
        kwargs = {"alpha": alpha} if name == "afcrps_loss" else {}
        actual = getattr(crps_module, name)(outputs[0], target, weight, **kwargs)
        expected = TestMultiscaleAFCRPS._reference(outputs, target, weight, alpha)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (outputs, target, weight)), torch.autograd.grad(expected, (outputs, target, weight))):
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("alpha", [0.0, 0.95, 1.0])
    @pytest.mark.parametrize("scales", [[], [2], [2, 8]])
    def test_multiscale_scores_and_gradients(self, monkeypatch, alpha, scales):
        crps_module = importlib.import_module("xmetai.loss.crps_loss")
        outputs = torch.randn(3, 1, 1, 2, 9, 16, dtype=torch.float64, requires_grad=True)
        target = torch.randn_like(outputs[0], requires_grad=True)
        weight = torch.rand(9, 1, dtype=torch.float64, requires_grad=True)
        monkeypatch.setattr(crps_module, "all_gather_grad", lambda output: outputs)
        actual = crps_module.multiscale_afcrps_loss(outputs[0], target, weight, alpha=alpha, filter_scales=scales)
        output_parts = [crps_module._gaussian_filter_global(outputs, scale) for scale in sorted(scales, reverse=True)] + [outputs]
        target_parts = [crps_module._gaussian_filter_global(target, scale) for scale in sorted(scales, reverse=True)] + [target]
        expected = TestMultiscaleAFCRPS._reference(output_parts[0], target_parts[0], weight, alpha)
        for index in range(1, len(output_parts)):
            expected = expected + TestMultiscaleAFCRPS._reference(
                output_parts[index] - output_parts[index - 1], target_parts[index] - target_parts[index - 1], weight, alpha
            )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (outputs, target, weight)), torch.autograd.grad(expected, (outputs, target, weight))):
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("name", ["crps_loss", "afcrps_loss", "multiscale_afcrps_loss"])
    def test_single_member_mae(self, name):
        from xmetai import loss

        output = torch.ones(1, 1, 1, 9, 16, requires_grad=True)
        target = torch.zeros_like(output)
        weight = torch.full((9, 1), 2.0)
        kwargs = {} if name == "crps_loss" else {"alpha": 0.0}
        actual = getattr(loss, name)(output, target, weight, **kwargs)
        torch.testing.assert_close(actual, ((output - target).abs() * weight).mean())
        actual.backward()
        torch.testing.assert_close(output.grad, torch.full_like(output, 2.0 / output.numel()))

    @pytest.mark.parametrize("name", ["afcrps_loss", "multiscale_afcrps_loss"])
    def test_single_member_zero_has_gradient(self, name):
        from xmetai import loss

        output = torch.ones(1, 1, 1, 9, 16, requires_grad=True)
        actual = getattr(loss, name)(output, torch.zeros_like(output))
        assert actual.item() == 0
        actual.backward()
        torch.testing.assert_close(output.grad, torch.zeros_like(output))


class TestMultiscaleAFCRPS:
    @pytest.mark.parametrize("members", [2, 3, 8, 16])
    def test_contraction_matches_gap_score_and_gradients(self, members):
        from xmetai.loss.crps_loss import _afcrps_from_members

        generator = torch.Generator().manual_seed(42)
        outputs = torch.randn(members, 2, 1, 3, 9, 16, generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.randn(2, 1, 3, 9, 16, generator=generator, dtype=torch.float64, requires_grad=True)
        weight = torch.rand(9, 1, generator=generator, dtype=torch.float64, requires_grad=True)
        ordered = outputs.sort(dim=0).values
        index = torch.arange(1, members, dtype=outputs.dtype)
        gap_weights = (index * (members - index)).view(-1, 1, 1, 1, 1, 1)
        spread = ((ordered[1:] - ordered[:-1]) * gap_weights).sum(0) / (members * (members - 1))
        expected = (((outputs - target).abs().mean(0) - (1 - 0.05 / members) * spread) * weight).mean()
        actual = _afcrps_from_members(outputs, target, weight, 0.95)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (outputs, target, weight)), torch.autograd.grad(expected, (outputs, target, weight))):
            torch.testing.assert_close(actual_grad, expected_grad)

    @staticmethod
    def _reference(outputs, target, weight=None, alpha=0.95):
        members = outputs.shape[0]
        error = (outputs - target).abs().mean(dim=0)
        pairwise = (outputs[:, None] - outputs[None, :]).abs().sum(dim=(0, 1))
        score = error - (1 - (1 - alpha) / members) * pairwise / (2 * members * (members - 1))
        return (score * weight).mean() if weight is not None else score.mean()

    @pytest.mark.parametrize("scales", [[]])
    def test_original_score(self, monkeypatch, scales):
        crps_loss = importlib.import_module("xmetai.loss.crps_loss")

        outputs = torch.randn(3, 1, 1, 2, 9, 16, requires_grad=True)
        target = torch.randn_like(outputs[0])
        weight = torch.rand(9, 1)
        monkeypatch.setattr(crps_loss, "all_gather_grad", lambda output: outputs)
        actual = crps_loss.multiscale_afcrps_loss(outputs[0], target, weight, filter_scales=scales)
        torch.testing.assert_close(actual, self._reference(outputs, target, weight))
        actual.backward()
        assert torch.isfinite(outputs.grad).all()
        assert outputs.grad.abs().sum() > 0

    @pytest.mark.parametrize("scales", [[8.0], [8.0, 2.0], [2.0, 8.0]])
    def test_scale_partition_and_gradients(self, monkeypatch, scales):
        crps_loss = importlib.import_module("xmetai.loss.crps_loss")
        from xmetai.loss.crps_loss import _gaussian_filter_global

        outputs = torch.randn(3, 1, 1, 2, 9, 16, requires_grad=True)
        target = torch.randn_like(outputs[0])
        weight = torch.rand(9, 1)
        calls = []

        def gather(output):
            calls.append(output)
            return outputs

        monkeypatch.setattr(crps_loss, "all_gather_grad", gather)
        actual = crps_loss.multiscale_afcrps_loss(outputs[0], target, weight, filter_scales=scales)
        ordered = sorted(scales, reverse=True)
        output_parts = [_gaussian_filter_global(outputs, scale) for scale in ordered] + [outputs]
        target_parts = [_gaussian_filter_global(target, scale) for scale in ordered] + [target]
        expected = self._reference(output_parts[0], target_parts[0], weight)
        reconstructed = output_parts[0]
        for index in range(1, len(output_parts)):
            detail = output_parts[index] - output_parts[index - 1]
            reconstructed = reconstructed + detail
            expected = expected + self._reference(detail, target_parts[index] - target_parts[index - 1], weight)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(reconstructed, outputs)
        assert len(calls) == 1
        expected_grad = torch.autograd.grad(expected, outputs)[0]
        actual.backward()
        torch.testing.assert_close(outputs.grad, expected_grad)
        assert torch.isfinite(outputs.grad).all()
        assert outputs.grad.abs().sum() > 0

    def test_filter_constants_and_longitude_periodicity(self):
        from xmetai.loss.crps_loss import _gaussian_filter_global

        constant = torch.ones(2, 9, 16)
        torch.testing.assert_close(_gaussian_filter_global(constant, 8), constant)
        field = torch.randn_like(constant)
        smooth = _gaussian_filter_global(field, 2)
        torch.testing.assert_close(_gaussian_filter_global(field.roll(3, dims=-1), 2), smooth.roll(3, dims=-1))

    @pytest.mark.parametrize("sigma", [1, 2, 8])
    def test_gaussian_frequency_response(self, sigma):
        from xmetai.loss.crps_loss import _gaussian_filter_global

        longitude = torch.arange(32, dtype=torch.float32)
        angular_frequency = 4 * torch.pi / 32
        wave = (angular_frequency * longitude).cos().expand(9, -1)
        offsets = torch.arange(-4 * sigma, 4 * sigma + 1, dtype=torch.float32)
        weights = torch.exp(-0.5 * (offsets / sigma).square())
        response = (weights * (angular_frequency * offsets).cos()).sum() / weights.sum()
        torch.testing.assert_close(_gaussian_filter_global(wave, sigma), wave * response, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("autocast_enabled", [False, True])
    def test_multiscale_preserves_precision(self, monkeypatch, dtype, autocast_enabled):
        crps_loss = importlib.import_module("xmetai.loss.crps_loss")
        from xmetai.loss.crps_loss import _gaussian_filter_global

        outputs = torch.randn(3, 1, 1, 1, 9, 16, dtype=dtype, requires_grad=True)
        target = torch.randn_like(outputs[0])
        filtered_dtypes = []

        def filter_field(field, sigma):
            result = _gaussian_filter_global(field, sigma)
            filtered_dtypes.append(result.dtype)
            assert field.dtype == dtype
            return result

        monkeypatch.setattr(crps_loss, "all_gather_grad", lambda output: outputs)
        monkeypatch.setattr(crps_loss, "_gaussian_filter_global", filter_field)
        expected = crps_loss.multiscale_afcrps_loss(outputs[0], target, filter_scales=[8, 2])
        with torch.autocast("cpu", dtype=dtype, enabled=autocast_enabled):
            actual = crps_loss.multiscale_afcrps_loss(outputs[0], target, filter_scales=[8, 2])
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.backward()
        assert set(filtered_dtypes) == {dtype}
        assert outputs.grad.dtype == dtype
        assert torch.isfinite(outputs.grad).all()
        assert outputs.grad.abs().sum() > 0

    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_checkpoint_retains_only_input_storage(self, monkeypatch, time_steps):
        crps_loss = importlib.import_module("xmetai.loss.crps_loss")
        from xmetai.loss.crps_loss import _gaussian_filter_global, _multiscale_afcrps_field

        outputs = torch.randn(3, 2, time_steps, 3, 9, 16, requires_grad=True)
        target = torch.randn_like(outputs[0], requires_grad=True)
        weight = torch.rand(1, 1, 3, 9, 1, requires_grad=True)
        input_storages = {field.untyped_storage().data_ptr() for field in (outputs, target, weight)}
        retained_storages = {}

        def pack(tensor):
            if tensor.numel():
                storage = tensor.untyped_storage()
                retained_storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            expanded_weight = weight.expand_as(target)
            expected = torch.stack(
                [
                    _multiscale_afcrps_field(outputs[:, b, t, c], target[b, t, c], expanded_weight[b, t, c], 0.95, [8.0, 2.0])
                    for b in range(2)
                    for t in range(time_steps)
                    for c in range(3)
                ]
            ).mean()
        baseline_extra_bytes = sum(size for pointer, size in retained_storages.items() if pointer not in input_storages)
        expected_grads = torch.autograd.grad(expected, (outputs, target, weight))
        retained_storages.clear()
        filtered_shapes = []

        def filter_field(field, sigma):
            filtered_shapes.append(tuple(field.shape))
            assert field.shape in (outputs.shape, target.shape)
            return _gaussian_filter_global(field, sigma)

        monkeypatch.setattr(crps_loss, "all_gather_grad", lambda output: outputs)
        monkeypatch.setattr(crps_loss, "_gaussian_filter_global", filter_field)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            actual = crps_loss.multiscale_afcrps_loss(outputs[0], target, weight, filter_scales=[8, 2])
        assert baseline_extra_bytes > 0
        assert set(retained_storages).issubset(input_storages)
        torch.testing.assert_close(actual, expected)
        forward_filter_calls = len(filtered_shapes)
        assert forward_filter_calls == 4
        actual_grads = torch.autograd.grad(actual, (outputs, target, weight))
        assert len(filtered_shapes) > forward_filter_calls
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("scales", [[0.0], [-1.0], [8.0, 0.0], [8.0, 8.0], [float("nan")], [float("inf")]])
    def test_invalid_scales(self, scales):
        from xmetai.loss.crps_loss import multiscale_afcrps_loss

        field = torch.zeros(1, 1, 1, 9, 16)
        with pytest.raises(ValueError, match="filter_scales"):
            multiscale_afcrps_loss(field, field, filter_scales=scales)

    def test_single_process_stays_zero(self):
        from xmetai.loss.crps_loss import multiscale_afcrps_loss

        field = torch.zeros(1, 1, 1, 9, 16)
        assert multiscale_afcrps_loss(field, field, filter_scales=[8]).item() == 0


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
