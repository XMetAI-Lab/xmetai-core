import math

import pytest
import torch

from xmetai.loss.precip_loss import precip_loss, spatial_distribution_loss

CHANNELS = ["z500", "tp", "t2m", "ncpcp"]
B, T, C, H, W = 2, 1, len(CHANNELS), 8, 10


TP_INDEX = CHANNELS.index("tp")


def test_empty_precip_indices_return_empty_loss():
    output = torch.zeros(1, 1, 2, 4, 4)
    target = torch.zeros_like(output)

    losses = precip_loss(output, target, [], threshold=0.0, coefficients={"occurrence": 1.0})

    assert losses == {}


def test_occurrence_and_distribution_are_scalar_and_differentiable():
    torch.manual_seed(0)
    output = torch.randn(B, T, C, H, W, requires_grad=True)
    target = torch.randn(B, T, C, H, W)

    losses = precip_loss(
        output,
        target,
        [TP_INDEX],
        threshold=0.0,
        coefficients={"occurrence": 1.0, "distribution": 1.0},
    )

    assert set(losses) == {"loss_occurrence", "loss_distribution"}
    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    total = sum(losses.values())
    total.backward()
    assert output.grad is not None
    assert output.grad.abs().sum() > 0


def test_distribution_loss_penalizes_errors_at_rain_locations():
    target = torch.zeros(1, 1, C, 6, 6)
    target[:, :, TP_INDEX, 1, 1] = 5.0
    target[:, :, TP_INDEX, 4, 3] = 3.0

    same_output = target.clone()
    shifted_output = torch.zeros_like(target)
    shifted_output[:, :, TP_INDEX, 1, 2] = 5.0
    shifted_output[:, :, TP_INDEX, 4, 4] = 3.0

    same_loss = precip_loss(
        same_output,
        target,
        [TP_INDEX],
        threshold=0.1,
        coefficients={"distribution": 1.0},
        occurrence_scale=100.0,
    )["loss_distribution"]
    shifted_loss = precip_loss(
        shifted_output,
        target,
        [TP_INDEX],
        threshold=0.1,
        coefficients={"distribution": 1.0},
        occurrence_scale=100.0,
    )["loss_distribution"]

    assert same_loss.item() < 1e-4
    assert shifted_loss.item() > same_loss.item() + 0.1


def test_distribution_loss_weights_heavier_target_rain_more():
    target = torch.zeros(1, 1, C, 2, 2)
    target[:, :, TP_INDEX, 0, 0] = 1.0
    target[:, :, TP_INDEX, 0, 1] = 5.0

    light_error = target.clone()
    light_error[:, :, TP_INDEX, 0, 0] += 1.0
    heavy_error = target.clone()
    heavy_error[:, :, TP_INDEX, 0, 1] += 1.0

    light_loss = precip_loss(light_error, target, [TP_INDEX], threshold=0.1, coefficients={"distribution": 1.0})["loss_distribution"]
    heavy_loss = precip_loss(heavy_error, target, [TP_INDEX], threshold=0.1, coefficients={"distribution": 1.0})["loss_distribution"]

    assert heavy_loss.item() > light_loss.item()


def test_distribution_loss_zero_for_all_dry_target():
    target = torch.zeros(1, 1, C, 4, 4)
    output = torch.zeros_like(target)
    output[:, :, TP_INDEX] = 1.0

    losses = precip_loss(
        output,
        target,
        [TP_INDEX],
        threshold=0.0,
        coefficients={"occurrence": 1.0, "distribution": 1.0},
    )

    assert losses["loss_distribution"].item() == pytest.approx(0.0, abs=1e-8)
    assert losses["loss_occurrence"].item() > 0.0


def test_coefficients_scale_occurrence_loss():
    target = torch.ones(1, 1, C, H, W)
    output = torch.zeros_like(target)

    loss_small = precip_loss(
        output,
        target,
        [TP_INDEX],
        threshold=0.5,
        coefficients={"occurrence": 0.5},
    )["loss_occurrence"]
    loss_large = precip_loss(
        output,
        target,
        [TP_INDEX],
        threshold=0.5,
        coefficients={"occurrence": 1.0},
    )["loss_occurrence"]

    assert loss_large.item() == pytest.approx(loss_small.item() * 2)


def test_distribution_uses_training_weights_across_separate_scenes():
    target = torch.tensor([1.1, 10.1], dtype=torch.float64).reshape(2, 1, 1, 1, 1)
    output = (target + 1.0).requires_grad_()
    loss = spatial_distribution_loss(output, target, threshold=0.1)
    point_loss = math.sqrt(1.0 + 1e-6) - math.sqrt(1e-6)
    assert loss.item() == pytest.approx(point_loss * (1.0 + 10.0) / 2)
    loss.backward()
    assert output.grad[1].item() / output.grad[0].item() == pytest.approx(10.0)


def test_distribution_extreme_rain_has_bounded_gradient():
    target = torch.tensor([100.1, 1000.1], dtype=torch.float64).reshape(2, 1, 1, 1, 1)
    output = (target + 1.0).requires_grad_()
    loss = spatial_distribution_loss(output, target, threshold=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert output.grad[0].item() == pytest.approx(output.grad[1].item())
    assert output.grad.abs().max().item() <= 10.0 / target.numel()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_distribution_dry_pixels_have_no_auxiliary_gradient(reduction):
    target = torch.tensor([-1.0, -0.5, 0.5], dtype=torch.float64).reshape(1, 1, 1, 1, 3)
    output = (target + 1.0).requires_grad_()
    loss = spatial_distribution_loss(output, target, threshold=-0.5, reduction=reduction)
    point_loss = math.sqrt(1.0 + 1e-6) - math.sqrt(1e-6)
    assert loss.item() == pytest.approx(point_loss / (3 if reduction == "mean" else 1))
    loss.backward()
    assert output.grad.flatten()[:2].abs().max().item() == pytest.approx(0.0, abs=1e-12)
    assert output.grad.flatten()[2].item() > 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_distribution_extreme_training_target_has_finite_gradient(dtype):
    target = torch.full((1, 1, 1, 1, 1), 100.0, dtype=dtype, requires_grad=True)
    output = torch.full_like(target, 101.0, requires_grad=True)
    loss = spatial_distribution_loss(output, target, threshold=0.1)
    expected = (math.sqrt(1.0 + 1e-6) - math.sqrt(1e-6)) * 10.0
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert target.grad is None
