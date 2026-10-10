import math

import pytest
import torch

from xmetai.loss import SphericalAMSELoss


def fields():
    theta = torch.linspace(0, math.pi, 17)[:, None]
    lon = torch.arange(32)[None, :] * (2 * math.pi / 32)
    return theta.sin() * lon.cos(), theta.sin() * lon.sin()


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("correlation", [-0.5, 0.0, 0.5, 1.0])
def test_amse_matches_equation_and_restores_amplitude(amplitude, correlation):
    loss = SphericalAMSELoss((17, 32))
    target, orthogonal = fields()
    scale = torch.tensor(amplitude, requires_grad=True)
    prediction = scale * (correlation * target + math.sqrt(1 - correlation**2) * orthogonal)
    value = loss(prediction, target)
    expected = ((amplitude - 1) ** 2 + 2 * max(amplitude**2, 1) * (1 - correlation)) / 3
    torch.testing.assert_close(value, torch.tensor(expected), atol=2e-6, rtol=2e-5)
    if amplitude != 1:
        value.backward()
        assert scale.grad * (amplitude - 1) > 0


def test_amse_penalizes_phase_and_mean_errors():
    loss = SphericalAMSELoss((17, 32))
    target, orthogonal = fields()
    torch.testing.assert_close(loss(orthogonal, target), torch.tensor(2 / 3))
    torch.testing.assert_close(loss(target + 3, target + 1), torch.tensor(4.0))
    torch.testing.assert_close(loss(target, target), torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("zero_prediction,zero_target", [(True, True), (True, False), (False, True), (False, False)])
def test_amse_zero_energy_autocast_and_target_detachment(zero_prediction, zero_target):
    target, orthogonal = fields()
    target = (target * (not zero_target)).requires_grad_()
    output = (orthogonal * (not zero_prediction)).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = SphericalAMSELoss((17, 32))(output, target)
    value.backward()
    assert value.dtype == torch.float32 and torch.isfinite(value)
    assert torch.isfinite(output.grad).all()
    assert target.grad is None
    if zero_prediction and not zero_target:
        assert output.grad.abs().sum() > 0


def test_amse_symmetry_and_squared_units():
    loss = SphericalAMSELoss((17, 32))
    target, output = fields()
    torch.testing.assert_close(loss(output, target), loss(target, output))
    torch.testing.assert_close(loss(3 * output, 3 * target), 9 * loss(output, target))
    assert not loss.state_dict()
