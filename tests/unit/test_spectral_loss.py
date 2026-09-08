import math

import pytest
import torch

from xmetai.loss import SphericalBandPowerLoss


def make_loss(image_size, bands):
    return SphericalBandPowerLoss(image_size, bands=bands)


def harmonic_field(nlat=33) -> torch.Tensor:
    theta = torch.linspace(0, math.pi, nlat)[:, None]
    longitude = torch.arange(64)[None, :] * (2 * math.pi / 64)
    return (theta.sin() * longitude.cos())[None, None, None]


@pytest.mark.parametrize("nlat", [32, 33, 120])
def test_spherical_power_normalization_and_degree_isolation(nlat) -> None:
    loss = make_loss((nlat, 64), bands=((1, 4),))
    torch.testing.assert_close(loss.latitude_weights.sum(), torch.tensor(1.0))
    power = loss._power(harmonic_field(nlat))
    torch.testing.assert_close(power[0, 1], torch.tensor(4 * math.pi / 3), rtol=1e-5, atol=1e-6)
    assert power[0, [0, 2, 3, 4]].sum() < 1e-10


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_log_power_ratio_and_restoring_gradient(scale: float) -> None:
    loss = make_loss((33, 64), bands=((1, 4),))
    target = harmonic_field().requires_grad_()
    amplitude = torch.tensor(scale, requires_grad=True)
    value = loss(amplitude * target.detach(), target)
    torch.testing.assert_close(value, torch.tensor(math.log(scale**2) ** 2), rtol=1e-5, atol=1e-5)
    value.backward()
    assert amplitude.grad.item() * (scale - 1) > 0
    assert target.grad is None


def test_units_offsets_and_batch_size_do_not_change_loss() -> None:
    loss = make_loss((33, 64), bands=((1, 2), (3, 4)))
    target = harmonic_field()
    prediction = target * 0.7
    expected = loss(prediction, target)
    torch.testing.assert_close(loss(prediction * 100 + 7, target * 100 + 12), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(loss(prediction.expand(3, 2, 4, -1, -1), target.expand(3, 2, 4, -1, -1)), expected)


def test_constant_fields_and_autocast_have_finite_gradients() -> None:
    loss = make_loss((33, 64), bands=((1, 4),))
    output = torch.zeros(1, 1, 1, 33, 64, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = loss(output, torch.ones_like(output))
    assert value.dtype == torch.float32
    value.backward()
    assert value == 0
    assert torch.isfinite(output.grad).all()
    assert not loss.state_dict()


def test_separate_bands_detect_redistributed_power() -> None:
    loss = make_loss((33, 64), bands=((1, 1), (2, 2)))
    x = torch.linspace(0, math.pi, 33).cos()[:, None]
    y1 = math.sqrt(3 / (4 * math.pi)) * x + torch.zeros(1, 64)
    y2 = math.sqrt(5 / (16 * math.pi)) * (3 * x.square() - 1) + torch.zeros(1, 64)
    target = y1 + y2
    output = math.sqrt(0.5) * y1 + math.sqrt(1.5) * y2
    torch.testing.assert_close(loss._power(output).sum(), loss._power(target).sum(), rtol=1e-5, atol=1e-5)
    assert loss(output, target) > 0.1
    torch.testing.assert_close(loss(target, target), torch.tensor(0.0))


@pytest.mark.parametrize("size", [(32, 64), (33, 64), (261, 128)])
def test_invalid_spectral_grid_fails_at_construction(size) -> None:
    with pytest.raises(ValueError):
        SphericalBandPowerLoss(size)


def test_group_vae_grid_resolves_highest_configured_degree() -> None:
    loss = make_loss((120, 240), bands=((1, 4), (5, 20), (21, 40), (41, 59)))
    colatitude = torch.linspace(0, math.pi, 120, dtype=torch.float64)
    field = math.sqrt(119 / (4 * math.pi)) * torch.special.legendre_polynomial_p(colatitude.cos(), 59)
    power = loss.band_power(field[:, None].expand(-1, 240))
    torch.testing.assert_close(power, torch.tensor([0.0, 0.0, 0.0, 1.0]), rtol=1e-5, atol=1e-6)


def test_production_bands_are_fixed() -> None:
    assert SphericalBandPowerLoss.BANDS == ((1, 4), (5, 20), (21, 40), (41, 64), (65, 128))
