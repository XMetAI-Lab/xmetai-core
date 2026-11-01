"""Physical units, geography and member semantics of initial perturbations."""

import math

import pytest
import torch

from xmetai.utils.physical_perturb import physical_input_perturb


def test_geostrophic_increments_have_requested_rms_and_member_semantics():
    names = ["z500", "u500", "v500", "z850", "u850", "v850", "t500"]
    source = torch.zeros(2, 2, len(names), 721, 1440)
    lat = torch.linspace(-math.pi / 2, math.pi / 2, 721)
    weight = lat.cos().clamp_min(0)[:, None]
    torch.manual_seed(4)
    output = physical_input_perturb(source, weight, names, height_rms=3.0)
    assert not source.any()
    assert torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1])
    torch.testing.assert_close(output[:, 0], output[:, 1])
    torch.testing.assert_close(output[:, :, :3], output[:, :, 3:6])
    assert not output[:, :, 6].any()
    phi = output[:, 0, 0]
    weights = weight.expand(721, 1440)
    rms = ((phi.square() * weights).sum((-2, -1)) / weights.sum()).sqrt() / 9.80665
    torch.testing.assert_close(rms, torch.full((2,), 3.0))
    assert not output[..., lat.abs() < math.radians(19), :].any()
    dy = (phi[:, 2:] - phi[:, :-2]) / (2 * math.pi / 720 * 6371000)
    dx = (phi.roll(-1, -1) - phi.roll(1, -1)) / (2 * (2 * math.pi / 1440) * 6371000 * lat.cos().clamp_min(0.1)[None, :, None])
    mask = (lat.abs() >= math.radians(21)) & (lat.abs() <= math.radians(79))
    f = 2 * 7.292115e-5 * lat.sin()
    torch.testing.assert_close(output[:, 0, 1, 1:-1][:, mask[1:-1]], (-dy / f[None, 1:-1, None])[:, mask[1:-1]])
    torch.testing.assert_close(output[:, 0, 2][:, mask], (dx / f[None, :, None])[:, mask])


def test_latitude_order_and_zero_amplitude():
    names = ["z500", "u500", "v500"]
    source = torch.zeros(1, 1, 3, 721, 1440)
    weight = torch.linspace(-math.pi / 2, math.pi / 2, 721).cos().clamp_min(0)[:, None]
    torch.manual_seed(1)
    ascending = physical_input_perturb(source, weight, names)
    torch.manual_seed(1)
    descending = physical_input_perturb(source.flip(-2), weight.flip(0), names, latitude_ascending=False)
    torch.testing.assert_close(descending.flip(-2), ascending, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(physical_input_perturb(source, weight, names, height_rms=0), source)


@pytest.mark.parametrize("amplitude", [-1.0, float("nan"), float("inf")])
def test_invalid_amplitude_fails(amplitude):
    with pytest.raises(ValueError, match="height_rms"):
        physical_input_perturb(torch.zeros(1, 1, 3, 721, 1440), torch.ones(721, 1), ["z500", "u500", "v500"], height_rms=amplitude)


def test_raw_cosine_polar_roundoff_preserves_physical_rms():
    source = torch.zeros(1, 1, 3, 721, 1440)
    weight = torch.linspace(-math.pi / 2, math.pi / 2, 721).cos()[:, None]
    assert weight[0] < 0 and weight[-1] < 0
    original = weight.clone()
    torch.manual_seed(7)
    actual = physical_input_perturb(source, weight, ["z500", "u500", "v500"])
    torch.manual_seed(7)
    expected = physical_input_perturb(source, weight.clamp_min(0), ["z500", "u500", "v500"])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weight, original)
    weights = weight.clamp_min(0).expand(721, 1440)
    rms = ((actual[0, 0, 0].square() * weights).sum() / weights.sum()).sqrt() / 9.80665
    torch.testing.assert_close(rms, torch.tensor(5.0))


@pytest.mark.parametrize("row,value", [(0, -0.01), (360, -1e-8), (360, float("nan")), (360, float("inf"))])
def test_invalid_weights_still_fail(row, value):
    weight = torch.ones(721, 1)
    weight[row] = value
    with pytest.raises(ValueError, match="weight must be finite"):
        physical_input_perturb(torch.zeros(1, 1, 3, 721, 1440), weight, ["z500", "u500", "v500"])
