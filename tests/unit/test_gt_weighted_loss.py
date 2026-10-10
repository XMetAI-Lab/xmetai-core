import math

import pytest
import torch

from xmetai.loss import GTWeightedLoss
from xmetai.recipes.gt_loss import gt_loss_profiles


@pytest.fixture(autouse=True)
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("GTWeightedLoss requires CUDA")
    with torch.device("cuda"):
        yield


def weights_from_gradient(loss, target):
    # A unit prediction error has a known Charbonnier derivative.
    output = (target + 1).requires_grad_()
    (gradient,) = torch.autograd.grad(loss(output, target), output)
    return gradient * target.numel() * math.sqrt(1 + loss.eps)


@pytest.fixture
def features():
    return {
        "t2m": {"mode": "both", "thresholds": [1, 2, 3, 4, 6, 7, 8, 9]},
        "msl": {"mode": "lower", "thresholds": [2, 4, 6, 8]},
        "tp": {"mode": "upper", "thresholds": [2, 4, 6, 8]},
        "wind10": {"mode": "upper", "thresholds": [2, 4, 6, 8], "channels": ["u10", "v10"]},
        "sst": {"mode": "both", "thresholds": [1, 2, 3, 4, 6, 7, 8, 9]},
    }


def make_loss(features, names, **kwargs):
    return GTWeightedLoss(names, torch.zeros(len(names)), torch.ones(len(names)), features, **kwargs)


def test_tail_directions_gradient_ratios_and_unselected_channels(features):
    loss = make_loss(features, ["t2m", "msl", "tp", "z500"])
    target = torch.full((1, 1, 4, 2, 3), 5.0)
    target[..., 0] = 0
    target[..., 2] = 10
    weights = weights_from_gradient(loss, target)
    torch.testing.assert_close(weights[0, 0, :, 0], torch.tensor([[5, 1, 5], [5, 3, 1], [1, 3, 5], [1, 1, 1]], dtype=weights.dtype))
    output = (target + 1).requires_grad_()
    value = loss(output, target)
    value.backward()
    assert torch.allclose(output.grad / output.grad[0, 0, 0, 0, 1], weights)
    assert value.item() == pytest.approx((math.sqrt(1 + 1e-6) - 1e-3) * weights.mean().item())


def test_weights_are_independent_of_other_scenes_and_grid_shape(features):
    loss = make_loss(features, ["t2m"])
    target = torch.tensor([0, 5, 10], dtype=torch.float32).reshape(3, 1, 1, 1, 1)
    together = weights_from_gradient(loss, target)
    separately = torch.cat([weights_from_gradient(loss, t[None]) for t in target])
    torch.testing.assert_close(together, separately)
    assert weights_from_gradient(loss, target.expand(-1, -1, -1, 4, 6)).shape == (3, 1, 1, 4, 6)


def test_wind_uses_physical_speed_and_shares_weights(features):
    loss = GTWeightedLoss(["u10", "v10"], torch.tensor([2.0, -3.0]), torch.tensor([2.0, 5.0]), features)
    physical = torch.tensor([[10, -10, 10 / math.sqrt(2)], [0, 0, 10 / math.sqrt(2)]])
    target = ((physical - torch.tensor([2.0, -3.0])[:, None]) / torch.tensor([2.0, 5.0])[:, None]).reshape(1, 1, 2, 1, 3)
    torch.testing.assert_close(weights_from_gradient(loss, target), torch.full_like(target, 5))
    output = target.clone()
    output[:, :, 1] += 1
    output.requires_grad_()
    loss(output, target).backward()
    assert (output.grad[:, :, 1] > 0).all()


def test_pressure_level_wind_is_not_surface_wind(features):
    loss = make_loss(features, ["t2m", "u100", "v100"])
    target = torch.full((1, 1, 3, 2, 3), 1000.0)
    torch.testing.assert_close(weights_from_gradient(loss, target)[:, :, 1:], torch.ones_like(target[:, :, 1:]))


def test_normalization_preserves_bins(features):
    a = make_loss(features, ["t2m"])
    b = GTWeightedLoss(["t2m"], torch.tensor([10.0]), torch.tensor([2.0]), features)
    physical = torch.tensor([0, 5, 10], dtype=torch.float32).reshape(1, 1, 1, 1, 3)
    torch.testing.assert_close(weights_from_gradient(a, physical), weights_from_gradient(b, (physical - 10) / 2))


def test_duplicate_cutoffs_and_exact_boundary_ties(features):
    features["tp"]["thresholds"] = [0, 0, 3, 4]
    loss = make_loss(features, ["tp"])
    target = torch.tensor([0, 1, 3, 4, 5], dtype=torch.float32).reshape(1, 1, 1, 1, 5)
    torch.testing.assert_close(weights_from_gradient(loss, target).flatten(), torch.tensor([1, 3, 3, 4, 5], dtype=target.dtype))
    assert loss(torch.ones(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1)) > 0


def test_mask_nan_targets_and_all_invalid_gradients(features):
    loss = make_loss(features, ["sst"])
    target = torch.full((1, 1, 1, 2, 3), 5.0)
    target[..., 1, 0] = float("nan")
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 0, 0] = False
    output = (target + 1).requires_grad_()
    result = loss(output, target, mask=mask)
    expected = math.sqrt(1 + 1e-6) - 1e-3
    assert torch.isfinite(result).all() and result.sum().item() == pytest.approx(expected, rel=1e-5)
    result.sum().backward()
    assert torch.isfinite(output.grad).all() and output.grad[..., 0].eq(0).all()
    output.grad.zero_()
    empty = loss(output, target, mask=torch.zeros_like(target, dtype=torch.bool))
    empty.sum().backward()
    assert empty.eq(0).all() and output.grad.eq(0).all()


def test_one_invalid_wind_component_excludes_both(features):
    loss = make_loss(features, ["u10", "v10"])
    target = torch.ones(1, 1, 2, 2, 3)
    target[:, :, 0] = float("nan")
    output = torch.ones_like(target, requires_grad=True)
    value = loss(output, target)
    value.backward()
    assert value == 0 and output.grad.eq(0).all()


def test_external_weights_are_outside_point_loss_and_gt_is_detached(features):
    loss = make_loss(features, ["tp"])
    target = torch.zeros(1, 1, 1, 2, 3, requires_grad=True)
    output = torch.full_like(target, 1e-4, requires_grad=True)
    base = loss(output, target)
    weighted = loss(output, target, weight=torch.tensor(3.0))
    assert weighted.item() == pytest.approx(3 * base.item())
    weighted.backward()
    assert target.grad is None and (output.grad > 0).all()


@pytest.mark.parametrize("dataset", list(gt_loss_profiles))
def test_all_saved_dataset_configs_work_without_files(dataset, tmp_path, monkeypatch):
    config = gt_loss_profiles[dataset]["features"]
    names = [name for key, spec in config.items() for name in spec.get("channels", (key,))]
    monkeypatch.chdir(tmp_path)  # No data/statistics directory is available.
    loss = make_loss(config, names)
    assert len(loss.kernel_specs) == len(config)
    target = torch.zeros(1, 1, len(names), 2, 3, dtype=torch.bfloat16)
    output = torch.ones_like(target, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = loss(output, target)
    value.backward()
    assert value.dtype == torch.float32 and torch.isfinite(output.grad).all()


def test_atmosphere_land_configs_can_be_combined():
    config = {**gt_loss_profiles["era_s2s"]["features"], **gt_loss_profiles["era_land"]["features"]}
    loss = make_loss(config, ["tp", "stl1", "swvl1"])
    assert len(loss.kernel_specs) == 3
    assert not loss.state_dict()
    assert not list(loss.buffers())


@pytest.mark.parametrize("problem", ["shape", "duplicate", "partial_wind", "mode", "std", "nonmonotonic", "nan_cutoff"])
def test_bad_configuration_fails_at_initialization(features, problem):
    args = dict(channel_names=["t2m"], mean=torch.zeros(1), std=torch.ones(1), features=features)
    if problem == "shape":
        features["t2m"]["thresholds"] = [1, 2]
    elif problem == "duplicate":
        features["other"] = {**features["t2m"], "channels": ["t2m"]}
    elif problem == "partial_wind":
        args["channel_names"] = ["u10"]
    elif problem == "mode":
        features["t2m"]["mode"] = "bad"
    elif problem == "std":
        args["std"] = torch.zeros(1)
    elif problem == "nonmonotonic":
        features["t2m"]["thresholds"][0] = 1000
    elif problem == "nan_cutoff":
        features["t2m"]["thresholds"][0] = float("nan")
    with pytest.raises(ValueError):
        GTWeightedLoss(**args)
