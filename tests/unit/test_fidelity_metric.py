import math

import numpy as np
import pytest
import torch

from xmetai.loss import SphericalBandPowerLoss
from xmetai.metrics import ForecastFidelity


def build_metric(monkeypatch):
    monkeypatch.setattr(SphericalBandPowerLoss, "BANDS", ((1, 2), (3, 4)))
    return ForecastFidelity(["z500"], [6, 12], (17, 32))


def fields():
    theta = torch.linspace(0, math.pi, 17)
    target = theta.cos()[None, None, None, :, None].expand(3, 2, 1, 17, 32).clone()
    target *= torch.tensor([1., 2., 3.])[:, None, None, None, None]
    return target


def test_fidelity_amplitude_bias_and_stream_aggregation(monkeypatch):
    metric = build_metric(monkeypatch)
    target = fields()
    prediction = 2 * target + 3
    metric.process({}, dict(output=prediction, target=target))
    full = metric.evaluate()
    assert full["Bias"]["z500"] == pytest.approx([3, 3])
    assert full["SpatialAmplitudeRatio"]["z500"] == pytest.approx([2, 2])
    assert full["BandPowerRatio_1_2"]["z500"] == pytest.approx([4, 4])
    metric.reset()
    for batch in [slice(0, 1), slice(1, 3)]:
        for frame in range(2):
            metric.process({}, dict(output=prediction[batch, frame:frame+1], target=target[batch, frame:frame+1], frame_start=frame))
    result = metric.evaluate()
    for name in full:
        np.testing.assert_allclose(result[name]["z500"], full[name]["z500"], rtol=1e-5, equal_nan=True)


def test_fidelity_rank_merge_and_modes(monkeypatch):
    metric = build_metric(monkeypatch)
    target = fields()
    metric.process({}, dict(output_by_mode={"prior": target[:1]}, target=target[:1]))
    first = metric.statistics
    metric.reset()
    metric.process({}, dict(output_by_mode={"prior": 2*target[1:]}, target=target[1:]))
    second = metric.statistics
    monkeypatch.setattr("xmetai.metrics.fidelity.comm.gather", lambda value, dst: [first, second])
    result = metric.evaluate()
    assert result["prior/BandPowerRatio_1_2"]["z500"] == pytest.approx([53/14, 53/14])
    assert result["prior/SpatialAmplitudeRatio"]["z500"] == pytest.approx([math.sqrt(53/14)]*2)


def test_fidelity_constant_and_invalid_frames(monkeypatch):
    metric = build_metric(monkeypatch)
    target = torch.zeros_like(fields())
    metric.process({}, dict(output=target, target=target))
    assert np.isnan(metric.evaluate()["SpatialAmplitudeRatio"]["z500"]).all()
    with pytest.raises(ValueError, match="frames"):
        metric.process({}, dict(output=target, target=target, frame_start=1))
