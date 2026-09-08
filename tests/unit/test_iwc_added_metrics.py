import numpy as np
import pytest
import torch

from xmetai.metrics import ForecastFidelity, FrequencyBias, RadarFSS, ThreatScore


def test_zonal_power_matches_numpy_reference(monkeypatch):
    monkeypatch.setattr(ForecastFidelity, "BANDS", ((1, 2), (3, 4)))
    metric = ForecastFidelity(["x"], [6], (17, 32))
    rng = np.random.default_rng(1)
    field = rng.normal(size=(2, 1, 1, 17, 32)).astype("float32")
    weights = np.cos(np.linspace(-np.pi / 2, np.pi / 2, 17))
    centered = field - field.mean(-1, keepdims=True)
    power = (abs(np.fft.rfft(centered, axis=-1) / 32) ** 2 * weights[:, None]).sum(-2) / weights.sum()
    expected = np.stack([power[..., 1:3].sum(-1), power[..., 3:5].sum(-1)], -1)
    np.testing.assert_allclose(metric._zonal_band_power(torch.from_numpy(field)).numpy(), expected, rtol=1e-5)
    metric.process({}, {"output": torch.from_numpy(field) * 2, "target": torch.from_numpy(field)})
    assert metric.evaluate()["ZonalPowerRatio_1_2"]["x"] == pytest.approx([4.0])


def test_precip_fss_streaming_and_neighborhood():
    pred = torch.zeros(2, 2, 1, 8, 8)
    obs = pred.clone()
    pred[..., 3, 3] = 20
    obs[..., 3, 4] = 20
    scores = []
    for window in [1, 3]:
        metric = RadarFSS([13], test_times=[6, 12], test_names=["tp"], channel_name="tp", neighborhood_size=window, empty_score=float("nan"))
        metric.reset()
        metric.process({}, dict(output=pred, target=obs))
        full = metric.evaluate()[metric.metric_name]["FSS"]
        metric.reset()
        for b in range(2):
            for t in range(2):
                metric.process({}, dict(output=pred[b : b + 1, t : t + 1], target=obs[b : b + 1, t : t + 1], frame_start=t))
        assert metric.evaluate()[metric.metric_name]["FSS"] == pytest.approx(full)
        scores.append(full[0])
    assert scores[0] == pytest.approx(0, abs=1e-6) and scores[1] > scores[0]
    metric.reset()
    metric.process({}, dict(output=obs, target=obs))
    assert metric.evaluate()[metric.metric_name]["FSS"] == pytest.approx([1, 1])
    metric.reset()
    metric.process({}, dict(output=obs * 0, target=obs * 0))
    assert np.isnan(metric.evaluate()[metric.metric_name]["FSS"]).all()


@pytest.mark.parametrize("cls", [ThreatScore, FrequencyBias])
def test_composite_weights_and_empty_thresholds(cls):
    metric = cls([1, 10, 100], test_times=[6], composite_weights=[1, 3, 2])
    metric.reset()
    pred = torch.tensor([2.0, 20.0, 0.0]).reshape(1, 1, 1, 1, 3)
    obs = torch.tensor([2.0, 0.0, 20.0]).reshape(1, 1, 1, 1, 3)
    metric.process({}, dict(output=pred, target=obs))
    result = metric.evaluate()[metric.metric_name]
    expected = (result[f"{metric.metric_prefix}@1"][0] + 3 * result[f"{metric.metric_prefix}@10"][0]) / 4
    assert result["Composite"][0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        cls([1, 10], test_times=[6], composite_weights=[1])
