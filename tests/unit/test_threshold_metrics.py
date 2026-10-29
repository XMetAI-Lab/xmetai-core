import math

import pytest
import torch

from xmetai.metrics.threshold import FalseAlarmRate, FrequencyBias, ProbabilityOfDetection, ThreatScore, ThreatScore_nan


def test_threshold_metrics_score_contingency_table():
    metrics = [
        ThreatScore(thresholds=(35.0, 50.0), test_times=(0,), neighborhood_size=1),
        FalseAlarmRate(thresholds=(35.0, 50.0), test_times=(0,), neighborhood_size=1),
        ProbabilityOfDetection(thresholds=(35.0, 50.0), test_times=(0,), neighborhood_size=1),
        FrequencyBias(thresholds=(35.0, 50.0), test_times=(0,), neighborhood_size=1),
    ]
    target = torch.zeros(1, 1, 1, 8, 8)
    output = torch.zeros_like(target)
    target[..., 2:4, 2:4] = 45.0
    output[..., 2:4, 2:4] = 45.0
    output[..., 6, 6] = 45.0

    results = {}
    for metric in metrics:
        metric.reset()
        metric.process({}, {"output": output, "target": target})
        metric_results = metric.evaluate()
        assert metric_results is not None
        results.update(metric_results)

    assert results["ThreatScore"]["Times"] == [0]
    assert list(results["ThreatScore"]) == ["Times", "TS@35", "TS@50"]
    assert abs(results["ThreatScore"]["TS@35"][0] - 0.8) < 1.0e-6
    assert results["ThreatScore"]["TS@50"] == [0.0]
    assert results["FalseAlarmRate"]["Times"] == [0]
    assert abs(results["FalseAlarmRate"]["FAR@35"][0] - 0.2) < 1.0e-6
    assert results["FalseAlarmRate"]["FAR@50"] == [0.0]
    assert results["ProbabilityOfDetection"]["Times"] == [0]
    assert abs(results["ProbabilityOfDetection"]["POD@35"][0] - 1.0) < 1.0e-6
    assert results["ProbabilityOfDetection"]["POD@50"] == [0.0]
    assert results["FrequencyBias"]["Times"] == [0]
    assert abs(results["FrequencyBias"]["Bias@35"][0] - 1.25) < 1.0e-6
    assert results["FrequencyBias"]["Bias@50"] == [0.0]


def test_threshold_metrics_require_test_times():
    with pytest.raises(TypeError, match="test_times"):
        ThreatScore(thresholds=(35.0,), neighborhood_size=1)


def test_threshold_metrics_use_test_times_with_frame_start():
    metric = ThreatScore(thresholds=(35.0, 50.0), test_times=(6, 12, 18), neighborhood_size=1)
    metric.reset()

    target = torch.zeros(1, 2, 1, 4, 4)
    output = torch.zeros_like(target)
    target[:, 0, 0, 1, 1] = 45.0
    output[:, 0, 0, 1, 1] = 45.0
    target[:, 1, 0, 1, 1] = 45.0
    output[:, 1, 0, 3, 3] = 45.0

    metric.process({}, {"output": output, "target": target, "frame_start": 1})
    results = metric.evaluate()
    assert results is not None
    assert results["ThreatScore"]["Times"] == [6, 12, 18]
    assert results["ThreatScore"]["TS@35"][0] == 0.0
    assert abs(results["ThreatScore"]["TS@35"][1] - 1.0) < 1.0e-6
    assert results["ThreatScore"]["TS@35"][2] == 0.0
    assert results["ThreatScore"]["TS@50"] == [0.0, 0.0, 0.0]


def test_threshold_metrics_apply_value_scale_before_mm_threshold():
    metric = ThreatScore(thresholds=(10.0,), test_times=(6,), value_scale=21.6)
    metric.reset()
    target = torch.full((1, 1, 1, 2, 2), 0.5)
    output = target.clone()

    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()

    assert results is not None
    assert results["ThreatScore"]["Times"] == [6]
    assert results["ThreatScore"]["TS@10"][0] == pytest.approx(1.0)


def test_threshold_metrics_reject_non_positive_value_scale():
    with pytest.raises(ValueError, match="value_scale"):
        ThreatScore(thresholds=(10.0,), test_times=(6,), value_scale=0.0)


def test_threat_score_nan_uses_competition_missing_value_convention():
    metric = ThreatScore_nan(thresholds=(1.0,), test_times=(6, 12), neighborhood_size=1)
    metric.reset()
    output = torch.zeros(1, 2, 1, 2, 2)
    target = torch.zeros_like(output)
    output[0, 0, 0, 0, 0] = 1.0
    target[0, 0, 0, 0, 0] = 1.0

    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()

    assert results is not None
    assert results["ThreatScore_nan"]["TS_nan@1"][0] == 1.0
    assert math.isnan(results["ThreatScore_nan"]["TS_nan@1"][1])
