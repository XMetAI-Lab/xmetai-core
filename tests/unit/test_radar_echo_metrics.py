import pytest
import torch

from xmetai.metrics.radar_echo import FSS, LPIPS, SSIM, RadarFSS, RadarLPIPS, RadarSSIM


def test_radar_ssim_identical_fields_score_one():
    metric = RadarSSIM(test_times=(6,), data_range=70.0, window_size=7)
    field = torch.zeros(2, 1, 1, 16, 16)
    field[..., 4:10, 4:10] = 45.0
    metric.reset()
    metric.process({}, {"output": field, "target": field})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarSSIM"]["Times"] == [6]
    assert results["RadarSSIM"]["SSIM"][0] > 0.99


def test_radar_ssim_different_fields_score_below_one():
    metric = SSIM(test_times=(12,), data_range=70.0, window_size=7)
    target = torch.zeros(1, 1, 1, 16, 16)
    output = torch.zeros_like(target)
    target[..., 4:10, 4:10] = 45.0
    output[..., 10:14, 10:14] = 45.0
    metric.reset()
    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarSSIM"]["SSIM"][0] < 0.95


def test_radar_ssim_uses_test_times_with_frame_start():
    metric = RadarSSIM(test_times=(6, 12, 18), data_range=70.0, window_size=7)
    target = torch.zeros(1, 2, 1, 8, 8)
    output = torch.zeros_like(target)
    target[:, 0, 0, 2, 2] = 40.0
    output[:, 0, 0, 2, 2] = 40.0
    target[:, 1, 0, 2, 2] = 40.0
    output[:, 1, 0, 5, 5] = 40.0

    metric.reset()
    metric.process({}, {"output": output, "target": target, "frame_start": 1})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarSSIM"]["Times"] == [6, 12, 18]
    assert results["RadarSSIM"]["SSIM"][0] == 0.0
    assert results["RadarSSIM"]["SSIM"][1] > 0.99
    assert results["RadarSSIM"]["SSIM"][2] < 0.95


def test_radar_fss_identical_fields_score_one():
    metric = RadarFSS(thresholds=(35.0,), test_times=(6,), neighborhood_size=5)
    field = torch.zeros(1, 1, 1, 16, 16)
    field[..., 4:10, 4:10] = 45.0
    metric.reset()
    metric.process({}, {"output": field, "target": field})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarFSS"]["Times"] == [6]
    assert results["RadarFSS"]["Threshold"] == [35.0]
    assert abs(results["RadarFSS"]["FSS"][0] - 1.0) < 1.0e-6


def test_radar_fss_partial_overlap_below_one():
    metric = FSS(thresholds=(35.0,), test_times=(6,), neighborhood_size=3)
    target = torch.zeros(1, 1, 1, 8, 8)
    output = torch.zeros_like(target)
    target[..., 2:4, 2:4] = 45.0
    output[..., 5:7, 5:7] = 45.0
    metric.reset()
    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarFSS"]["FSS"][0] < 1.0


def test_radar_fss_uses_test_times_with_frame_start():
    metric = RadarFSS(thresholds=(35.0,), test_times=(6, 12, 18), neighborhood_size=1)
    target = torch.zeros(1, 2, 1, 8, 8)
    output = torch.zeros_like(target)
    target[:, 0, 0, 2:4, 2:4] = 45.0
    output[:, 0, 0, 2:4, 2:4] = 45.0
    target[:, 1, 0, 2:4, 2:4] = 45.0
    output[:, 1, 0, 5:7, 5:7] = 45.0

    metric.reset()
    metric.process({}, {"output": output, "target": target, "frame_start": 1})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarFSS"]["Times"] == [6, 12, 18]
    assert results["RadarFSS"]["Threshold"] == [35.0, 35.0, 35.0]
    assert results["RadarFSS"]["FSS"][0] == 0.0
    assert abs(results["RadarFSS"]["FSS"][1] - 1.0) < 1.0e-6
    assert results["RadarFSS"]["FSS"][2] < 0.1

def test_radar_fss_requires_test_times():
    with pytest.raises(TypeError, match="test_times"):
        RadarFSS(thresholds=(35.0,), neighborhood_size=5)


def test_radar_fss_requires_odd_neighborhood_size():
    with pytest.raises(ValueError, match="neighborhood_size must be odd"):
        RadarFSS(thresholds=(35.0,), test_times=(6,), neighborhood_size=4)


def test_radar_ssim_requires_odd_window_size():
    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        RadarSSIM(test_times=(6,), window_size=10)


@pytest.mark.parametrize("metric_cls", [RadarSSIM, RadarFSS, RadarLPIPS])
def test_radar_metrics_validate_tensor_shape(metric_cls):
    if metric_cls is RadarFSS:
        metric = metric_cls(thresholds=(35.0,), test_times=(6,))
    else:
        metric = metric_cls(test_times=(6,))
    metric.reset()
    with pytest.raises(ValueError, match=r"\(B, T, C, H, W\)"):
        metric.process({}, {"output": torch.zeros(1, 1, 8, 8), "target": torch.zeros(1, 1, 8, 8)})


def test_radar_lpips_identical_fields_score_near_zero():
    pytest.importorskip("lpips")
    metric = RadarLPIPS(test_times=(6,), data_min=0.0, data_max=70.0)
    field = torch.zeros(1, 1, 1, 32, 32)
    field[..., 8:20, 8:20] = 45.0
    metric.reset()
    metric.process({}, {"output": field, "target": field})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarLPIPS"]["Times"] == [6]
    assert results["RadarLPIPS"]["LPIPS"][0] < 0.05


def test_radar_lpips_different_fields_score_higher():
    pytest.importorskip("lpips")
    metric = LPIPS(test_times=(6,), data_min=0.0, data_max=70.0)
    target = torch.zeros(1, 1, 1, 32, 32)
    output = torch.zeros_like(target)
    target[..., 8:20, 8:20] = 45.0
    output[..., 20:28, 20:28] = 45.0
    metric.reset()
    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarLPIPS"]["LPIPS"][0] > 0.05
