import pytest
import torch
import torch.nn.functional as F

from xmetai.metrics.radar_echo import FSS, LPIPS, RAPSD, SSIM, RadarFSS, RadarLPIPS, RadarRAPSD, RadarSSIM


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


def test_radar_fss_multiple_windows_match_single_windows():
    sizes = (1, 3, 5, 15, 31, 63)
    kwargs = dict(thresholds=(13.0, 35.0), test_times=(6, 12, 18), channel_name="tp", test_names=("t2m", "tp"), value_scale=21.6, empty_score=float("nan"))
    merged = RadarFSS(neighborhood_sizes=sizes, metric_name="PrecipFSS", **kwargs)
    singles = [RadarFSS(neighborhood_size=size, **kwargs) for size in sizes]
    target = torch.zeros(2, 1, 2, 8, 8)
    target[:, :, 1, 2:5, 2:5] = 2.0
    output = torch.roll(target, shifts=2, dims=-1)
    for metric in [merged, *singles]:
        metric.reset()
        for _ in range(2):
            metric.process({}, {"output": output, "target": target, "frame_start": 1})
        metric.process({}, {"output": torch.zeros_like(output), "target": torch.zeros_like(target), "frame_start": 2})
    table = merged.evaluate()["PrecipFSS"]
    assert list(table) == ["Times", "Threshold", *[f"FSS{size}" for size in sizes]]
    for size, single in zip(sizes, singles):
        expected = single.evaluate()["RadarFSS"]
        assert table["Times"] == expected["Times"]
        assert table["Threshold"] == expected["Threshold"]
        torch.testing.assert_close(torch.tensor(table[f"FSS{size}"]), torch.tensor(expected["FSS"]), rtol=0, atol=0, equal_nan=True)
        assert table[f"FSS{size}"][:2] == [0.0, 0.0]
        assert torch.tensor(table[f"FSS{size}"][-2:]).isnan().all()
    merged.reset()
    reset_table = merged.evaluate()["PrecipFSS"]
    assert all(reset_table[f"FSS{size}"] == [0.0] * 6 for size in sizes)


@pytest.mark.parametrize("sizes", [(), (3, 3), (0,), (2,)])
def test_radar_fss_rejects_invalid_multiple_windows(sizes):
    with pytest.raises(ValueError):
        RadarFSS(thresholds=(13.0,), test_times=(6,), neighborhood_sizes=sizes)


def test_radar_fss_multiple_windows_gather_workers(monkeypatch):
    metrics = [RadarFSS(thresholds=(13.0,), test_times=(6,), neighborhood_sizes=(1, 3)) for _ in range(3)]
    first, second, combined = metrics
    for metric in metrics:
        metric.reset()
    target = torch.full((1, 1, 1, 8, 8), 20.0)
    for worker, output in zip((first, second), (target, torch.zeros_like(target))):
        outputs = {"output": output, "target": target}
        worker.process({}, outputs)
        combined.process({}, outputs)
    expected = combined.evaluate()
    monkeypatch.setattr("xmetai.metrics.radar_echo.comm.synchronize", lambda: None)
    monkeypatch.setattr("xmetai.metrics.radar_echo.comm.is_main_process", lambda: True)
    worker_states = []
    monkeypatch.setattr("xmetai.metrics.radar_echo.comm.gather", lambda state, dst: worker_states.append(state) or [state])
    first.evaluate()
    second.evaluate()
    monkeypatch.setattr("xmetai.metrics.radar_echo.comm.gather", lambda state, dst: worker_states)
    assert first.evaluate() == expected
    monkeypatch.setattr("xmetai.metrics.radar_echo.comm.is_main_process", lambda: False)
    assert second.evaluate() is None


def test_radar_ssim_requires_odd_window_size():
    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        RadarSSIM(test_times=(6,), window_size=10)


@pytest.mark.parametrize("metric_cls", [RadarSSIM, RadarFSS, RadarLPIPS, RadarRAPSD])
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


def test_radar_rapsd_identical_fields_score_near_zero():
    metric = RadarRAPSD(test_times=(6,))
    field = torch.zeros(2, 1, 1, 16, 16)
    field[..., 4:12, 4:12] = 45.0
    field[..., 10:14, 10:14] = 20.0
    metric.reset()
    metric.process({}, {"output": field, "target": field})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarRAPSD"]["Times"] == [6]
    assert results["RadarRAPSD"]["RAPSD"][0] < 1.0e-6


def test_radar_rapsd_blurred_fields_score_higher():
    metric = RAPSD(test_times=(6,))
    target = torch.zeros(1, 1, 1, 16, 16)
    target[..., 4:12, 4:12] = 45.0
    torch.manual_seed(7)
    target += torch.randn_like(target) * 2.0
    blurred = F.avg_pool2d(target[:, 0], kernel_size=3, stride=1, padding=1)
    output = blurred.unsqueeze(1)
    metric.reset()
    metric.process({}, {"output": output, "target": target})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarRAPSD"]["RAPSD"][0] > 0.01


def test_radar_rapsd_uses_cpu_for_gpu_inputs():
    metric = RadarRAPSD(test_times=(6,))
    field = torch.zeros(1, 1, 1, 16, 16)
    field[..., 4:10, 4:10] = 40.0
    if torch.cuda.is_available():
        field = field.cuda()
    metric.reset()
    metric.process({}, {"output": field, "target": field})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarRAPSD"]["RAPSD"][0] < 1.0e-6


def test_radar_rapsd_uses_test_times_with_frame_start():
    metric = RadarRAPSD(test_times=(6, 12, 18))
    target = torch.zeros(1, 2, 1, 16, 16)
    output = torch.zeros_like(target)
    target[:, 0, 0, 4:10, 4:10] = 40.0
    output[:, 0, 0, 4:10, 4:10] = 40.0
    target[:, 1, 0, 4:10, 4:10] = 40.0
    output[:, 1, 0, 10:14, 10:14] = 40.0

    metric.reset()
    metric.process({}, {"output": output, "target": target, "frame_start": 1})
    results = metric.evaluate()
    assert results is not None
    assert results["RadarRAPSD"]["Times"] == [6, 12, 18]
    assert results["RadarRAPSD"]["RAPSD"][0] == 0.0
    assert results["RadarRAPSD"]["RAPSD"][1] < 1.0e-6
    assert results["RadarRAPSD"]["RAPSD"][2] > 0.01
