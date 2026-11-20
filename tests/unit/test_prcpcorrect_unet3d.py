"""Unit tests for UNet3D, correction losses and online precipitation metrics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import xmetai.metrics.prcpcorrect as metric_module
from xmetai.config import LazyConfig, instantiate
from xmetai.data.weather.prcpcorrect_dataset import PrcpCorrectionUNet3DDataset
from xmetai.metrics.prcpcorrect import PrcpMaskedCSI, PrcpMaskedRMSE
from xmetai.models.adapter.prcpcorrect_unet3d import PrcpCorrectionUNet3D


def test_training_config_instantiates_w1448(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[2] / "configs" / "prcpcorrect_unet3d.py"
    input_min = tmp_path / "min.npy"
    input_max = tmp_path / "max.npy"
    np.save(input_min, np.zeros((30, 1, 1), dtype=np.float32))
    np.save(input_max, np.ones((30, 1, 1), dtype=np.float32))
    monkeypatch.setenv("FENGQ_MIN_PATH", str(input_min))
    monkeypatch.setenv("FENGQ_MAX_PATH", str(input_max))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("DISTRIBUTED_MODE", "none")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("PRECISION", "fp32")
    monkeypatch.setenv("BASE_CHANNELS", "2")
    monkeypatch.setenv("PRCP_NUM_WORKERS", "0")
    monkeypatch.delenv("EVENT_LOSS_WEIGHT", raising=False)
    monkeypatch.delenv("EVENT_LOSS_BETA", raising=False)
    monkeypatch.delenv("PHYSICAL_LOSS_WEIGHT", raising=False)

    cfg = LazyConfig.load(str(config_path))
    model = instantiate(cfg.model)

    assert isinstance(model, PrcpCorrectionUNet3D)
    assert model.physical_loss_weights == (1.0, 4.0, 4.0, 8.0)
    assert model.physical_loss_weight == pytest.approx(0.1)
    assert model.event_loss_weight == pytest.approx(0.05)
    assert model.event_loss_beta == pytest.approx(1.25)
    assert cfg.dataloader.train.num_workers == 0
    assert cfg.train.val_metric == "Selection/csi_mean6_error"


def test_model_buffers_and_dual_mode_outputs():
    input_min = np.zeros((3, 1, 1), dtype=np.float32)
    input_max = np.ones((3, 1, 1), dtype=np.float32)
    model = PrcpCorrectionUNet3D(
        in_channels=3,
        base_channels=2,
        input_min=input_min,
        input_max=input_max,
        device="cpu",
    )
    assert {"input_min", "input_max", "precipitation_scale"}.issubset(dict(model.named_buffers()))

    inputs = torch.rand(1, 2, 3, 8, 8)
    targets = torch.rand(1, 2, 1, 8, 8)
    model.train()
    losses = model({"inputs": inputs, "targets": targets, "valid_mask": torch.ones_like(targets, dtype=torch.bool)})
    assert set(losses) == {"loss_log_smooth_l1", "loss_physical_l1", "loss_physical_l2"}
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())

    model.eval()
    outputs = model({"inputs": inputs, "targets": targets, "valid_mask": torch.ones_like(targets, dtype=torch.bool)})
    assert outputs["output"].shape == targets.shape
    assert set(outputs["output_by_mode"]) == {"corrected", "baseline"}
    assert outputs["output_by_mode"]["baseline"].shape == targets.shape
    assert torch.allclose(outputs["output"], outputs["output_by_mode"]["baseline"])


def test_model_decodes_at_full_resolution_and_log_loss_keeps_negative_gradient():
    model = PrcpCorrectionUNet3D(
        in_channels=3,
        base_channels=2,
        input_min=np.zeros((3, 1, 1), dtype=np.float32),
        input_max=np.ones((3, 1, 1), dtype=np.float32),
        device="cpu",
    )
    decoded_shapes = []
    handle = model.net.fullres_decoder.register_forward_hook(lambda _module, _args, output: decoded_shapes.append(output.shape))
    inputs = torch.zeros(1, 2, 3, 8, 8)
    targets = torch.ones(1, 2, 1, 8, 8)
    with torch.no_grad():
        model.net.head.bias.fill_(-1.0)
    model.train()
    losses = model({"inputs": inputs, "targets": targets, "valid_mask": torch.ones_like(targets, dtype=torch.bool)})
    sum(losses.values()).backward()
    handle.remove()
    assert decoded_shapes == [torch.Size([1, 2, 2, 8, 8])]
    assert model.net.head.bias.grad is not None
    assert model.net.head.bias.grad.abs().sum() > 0


def test_masked_spatial_sum_reduction():
    output = torch.zeros(1, 2, 1, 2, 3)
    target = torch.ones_like(output)
    loss = PrcpCorrectionUNet3D._masked_spatial_sum((output - target).abs(), torch.ones_like(target, dtype=torch.bool))
    assert loss.item() == pytest.approx(6.0)


def test_masked_spatial_sum_rejects_empty_leads_and_masks_gradients():
    output = torch.zeros(1, 1, 1, 2, 3, requires_grad=True)
    target = torch.ones_like(output)
    valid = torch.ones_like(output, dtype=torch.bool)
    valid[..., 0, 0] = False
    target[..., 0, 0] = 9999
    loss = PrcpCorrectionUNet3D._masked_spatial_sum((output - target).abs(), valid)
    assert loss.item() == pytest.approx(6.0)
    loss.backward()
    assert output.grad[..., 0, 0].item() == 0
    assert torch.all(output.grad[valid] != 0)
    with pytest.raises(ValueError, match="valid observation"):
        PrcpCorrectionUNet3D._masked_spatial_sum((output - target).abs(), torch.zeros_like(valid))


def test_balanced_physical_losses_boundaries_cap_mask_and_gradient():
    model = PrcpCorrectionUNet3D(in_channels=1, base_channels=2, input_min=np.zeros((1, 1, 1)), input_max=np.ones((1, 1, 1)), device="cpu")
    target = torch.tensor([0.0, 9.9, 10.0, 24.9, 25.0, 49.9, 50.0, 200.0, 9999.0]).reshape(1, 1, 1, 1, 9)
    output = (target + 2).requires_grad_()
    with torch.no_grad():
        output[..., 0] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., -1] = False
    losses = model._balanced_physical_losses(output, target, valid)
    assert losses["loss_physical_l1"].item() == pytest.approx(0.1 * 9 * 63 / 32)
    assert losses["loss_physical_l2"].item() == pytest.approx(0.0075 * 9 * 125 / 32)
    sum(losses.values()).backward()
    assert output.grad.flatten().tolist() == pytest.approx([9 * w * (0.115 if i == 0 else 0.13) / 32 for i, w in enumerate((1, 1, 3, 3, 4, 4, 8, 8, 0))])
    with pytest.raises(ValueError, match="valid observation"):
        model._balanced_physical_losses(output, target, torch.zeros_like(valid))


def test_balanced_physical_losses_normalize_each_lead_and_mask():
    model = PrcpCorrectionUNet3D(in_channels=1, base_channels=2, input_min=np.zeros((1, 1, 1)), input_max=np.ones((1, 1, 1)), device="cpu")
    target = torch.tensor([0.0, 10.0, 0.0, 50.0]).reshape(1, 2, 1, 1, 2)
    output = target + torch.tensor([1.0, 3.0])
    valid = torch.ones_like(target, dtype=torch.bool)
    losses = model._balanced_physical_losses(output, target, valid)
    assert losses["loss_physical_l1"].item() == pytest.approx(0.1 * 95 / 18)
    assert losses["loss_physical_l2"].item() == pytest.approx(0.0075 * 136 / 9)
    model.physical_loss_weights = (1.0, 1.0, 1.0, 1.0)
    for masked in (False, True):
        valid[..., -1] = not masked
        losses = model._balanced_physical_losses(output, target, valid)
        assert torch.equal(losses["loss_physical_l1"], 0.1 * model._masked_spatial_sum((output - target).abs(), valid))
        assert torch.equal(losses["loss_physical_l2"], 0.0075 * model._masked_spatial_sum((output - target).square(), valid))


def test_balanced_mse_strengthens_large_errors_in_both_directions():
    model = PrcpCorrectionUNet3D(in_channels=1, base_channels=2, input_min=np.zeros((1, 1, 1)), input_max=np.ones((1, 1, 1)), device="cpu")
    target = torch.full((1, 1, 1, 1, 4), 25.0)
    output = (target + torch.tensor([-20.0, -10.0, 1.0, 5.0])).requires_grad_()
    losses = model._balanced_physical_losses(output, target, torch.ones_like(target, dtype=torch.bool))
    sum(losses.values()).backward()
    assert output.grad.flatten().tolist() == pytest.approx([-0.4, -0.25, 0.115, 0.175])


def test_training_forward_uses_balanced_physical_losses_without_changing_checkpoint_keys():
    model = PrcpCorrectionUNet3D(in_channels=1, base_channels=2, input_min=np.zeros((1, 1, 1)), input_max=np.ones((1, 1, 1)), device="cpu")
    target = torch.tensor([0.0, 10.0, 25.0, 50.0]).repeat(16).reshape(1, 1, 1, 8, 8)
    losses = model({"inputs": torch.zeros_like(target), "targets": target, "valid_mask": torch.ones_like(target, dtype=torch.bool)})
    assert losses["loss_physical_l1"].item() == pytest.approx(0.1 * 64 * (3 * 10 + 4 * 25 + 8 * 50) / 16)
    assert losses["loss_physical_l2"].item() == pytest.approx(0.0075 * 64 * (3 * 10**2 + 4 * 25**2 + 8 * 50**2) / 16)
    expected_log = torch.nn.functional.smooth_l1_loss(torch.zeros_like(target), torch.log1p(target), reduction="sum")
    assert losses["loss_log_smooth_l1"].item() == pytest.approx(expected_log.item())
    assert set(model.state_dict()) == {"input_min", "input_max", "precipitation_scale"} | {"net." + name for name in model.net.state_dict()}
    model.eval()
    assert model({"inputs": torch.zeros_like(target)})["output"].shape == target.shape


@pytest.mark.parametrize("thresholds, weights", [((10, 10), (1, 2, 4)), ((float("nan"),), (1, 2)), ((10,), (1,)), ((10,), (2, 1)), ((10,), (1, float("inf")))])
def test_balanced_mae_rejects_invalid_weight_configuration(thresholds, weights):
    with pytest.raises(ValueError, match="physical_loss_"):
        PrcpCorrectionUNet3D(
            in_channels=1,
            base_channels=2,
            input_min=np.zeros((1, 1, 1)),
            input_max=np.ones((1, 1, 1)),
            device="cpu",
            physical_loss_thresholds=thresholds,
            physical_loss_weights=weights,
        )


def test_masked_rmse_is_independent_of_batch_size():
    target = torch.zeros(2, 2, 1, 2, 3)
    corrected = torch.stack([torch.full_like(target[0], value) for value in (2, 4)])
    baseline = corrected + 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, ..., 0, 0] = False
    corrected[0, ..., 0, 0] = 9999
    results = []
    for batch_size in (1, 2):
        metric = PrcpMaskedRMSE(test_names=["tp"], test_times=[6, 12])
        metric.reset()
        for start in range(0, 2, batch_size):
            part = slice(start, start + batch_size)
            metric.process({}, {"target": target[part], "weight": valid[part], "output_by_mode": {"corrected": corrected[part], "baseline": baseline[part]}})
        results.append(metric.evaluate())
    assert results[0] == results[1]
    assert results[0]["RMSE"]["corrected/tp"] == pytest.approx([3.0, 3.0])
    assert results[0]["RMSE"]["baseline/tp"] == pytest.approx([4.0, 4.0])


def _csi_example():
    target = torch.tensor([[10, 25, 0, 0, 9999, 50, 10, 0, 0, 0], [10, 10, 10, 10, 0, 10, 10, 10, 10, 0]], dtype=torch.float32).reshape(2, 2, 1, 1, 5)
    prediction = torch.tensor([[10, 0, 10, 0, float("nan"), 0, 10, 0, 0, 0], [10] * 10], dtype=torch.float32).reshape_as(target)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0, 0, 0, 4] = False
    return target, prediction, valid


def test_csi_pools_counts_before_dividing_and_resets():
    target, prediction, valid = _csi_example()
    results = []
    for batch_size in (1, 2):
        metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6, 12])
        metric.reset()
        for start in range(0, 2, batch_size):
            part = slice(start, start + batch_size)
            metric.process({}, {"target": target[part], "weight": valid[part], "output_by_mode": {"corrected": prediction[part], "baseline": target[part]}})
        results.append(metric.evaluate())
        assert metric.counts["corrected"][metric.thresholds.index(10)].tolist() == [10, 2, 3]
        assert results[-1]["CSI10"]["corrected/tp"] == pytest.approx([10 / 15])
        assert results[-1]["CSI25"]["corrected/tp"] == [0.0]
        assert results[-1]["CSI50"]["baseline/tp"] == [1.0]
        assert results[-1]["CSImean6"]["corrected/tp"] == pytest.approx([4 / 9])
        assert results[-1]["CSImean6"]["baseline/tp"] == [1.0]
        assert results[-1]["Selection"] == {"csi_mean6_error": pytest.approx([5 / 9])}
        metric.reset()
        assert metric.counts == {}
    assert results[0] == results[1]


def test_csi_gathers_unequal_rank_counts_including_empty_rank(monkeypatch):
    target, prediction, valid = _csi_example()
    metrics = []
    for part in (slice(0, 1), slice(1, 2)):
        metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6, 12])
        metric.reset()
        metric.process({}, {"target": target[part], "weight": valid[part], "output_by_mode": {"corrected": prediction[part]}})
        metrics.append(metric)
    metric, other_rank = metrics
    local_counts = metric.counts["corrected"].copy()
    monkeypatch.setattr(metric_module.comm, "gather", lambda local, **kw: [local, other_rank.counts, {}])
    result = metric.evaluate()
    assert result["CSI10"]["corrected/tp"] == pytest.approx([10 / 15])
    assert metric.evaluate() == result
    assert np.array_equal(metric.counts["corrected"], local_counts)
    monkeypatch.setattr(metric_module.comm, "is_main_process", lambda: False)
    assert metric.evaluate() is None


def test_csi_empty_union_is_unavailable_and_valid_nan_is_rejected():
    target = torch.zeros(1, 1, 1, 1, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6])
    metric.reset()
    outputs = {"target": target, "weight": valid, "output_by_mode": {"corrected": target.clone()}}
    metric.process({}, outputs)
    result = metric.evaluate()
    assert np.isnan(result["CSI10"]["corrected/tp"][0])
    assert np.isnan(result["CSImean6"]["corrected/tp"][0])
    assert np.isnan(result["Selection"]["csi_mean6_error"][0])
    outputs["output_by_mode"]["corrected"][..., 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        metric.process({}, outputs)
    outputs["weight"] = valid.float()
    with pytest.raises(ValueError, match="boolean"):
        metric.process({}, outputs)


def test_csi_selection_works_with_native_min_checkpointer():
    from contextlib import nullcontext

    from xmetai.core.hooks import BestCheckpointer, flatten_results_dict

    saved = []
    hook = BestCheckpointer(
        eval_period=1, checkpointer=SimpleNamespace(save=lambda name, **kw: saved.append(kw["iteration"])), val_metric="Selection/csi_mean6_error", mode="min"
    )
    for iteration, score in enumerate((float("nan"), 0.2, 0.4, 0.3, 0.4)):
        flattened = flatten_results_dict({"Selection": {"csi_mean6_error": [1.0 - score]}})
        hook.trainer = SimpleNamespace(
            storage=SimpleNamespace(latest=lambda: {key: (value, iteration) for key, value in flattened.items()}), model_ema=nullcontext
        )
        hook._best_checking()
    assert saved == [1, 2]
    assert hook.best_metric == pytest.approx(0.6)


@pytest.mark.parametrize(
    "input_min,input_max,error",
    [
        (np.zeros((3, 1)), np.ones((3, 1)), "shape"),
        (np.zeros((3, 1, 1)), np.zeros((3, 1, 1)), "must exceed"),
        (np.full((3, 1, 1), np.nan), np.ones((3, 1, 1)), "NaN or Inf"),
    ],
)
def test_model_rejects_invalid_scales(input_min, input_max, error):
    with pytest.raises(ValueError, match=error):
        PrcpCorrectionUNet3D(
            in_channels=3,
            base_channels=2,
            input_min=input_min,
            input_max=input_max,
            device="cpu",
        )


def test_csi_six_threshold_boundaries_and_observation_mask():
    target = torch.tensor([0.099, 0.1, 0.999, 1, 4.999, 5, 9.999, 10, 24.999, 25, 49.999, 50, 9999]).reshape(1, 1, 1, 1, -1)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., -1] = False
    metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6])
    metric.reset()
    metric.process({}, {"target": target, "weight": valid, "output_by_mode": {"corrected": target.clone()}})
    assert metric.thresholds == (0.1, 1, 5, 10, 25, 50)
    assert metric.counts["corrected"][:, 0].tolist() == [11, 9, 7, 5, 3, 1]
    assert not metric.counts["corrected"][:, 1:].any()
    result = metric.evaluate()
    assert result["CSImean6"]["corrected/tp"] == [1.0]
    assert result["Selection"]["csi_mean6_error"] == [0.0]


def test_csi_mean6_is_macro_average_and_requires_every_threshold():
    metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6])
    metric.reset()
    metric.counts["corrected"] = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 1], [2, 3, 3], [1, 4, 5], [1, 9, 10]], dtype=np.int64)
    result = metric.evaluate()
    expected = np.mean([1, 0.5, 0.5, 0.25, 0.1, 0.05])
    assert result["CSImean6"]["corrected/tp"] == pytest.approx([expected])
    assert result["Selection"]["csi_mean6_error"] == pytest.approx([1 - expected])
    assert expected != pytest.approx(metric.counts["corrected"][:, 0].sum() / metric.counts["corrected"].sum())
    metric.counts["corrected"][-1] = 0
    result = metric.evaluate()
    assert np.isnan(result["CSImean6"]["corrected/tp"][0])
    assert np.isfinite(result["CSI10"]["corrected/tp"][0])


def _event_model(**overrides):
    kwargs = dict(
        in_channels=1,
        base_channels=2,
        input_min=np.zeros((1, 1, 1)),
        input_max=np.ones((1, 1, 1)),
        device="cpu",
        event_loss_weight=0.05,
    )
    kwargs.update(overrides)
    return PrcpCorrectionUNet3D(**kwargs)


def test_joint_event_losses_match_scalar_reference_and_mask():
    model = _event_model(event_loss_alpha=0.8, event_loss_beta=1.3)
    target = torch.tensor([0, 5, 10, 25, 50, 80, 0, 8, 15, 35, 65, 9999.0]).reshape(2, 2, 1, 1, 3)
    values = torch.tensor([0.2, 3, 12, 17, 38, 90, 1, 10, 4, 20, 70, 9999.0]).reshape_as(target)
    log_prediction = torch.log1p(values).requires_grad_()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid.flatten()[-1] = False
    with torch.no_grad():
        log_prediction.flatten()[-1] = float("nan")
    losses = model._joint_event_losses(log_prediction, target, valid)
    assert set(losses) == {f"loss_event_{threshold:g}mm" for threshold in (5, 10, 25, 50)}
    z = log_prediction.detach().numpy()[valid.numpy()].astype(np.float64)
    truth = target.numpy()[valid.numpy()]
    for threshold in model.event_thresholds:
        probability = 1 / (1 + np.exp(-(z - np.log1p(threshold)) / 0.25))
        observed = truth >= threshold
        hits = probability[observed].sum()
        false_alarms = probability[~observed].sum()
        misses = (1 - probability[observed]).sum()
        expected = 0.05 * 3 / 4 * (1 - hits / (hits + 0.8 * false_alarms + 1.3 * misses))
        assert losses[f"loss_event_{threshold:g}mm"].item() == pytest.approx(expected, rel=2e-6)
    sum(losses.values()).backward()
    assert torch.isfinite(log_prediction.grad).all()
    assert log_prediction.grad.flatten()[-1] == 0


@pytest.mark.parametrize("threshold", [5, 10, 25, 50])
def test_joint_event_gradients_recover_misses_and_suppress_false_alarms(threshold):
    model = _event_model()
    target = torch.tensor([float(threshold), 0]).reshape(1, 1, 1, 1, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    z = torch.full_like(target, float(np.log1p(threshold)), requires_grad=True)
    loss = model._joint_event_losses(z, target, valid)[f"loss_event_{threshold}mm"]
    loss.backward()
    assert z.grad.flatten()[0] < 0
    assert z.grad.flatten()[1] > 0
    with torch.no_grad():
        improved = z - 0.1 * z.grad.sign()
    after = model._joint_event_losses(improved, target, valid)[f"loss_event_{threshold}mm"]
    assert after < loss


@pytest.mark.parametrize("threshold", [5, 10, 25, 50])
@pytest.mark.parametrize("candidate_beta", [1.25, 1.5])
def test_joint_event_beta_increases_relative_miss_gradient(threshold, candidate_beta):
    target = torch.tensor([float(threshold), 0]).reshape(1, 1, 1, 1, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    gradients = []
    for beta in (1.0, candidate_beta):
        model = _event_model(event_loss_beta=beta)
        z = torch.full_like(target, float(np.log1p(threshold)), requires_grad=True)
        model._joint_event_losses(z, target, valid)[f"loss_event_{threshold}mm"].backward()
        gradient = z.grad.flatten()
        assert gradient[0] < 0 and gradient[1] > 0
        gradients.append(gradient.clone())
    original, recall_weighted = gradients
    assert recall_weighted[0].abs() / recall_weighted[1] > original[0].abs() / original[1]


def test_joint_event_empty_observations_keep_false_alarm_gradients():
    model = _event_model()
    target = torch.zeros(2, 2, 1, 2, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid.flatten()[-1] = False
    z = torch.full_like(target, float(np.log1p(60)), requires_grad=True)
    losses = model._joint_event_losses(z, target, valid)
    for threshold in model.event_thresholds:
        probability = 1 / (1 + np.exp(-(np.log1p(60) - np.log1p(threshold)) / 0.25))
        assert losses[f"loss_event_{threshold:g}mm"].item() == pytest.approx(0.05 * probability, rel=2e-6)
    sum(losses.values()).backward()
    assert torch.isfinite(z.grad).all()
    assert torch.all(z.grad[valid] > 0)
    assert z.grad.flatten()[-1] == 0


def test_joint_event_empty_high_threshold_still_has_one_quarter_weight():
    model = _event_model()
    target = torch.tensor([0, 5, 10, 25.0]).reshape(1, 1, 1, 1, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    z = torch.log1p(target + 1)
    losses = model._joint_event_losses(z, target, valid)
    p50 = torch.sigmoid((z - np.log1p(50)) / 0.25)
    assert len(losses) == 4
    assert losses["loss_event_50mm"].item() == pytest.approx(0.05 * p50.mean().item())


def test_joint_event_pooling_is_invariant_to_repeated_batch_and_weight_is_linear():
    model = _event_model()
    target = torch.tensor([0, 5, 10, 25, 50, 80.0]).reshape(1, 2, 1, 1, 3)
    valid = torch.ones_like(target, dtype=torch.bool)
    z = torch.log1p(target * 0.8 + 1)
    loss = model._joint_event_losses(z, target, valid)
    repeated = model._joint_event_losses(z.repeat(3, 1, 1, 1, 1), target.repeat(3, 1, 1, 1, 1), valid.repeat(3, 1, 1, 1, 1))
    model.event_loss_weight *= 2
    doubled = model._joint_event_losses(z, target, valid)
    for key in loss:
        assert repeated[key].item() == pytest.approx(loss[key].item())
        assert doubled[key].item() == pytest.approx(2 * loss[key].item())


def test_joint_event_can_be_disabled_without_changing_base_losses_or_checkpoint():
    baseline = _event_model(event_loss_weight=0)
    enabled = _event_model()
    enabled.load_state_dict(baseline.state_dict(), strict=True)
    inputs = torch.rand(1, 2, 1, 8, 8)
    target = torch.tensor([0, 5, 10, 25, 50, 80, 0, 1.0]).repeat(16).reshape_as(inputs)
    data = {"inputs": inputs, "targets": target, "valid_mask": torch.ones_like(target, dtype=torch.bool)}
    original = baseline(data)
    augmented = enabled(data)
    assert len(original) == 3 and len(augmented) == 7
    for key in original:
        assert torch.equal(original[key], augmented[key])
    event = {key: value for key, value in augmented.items() if key.startswith("loss_event_")}
    sum(event.values()).backward()
    assert torch.isfinite(enabled.net.head.bias.grad).all() and enabled.net.head.bias.grad.abs().sum() > 0
    enabled.event_loss_weight = 0
    disabled = enabled(data)
    assert set(disabled) == set(original)
    assert all(torch.equal(disabled[key], original[key]) for key in original)
    assert set(enabled.state_dict()) == set(baseline.state_dict())
    baseline.eval()
    enabled.eval()
    assert torch.equal(baseline({"inputs": inputs})["output"], enabled({"inputs": inputs})["output"])


@pytest.mark.parametrize(
    "name,value",
    [
        ("event_loss_weight", -1),
        ("event_loss_weight", float("nan")),
        ("event_loss_temperature", 0),
        ("event_loss_temperature", float("inf")),
        ("event_loss_alpha", 0),
        ("event_loss_alpha", float("nan")),
        ("event_loss_beta", -1),
        ("event_loss_beta", float("inf")),
    ],
)
def test_joint_event_rejects_invalid_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        _event_model(**{name: value})


@pytest.mark.parametrize("statistic", ["input_min", "input_max"])
@pytest.mark.parametrize("loader", ["module", "native", "parent"])
def test_checkpoint_statistics_mismatch_is_rejected_before_loading(statistic, loader):
    from xmetai.core.checkpoint import FSDPCheckpointer

    model = _event_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint = {key: value.clone() for key, value in before.items()}
    checkpoint[statistic] += 0.25
    checkpoint["net.head.bias"] += 7
    with pytest.raises(ValueError, match=statistic + ".*normalization"):
        if loader == "native":
            FSDPCheckpointer(model)._load_model({"model": {"module." + key: value for key, value in checkpoint.items()}})
        elif loader == "parent":
            torch.nn.ModuleDict({"correction": model}).load_state_dict({"correction." + key: value for key, value in checkpoint.items()})
        else:
            model.load_state_dict(checkpoint, strict=False)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_native_checkpoint_accepts_matching_statistics_and_preserves_state_keys():
    from xmetai.core.checkpoint import FSDPCheckpointer

    source = _event_model()
    model = _event_model()
    with torch.no_grad():
        source.net.head.bias.fill_(0.75)
    checkpoint = source.state_dict()
    incompatible = FSDPCheckpointer(model)._load_model({"model": checkpoint})
    assert not incompatible.missing_keys and not incompatible.unexpected_keys and not incompatible.incorrect_shapes
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in checkpoint.items())


def test_bf16_keeps_fp32_targets_for_losses_and_csi():
    class ZeroResidual(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.dtype == torch.bfloat16
            return torch.zeros_like(inputs)

    model = _event_model(precision="bf16")
    model.net = ZeroResidual()
    target = torch.tensor([9.99, 24.99, 49.9, 50.0]).reshape(1, 1, 1, 1, 4)
    valid = torch.ones_like(target, dtype=torch.bool)
    data = {"inputs": torch.zeros_like(target), "targets": target, "valid_mask": valid}
    losses = model(data)
    reference = _event_model()
    expected = {
        "loss_log_smooth_l1": reference._smooth_l1_spatial_sum(torch.zeros_like(target), torch.log1p(target), valid),
        **reference._balanced_physical_losses(torch.zeros_like(target), target, valid),
        **reference._joint_event_losses(torch.zeros_like(target), target, valid),
    }
    assert losses.keys() == expected.keys()
    for key in losses:
        torch.testing.assert_close(losses[key], expected[key], rtol=0, atol=0)
    model.eval()
    outputs = model(data)
    assert outputs["target"].dtype == torch.float32
    assert torch.equal(outputs["target"], target)
    metric = PrcpMaskedCSI(test_names=["tp"], test_times=[6])
    metric.reset()
    metric.process(data, outputs)
    assert metric.counts["corrected"][:, 1].tolist() == [4, 4, 4, 3, 2, 1]


@pytest.fixture
def partial_observation_stores(tmp_path):
    from xmetai.data.weather.prcpcorrect_dataset import _DTIMES, _FENGQ_CHANNELS, _FENGQ_SCHEMA, _LATITUDES, _LONGITUDES

    fengq_path = tmp_path / "fengq.zarr"
    cmpas_path = tmp_path / "cmpas.zarr"
    zarr_v3 = int(zarr.__version__.split(".", 1)[0]) >= 3
    store_options = {"zarr_format": 2} if zarr_v3 else {"zarr_version": 2}
    fengq = zarr.open_group(str(fengq_path), mode="w", **store_options)
    cmpas = zarr.open_group(str(cmpas_path), mode="w", **store_options)

    def create_array(group, name, **kwargs):
        create = group.create_array if zarr_v3 else group.create_dataset
        return create(name, **kwargs)

    times = np.array(["2024-07-02T08", "2024-07-10T08", "2024-07-20T08"], dtype="datetime64[ns]")
    valid_times = times[:, None] + _DTIMES.astype("timedelta64[h]")
    fengq.attrs.update(schema=_FENGQ_SCHEMA, normalization="none", time_zone="Beijing time")
    cmpas.attrs.update(time_zone="Beijing time")
    inputs = create_array(fengq, "input", shape=(3, 12, 30, 64, 40), chunks=(1, 12, 30, 64, 40), dtype="f4", fill_value=0)
    inputs.attrs["_ARRAY_DIMENSIONS"] = ["time", "dtime", "pl", "lat", "lon"]
    for name, values in {"time": times, "valid_time": valid_times, "dtime": _DTIMES, "pl": np.array(_FENGQ_CHANNELS)}.items():
        create_array(fengq, name, data=values)
    for group in (fengq, cmpas):
        create_array(group, "lat", data=_LATITUDES)
        create_array(group, "lon", data=_LONGITUDES)
    create_array(cmpas, "valid_time", data=valid_times[:2].reshape(-1))
    create_array(cmpas, "channel", data=np.array(["tp"]))
    targets = create_array(cmpas, "data", shape=(24, 1, 64, 40), chunks=(12, 1, 64, 40), dtype="f4", fill_value=0)
    targets[:12] = 2
    targets[12:] = 10
    mask = create_array(cmpas, "valid_mask", shape=targets.shape, chunks=targets.chunks, dtype="bool", fill_value=True)
    for array in (targets, mask):
        array.attrs["_ARRAY_DIMENSIONS"] = ["valid_time", "channel", "lat", "lon"]
    return dict(fengq_zarr_path=str(fengq_path), cmpas_zarr_path=str(cmpas_path), input_min=np.zeros((30, 1, 1)), input_max=np.ones((30, 1, 1)))


def test_dataset_checks_coverage_only_for_selected_time_range(partial_observation_stores):
    dataset = PrcpCorrectionUNet3DDataset(**partial_observation_stores, start_time="2024-07-10 08:00:00", end_time="2024-07-10 08:00:00")
    assert dataset.sample_indices == [1]
    assert torch.all(dataset[0]["targets"] == 10)
    assert dataset[0]["valid_mask"].all()


def test_dataset_csv_preserves_selected_target_order(partial_observation_stores, tmp_path):
    split = tmp_path / "split.csv"
    split.write_text("2024-07-10 08:00:00\n2024-07-02 08:00:00\n", encoding="utf-8")
    dataset = PrcpCorrectionUNet3DDataset(**partial_observation_stores, split_csv=str(split))
    assert dataset.sample_indices == [1, 0]
    assert torch.all(dataset[0]["targets"] == 10)
    assert torch.all(dataset[1]["targets"] == 2)


def test_dataset_rejects_missing_selected_targets(partial_observation_stores):
    with pytest.raises(ValueError, match="CMPAS is missing target time 2024-07-20 14:00:00"):
        PrcpCorrectionUNet3DDataset(**partial_observation_stores, start_time="2024-07-20 08:00:00")


def test_dataset_inference_does_not_require_observation_coverage(partial_observation_stores):
    partial_observation_stores["cmpas_zarr_path"] = None
    dataset = PrcpCorrectionUNet3DDataset(**partial_observation_stores, start_time="2024-07-20 08:00:00")
    assert dataset.sample_indices == [2]
    assert dataset.target_indices is None
    assert set(dataset[0]) == {"inputs", "times"}
