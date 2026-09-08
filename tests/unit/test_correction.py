import ctypes
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from xmetai import MODELS
from xmetai.models.correction import CorrectionModel
from xmetai.models.onnx_infer import OnnxInferModel


def _build_model(monkeypatch, correct_chans=(1, 3), test_frames=(1, 2), test_chans=None, in_names=None, logid=(), **model_kwargs):
    def fake_load_model(self, onnx_path):
        self.session = object()

    monkeypatch.setattr(OnnxInferModel, "load_model", fake_load_model)
    channels = 4
    in_names = [f"c{index}" for index in range(channels)] if in_names is None else in_names
    test_chans = list(range(channels)) if test_chans is None else list(test_chans)
    return CorrectionModel(
        onnx_path="base.onnx",
        correct_chans=correct_chans,
        backbone_kwargs=dict(base_channels=8, ch_mults=(1, 2), layers_per_block=1, time_dim=16, dropout=0.0),
        in_chans=channels,
        out_chans=channels,
        in_frames=2,
        out_frames=1,
        members=1,
        step_range=[max(test_frames)],
        test_frames=list(test_frames),
        test_chans=test_chans,
        test_names=[in_names[index] for index in test_chans],
        in_names=in_names,
        logid=list(logid),
        buffers=dict(
            mean=np.zeros((channels, 1, 1), dtype=np.float32), std=np.ones((channels, 1, 1), dtype=np.float32), weight=np.ones((1, 8, 1), dtype=np.float32)
        ),
        precision="fp32",
        device="cpu",
        **model_kwargs,
    )


def _install_incrementing_onnx(model, calls=None):
    def fake_run_onnx_step(self, physical_input, step, times=None):
        if calls is not None:
            calls.append(physical_input[:, -1:].clone())
        prediction = physical_input[:, -1:] + 1.0
        return torch.cat([physical_input[:, -1:], prediction], dim=1)

    model._run_onnx_step = types.MethodType(fake_run_onnx_step, model)


def _batch(future_steps=1):
    inputs = torch.zeros(1, 2, 4, 8, 8)
    targets = torch.ones(1, future_steps, 4, 8, 8)
    times = torch.tensor([np.datetime64("2026-01-01T00:00:00", "ns").astype(np.int64)])
    return dict(inputs=inputs, targets=targets, times=times)


def test_model_is_registered():
    assert MODELS.get("CorrectionModel") is CorrectionModel


def test_zero_initialized_corrector_preserves_raw_forecast(monkeypatch):
    model = _build_model(monkeypatch)
    raw_forecast = torch.randn(1, 1, 4, 7, 9)

    correction = model.predict_correction(raw_forecast, lead=1)
    corrected = model.merge_correction(raw_forecast, correction)

    assert correction.shape == (1, 2, 7, 9)
    assert torch.count_nonzero(correction) == 0
    torch.testing.assert_close(corrected, raw_forecast)


@pytest.mark.parametrize("correct_chans", [(), (1, 1), (-1,), (4,)])
def test_rejects_invalid_correct_channels(monkeypatch, correct_chans):
    with pytest.raises(ValueError, match="correct_chans"):
        _build_model(monkeypatch, correct_chans=correct_chans)


def test_training_learns_target_minus_raw_forecast(monkeypatch):
    model = _build_model(monkeypatch, test_frames=(1,))
    _install_incrementing_onnx(model)
    data = _batch()
    data["targets"][:, :, model.correct_chans] = 3.0
    model.train()

    loss = model(data)["loss_rec"]
    loss.backward()

    assert loss.item() == pytest.approx(2.0, rel=1e-5)
    assert model.corrector.out.weight.grad is not None
    assert torch.count_nonzero(model.corrector.out.weight.grad) > 0


def test_training_adds_precipitation_losses_on_corrected_forecast(monkeypatch):
    model = _build_model(
        monkeypatch,
        correct_chans=(1,),
        test_frames=(1,),
        in_names=["c0", "tp", "c2", "c3"],
        logid=(1,),
        precip_loss_enable=True,
        precip_loss_coeffs={"occurrence": 0.01, "distribution": 0.005},
        precip_names=("tp",),
        precip_threshold_mm=0.1,
        precip_mm_scale=21.6,
    )
    _install_incrementing_onnx(model)
    model.train()

    losses = model(_batch())

    assert set(losses) == {"loss_rec", "precip_loss_occurrence", "precip_loss_distribution"}
    assert all(torch.isfinite(loss) for loss in losses.values())


def test_precipitation_loss_requires_correcting_precipitation_channel(monkeypatch):
    with pytest.raises(ValueError, match="precipitation loss channels"):
        _build_model(
            monkeypatch,
            correct_chans=(0,),
            in_names=["c0", "tp", "c2", "c3"],
            precip_loss_enable=True,
            precip_loss_coeffs={"occurrence": 0.01},
            precip_names=("tp",),
        )


def test_onnx_step_runs_multi_batch_one_sample_at_a_time(monkeypatch):
    class FakeBinding:
        def __init__(self):
            self.inputs = {}
            self.output = None

        def bind_input(self, name, shape, buffer_ptr, **kwargs):
            self.inputs[name] = (tuple(shape), buffer_ptr)

        def bind_output(self, name, shape, buffer_ptr, **kwargs):
            self.output = (tuple(shape), buffer_ptr)

    class SingleBatchSession:
        def __init__(self):
            self.batch_shapes = []

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=[1, 1, 4, 8, 8])]

        def io_binding(self):
            return FakeBinding()

        def run_with_iobinding(self, binding):
            input_shape, input_ptr = binding.inputs["input"]
            output_shape, output_ptr = binding.output
            self.batch_shapes.append(input_shape)
            source = np.ctypeslib.as_array((ctypes.c_float * int(np.prod(input_shape))).from_address(input_ptr)).reshape(input_shape)
            destination = np.ctypeslib.as_array((ctypes.c_float * int(np.prod(output_shape))).from_address(output_ptr)).reshape(output_shape)
            destination[:] = source[:, -1:] + 1.0

    model = _build_model(monkeypatch, test_frames=(1,))
    model.session = SingleBatchSession()
    physical_input = torch.stack([torch.full((2, 4, 8, 8), float(index)) for index in range(3)])
    times = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T06:00:00", "2026-01-01T12:00:00"])

    output = model._run_onnx_step(physical_input, step=0, times=times)

    assert model.session.batch_shapes == [(1, 2, 4, 8, 8)] * 3
    torch.testing.assert_close(output[:, 0, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_correction_is_postprocessing_and_not_onnx_feedback(monkeypatch):
    model = _build_model(monkeypatch)
    calls = []
    _install_incrementing_onnx(model, calls=calls)

    class UnitCorrection(nn.Module):
        def forward(self, x, timesteps, return_dict=False):
            return (x.new_ones(x.shape[0], len(model.correct_chans), *x.shape[-2:]),)

    model.corrector = UnitCorrection()
    model.eval()
    processed = model.process_data(_batch(future_steps=2))

    output = model.test_cascade(processed)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.zeros_like(calls[0]))
    torch.testing.assert_close(calls[1], torch.ones_like(calls[1]))
    torch.testing.assert_close(output[:, 0, model.correct_chans], torch.full((1, 2, 8, 8), 2.0))
    torch.testing.assert_close(output[:, 1, model.correct_chans], torch.full((1, 2, 8, 8), 3.0))
    torch.testing.assert_close(output[:, 0, [0, 2]], torch.ones(1, 2, 8, 8))
    torch.testing.assert_close(output[:, 1, [0, 2]], torch.full((1, 2, 8, 8), 2.0))


def test_cascade_only_stores_test_channels(monkeypatch):
    model = _build_model(monkeypatch, test_chans=(1, 3))
    _install_incrementing_onnx(model)
    model.eval()
    processed = model.process_data(_batch(future_steps=2))

    output = model.test_cascade(processed)

    assert output.shape == (1, 2, 2, 8, 8)
    torch.testing.assert_close(output[:, 0], torch.ones(1, 2, 8, 8))
    torch.testing.assert_close(output[:, 1], torch.full((1, 2, 8, 8), 2.0))
