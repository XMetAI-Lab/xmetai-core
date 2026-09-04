"""Residual correction for frozen ONNX weather forecasts."""

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xmetai.layers import DarknetConvUNet2D
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

from .onnx_infer import OnnxInferModel

__all__ = ["CorrectionModel"]


@MODELS.register()
class CorrectionModel(OnnxInferModel):
    """Correct selected channels from a frozen ONNX forecast with a Darknet U-Net.

    The ONNX model runs in physical space, while the corrector learns
    ``target - raw_forecast`` in the normalized training space. Corrected
    forecasts are post-processing outputs and are not fed back into the ONNX
    rollout.
    """

    def __init__(
        self,
        correct_chans: Sequence[int],
        backbone_kwargs: Mapping[str, Any],
        rollout_max: int = 0,
        rollout_alpha: float = 1.0,
        zero_init_output: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if self.in_chans != self.out_chans:
            raise ValueError(f"CorrectionModel requires in_chans == out_chans for autoregressive rollout, got {self.in_chans} and {self.out_chans}")
        if self.out_frames != 1:
            raise ValueError(f"CorrectionModel currently requires out_frames=1, got {self.out_frames}")
        if self.members <= 0:
            raise ValueError(f"members must be positive, got {self.members}")
        if len(self.test_frames) == 0 or any(frame <= 0 for frame in self.test_frames):
            raise ValueError(f"test_frames must contain positive forecast steps, got {self.test_frames}")
        if len(set(self.test_frames)) != len(self.test_frames):
            raise ValueError(f"test_frames contains duplicates: {self.test_frames}")
        if rollout_max < 0:
            raise ValueError(f"rollout_max must be non-negative, got {rollout_max}")
        if not 0.0 < rollout_alpha <= 1.0:
            raise ValueError(f"rollout_alpha must be in (0, 1], got {rollout_alpha}")
        if sum(self.step_range) > max(self.test_frames):
            raise ValueError(f"sum(step_range)={sum(self.step_range)} exceeds max(test_frames)={max(self.test_frames)}")

        self.correct_chans = [int(channel) for channel in correct_chans]
        if len(self.correct_chans) == 0:
            raise ValueError("correct_chans must contain at least one channel")
        if len(set(self.correct_chans)) != len(self.correct_chans):
            raise ValueError(f"correct_chans contains duplicates: {self.correct_chans}")
        if min(self.correct_chans) < 0 or max(self.correct_chans) >= self.out_chans:
            raise ValueError(f"correct_chans must be in [0, {self.out_chans - 1}], got {self.correct_chans}")
        if self.precip_loss_enable:
            missing_precip_chans = sorted(set(self.precip_indices) - set(self.correct_chans))
            if missing_precip_chans:
                raise ValueError(f"precipitation loss channels must be included in correct_chans, missing {missing_precip_chans}")

        options = dict(backbone_kwargs)
        base_channels = int(options.pop("base_channels"))
        ch_mults = tuple(int(mult) for mult in options.pop("ch_mults"))
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if len(ch_mults) < 2 or any(mult <= 0 for mult in ch_mults):
            raise ValueError(f"ch_mults must contain at least two positive values, got {ch_mults}")
        options.setdefault("layers_per_block", 2)
        options.setdefault("time_dim", base_channels * 4)
        options.setdefault("dropout", 0.0)

        self.corrector = DarknetConvUNet2D(
            in_channels=self.out_chans, out_channels=len(self.correct_chans), base_channels=base_channels, ch_mults=ch_mults, **options
        )
        self.corrector_multiple = 2 ** (len(ch_mults) - 1)
        self.rollout_max = rollout_max
        self.rollout_alpha = rollout_alpha

        if zero_init_output:
            nn.init.zeros_(self.corrector.out.weight)
            if self.corrector.out.bias is not None:
                nn.init.zeros_(self.corrector.out.bias)

    def predict_correction(self, raw_forecast: Tensor, lead: int) -> Tensor:
        """Predict normalized residuals for ``correct_chans`` at one lead."""
        model_input = raw_forecast[:, 0]
        height, width = model_input.shape[-2:]
        pad_h = (self.corrector_multiple - height % self.corrector_multiple) % self.corrector_multiple
        pad_w = (self.corrector_multiple - width % self.corrector_multiple) % self.corrector_multiple
        model_input = F.pad(model_input, (0, pad_w, 0, pad_h))
        timesteps = torch.full((model_input.shape[0],), float(lead), device=model_input.device, dtype=torch.float32)
        correction = self.corrector(model_input, timesteps, return_dict=False)[0]
        return correction[:, :, :height, :width]

    def merge_correction(self, raw_forecast: Tensor, correction: Tensor) -> Tensor:
        corrected = raw_forecast.clone()
        corrected[:, 0, self.correct_chans] = corrected[:, 0, self.correct_chans] + correction.to(corrected)
        return corrected

    def _raw_forecast(self, physical_input: Tensor, times: Any, step: int) -> tuple[Tensor, Tensor]:
        onnx_output = self._run_onnx_step(physical_input, step=step, times=times)
        raw_physical = onnx_output[:, -1:]
        raw_normalized = self.normalize(raw_physical)
        return raw_physical, raw_normalized

    def forward(self, data: dict[str, Any]) -> dict[str, Tensor]:
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        normalized_input = data["input"]
        targets = data["targets"]
        future_steps = targets.shape[1]

        loss_steps, rolling_weights = self.build_loss_steps_and_weights(future_steps, self.rollout_alpha, self.rollout_max)
        loss_steps = set(loss_steps)
        physical_input = self.inv_normalize(normalized_input).detach()
        loss_rec = targets.new_zeros((), dtype=torch.float32)
        precip_losses: dict[str, Tensor] = {}

        for step in range(future_steps):
            raw_physical, raw_normalized = self._raw_forecast(physical_input, data["times"], step=self.start_frames + step)
            if step in loss_steps:
                lead = self.start_frames + step + 1
                correction = self.predict_correction(raw_normalized, lead=lead)
                residual_target = targets[:, step, self.correct_chans].float() - raw_normalized[:, 0, self.correct_chans]
                loss_rec = loss_rec + rolling_weights[step] * charbonnier_loss(correction, residual_target)
                if self.precip_loss_enable:
                    corrected = self.merge_correction(raw_normalized, correction)
                    for name, value in self.compute_precip_losses(corrected, targets[:, step : step + 1]).items():
                        precip_losses[name] = precip_losses.get(name, 0) + rolling_weights[step] * value
            physical_input = torch.cat([physical_input, raw_physical], dim=1)[:, -self.in_frames :].contiguous()

        return {"loss_rec": loss_rec, **precip_losses}

    @torch.no_grad()
    def test_cascade(self, data: dict[str, Any]) -> Tensor:
        normalized_input = data["input"]
        total_step = max(self.test_frames)

        save_steps = {int(step): index for index, step in enumerate(np.asarray(self.test_frames) // self.out_frames)}
        height, width = normalized_input.shape[-2:]
        outputs = torch.zeros(
            normalized_input.shape[0], len(save_steps), len(self.test_chans), height, width, device=normalized_input.device, dtype=torch.float32
        )

        for _ in range(self.members):
            physical_input = self.inv_normalize(normalized_input)
            for step in range(total_step):
                raw_physical, raw_normalized = self._raw_forecast(physical_input, data["times"], step=self.start_frames + step)
                forecast_step = step + 1
                if forecast_step in save_steps:
                    lead = self.start_frames + forecast_step
                    correction = self.predict_correction(raw_normalized, lead=lead)
                    outputs[:, save_steps[forecast_step]] += self.merge_correction(raw_normalized, correction).float()[:, 0, self.test_chans]
                physical_input = torch.cat([physical_input, raw_physical], dim=1)[:, -self.in_frames :].contiguous()

        return (outputs / self.members).to(normalized_input)
