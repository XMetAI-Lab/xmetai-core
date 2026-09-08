"""Residual correction for frozen ONNX weather forecasts."""

from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xmetai.layers import DarknetConvUNet2D
from xmetai.loss import charbonnier_loss, precip_loss
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
        target_chans: Sequence[int] | None = None,
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
        target_chans = list(range(self.out_chans)) if target_chans is None else list(target_chans)
        self.correct_target_chans = [target_chans.index(channel) for channel in self.correct_chans]
        if self.precip_loss_enable:
            missing_precip_chans = sorted(set(self.precip_indices) - set(self.correct_chans))
            if missing_precip_chans:
                raise ValueError(f"precipitation loss channels must be included in correct_chans, missing {missing_precip_chans}")
            self.correct_precip_indices = tuple(self.correct_chans.index(channel) for channel in self.precip_indices)

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

    def predict_correction(self, raw_forecast: Tensor, lead: int | Tensor) -> Tensor:
        """Predict residuals with a one-based lead (one step is ``freq`` hours)."""
        model_input = raw_forecast[:, 0]
        height, width = model_input.shape[-2:]
        pad_h = (self.corrector_multiple - height % self.corrector_multiple) % self.corrector_multiple
        pad_w = (self.corrector_multiple - width % self.corrector_multiple) % self.corrector_multiple
        model_input = F.pad(model_input, (0, pad_w, 0, pad_h))
        lead_tensor = lead if isinstance(lead, Tensor) else torch.tensor(lead, device=model_input.device)
        lead_tensor = lead_tensor.to(device=model_input.device, dtype=model_input.dtype)
        timesteps = lead_tensor.expand(model_input.shape[0]) if lead_tensor.ndim == 0 else lead_tensor
        correction = self.corrector(model_input, timesteps, return_dict=False)[0]
        return correction[:, :, :height, :width]

    def merge_correction(self, raw_forecast: Tensor, correction: Tensor) -> Tensor:
        corrected = raw_forecast.clone()
        corrected[:, 0, self.correct_chans] = corrected[:, 0, self.correct_chans] + correction.to(corrected)
        return corrected

    @torch.no_grad()
    def _raw_forecast(self, physical_input: Tensor, times: Any, step: int) -> tuple[Tensor, Tensor]:
        onnx_output = self._run_onnx_step(physical_input, step=step, times=times)
        raw_physical = onnx_output[:, -1:]
        # normalize updates log channels in-place; preserve physical ONNX feedback.
        raw_normalized = self.normalize(raw_physical.clone())
        return raw_physical, raw_normalized

    def forward(
        self, data: dict[str, Any] | None, *, rollout_state: dict[str, Any] | None = None, rollout_segment_size: int | None = None
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Any], bool]:
        """Roll out frozen forecasts and supervise corrections at selected leads.

        With ExpandTrainer, segments are always one supervised lead, regardless
        of the supplied segment size. Frozen ONNX steps before that lead run
        without a graph. Targets stay on the host; only the current correction
        channels reach the GPU. ``rollout_max=0`` supervises every lead in the
        current training horizon; positive values limit the selected loss leads.
        """
        if not self.training:
            return self.test(data)

        if rollout_state is None:
            # The first segment receives data; continuation calls receive state.
            data = cast(dict[str, Any], data)
            targets = data["targets"][:, : data.get("future_frames", data["targets"].shape[1])]
            model_data = data.copy()
            del model_data["targets"]
            processed = self.process_data(model_data)
            physical_input = self.inv_normalize(processed["input"]).detach()
            for step in range(self.start_frames):
                raw_physical, _ = self._raw_forecast(physical_input, processed["times"], step=step)
                physical_input = torch.cat([physical_input[:, 1:], raw_physical], dim=1).contiguous()
            loss_steps, rolling_weights = self.build_loss_steps_and_weights(targets.shape[1], self.rollout_alpha, self.rollout_max)
            rollout_state = dict(
                targets=targets, physical_input=physical_input, times=processed["times"], loss_steps=loss_steps, rolling_weights=rolling_weights, step=0
            )

        physical_input = rollout_state["physical_input"]
        targets = rollout_state["targets"]
        loss_steps = rollout_state["loss_steps"]
        rolling_weights = rollout_state["rolling_weights"]
        first_step = rollout_state["step"]
        last_step = targets.shape[1]
        if rollout_segment_size is not None:
            # Segment size is fixed to 1: only rollout_max controls supervision.
            next_loss_step = next(step for step in loss_steps if step >= first_step)
            last_step = next_loss_step + 1

        loss_rec = physical_input.new_zeros((), dtype=torch.float32)
        precip_losses: dict[str, Tensor] = {}

        for step in range(first_step, last_step):
            raw_physical, raw_normalized = self._raw_forecast(physical_input, rollout_state["times"], step=self.start_frames + step)
            if step in loss_steps:
                lead = self.start_frames + step + 1
                correction = self.predict_correction(raw_normalized, lead=lead)
                target = targets[:, step : step + 1, self.correct_target_chans].to(device=physical_input.device, dtype=torch.float32, non_blocking=True)
                raw_selected = raw_normalized[:, :, self.correct_chans]
                residual_target = target[:, 0] - raw_selected[:, 0]
                loss_rec = loss_rec + rolling_weights[step] * charbonnier_loss(correction, residual_target)
                if self.precip_loss_enable:
                    corrected = raw_selected + correction[:, None].float()
                    for name, value in precip_loss(
                        corrected,
                        target,
                        self.correct_precip_indices,
                        self.precip_threshold,
                        coefficients=self.precip_loss_coeffs,
                        occurrence_scale=self.precip_occurrence_scale,
                        reduction="mean",
                    ).items():
                        precip_losses[name] = precip_losses.get(name, 0) + rolling_weights[step] * value
            physical_input = torch.cat([physical_input[:, 1:], raw_physical], dim=1).contiguous()

        loss_dict = {"loss_rec": loss_rec, **{f"precip_{name}": value for name, value in precip_losses.items()}}
        if rollout_segment_size is None:
            return loss_dict
        rollout_state.update(physical_input=physical_input.detach(), step=last_step)
        return loss_dict, rollout_state, last_step > loss_steps[-1]

    @torch.no_grad()
    def test_cascade(self, data: dict[str, Any]) -> Tensor:
        normalized_input = data["input"]
        total_step = self.start_frames + max(self.test_frames)

        save_steps = {int(step): index for index, step in enumerate(np.asarray(self.test_frames) // self.out_frames)}
        height, width = normalized_input.shape[-2:]
        outputs = torch.zeros(
            normalized_input.shape[0], len(save_steps), len(self.test_chans), height, width, device=normalized_input.device, dtype=torch.float32
        )

        for _ in range(self.members):
            physical_input = self.inv_normalize(normalized_input)
            for step in range(total_step):
                raw_physical, raw_normalized = self._raw_forecast(physical_input, data["times"], step=step)
                forecast_step = step + 1 - self.start_frames
                if forecast_step in save_steps:
                    lead = self.start_frames + forecast_step
                    correction = self.predict_correction(raw_normalized, lead=lead)
                    outputs[:, save_steps[forecast_step]] += self.merge_correction(raw_normalized, correction).float()[:, 0, self.test_chans]
                physical_input = torch.cat([physical_input[:, 1:], raw_physical], dim=1).contiguous()

        return (outputs / self.members).to(normalized_input)

    @torch.no_grad()
    def export_onnx(self, raw_pred: Tensor, lead: Tensor) -> Tensor:
        """Export the standalone corrector, without calling the original ONNX.

        ``raw_pred`` is a physical full-channel forecast, [B, T, C, H, W]; only
        its last frame is used. ``lead`` is float32 [B], measured in forecast
        steps (1 means +6 hours for IWC), and stays an ONNX input. The output is
        physical corrected channels in ``correct_chans`` order, [B, 1, K, H, W].
        This output is postprocessing and must not replace the raw ONNX feedback.
        """
        raw_normalized = self.normalize(raw_pred[:, -1:].clone())
        correction = self.predict_correction(raw_normalized.to(dtype=self.dtype), lead=lead)
        corrected = raw_normalized[:, :, self.correct_chans] + correction[:, None].float()
        return self.inv_normalize(corrected, self.correct_chans)
