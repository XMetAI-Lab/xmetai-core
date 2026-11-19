"""UNet3D adapter for six-hour FENGQ15 precipitation correction."""

from __future__ import annotations

import os
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F

from xmetai.core.meta import Meta
from xmetai.layers.unet3d import UNet3D
from xmetai.registries import MODELS

ScaleValue = Union[str, os.PathLike, np.ndarray, torch.Tensor]


@MODELS.register()
class PrcpCorrectionUNet3D(Meta):
    """Log-space residual correction around the FENGQ precipitation baseline."""

    def __init__(
        self,
        input_min: ScaleValue,
        input_max: ScaleValue,
        in_channels: int = 30,
        base_channels: int = 128,
        precipitation_scale: float = 1.0,
        physical_loss_weight: float = 0.1,
        precision: str = "fp32",
        device: str = "cuda",
        physical_loss_thresholds: tuple[float, ...] = (10.0, 25.0, 50.0),
        physical_loss_weights: tuple[float, ...] = (1.0, 3.0, 4.0, 8.0),
        event_loss_weight: float = 0.0,
        event_loss_temperature: float = 0.25,
        event_loss_alpha: float = 1.0,
        event_loss_beta: float = 1.0,
    ):
        super().__init__(precision=precision, device=device)
        self.in_channels = int(in_channels)
        precipitation_scale = float(precipitation_scale)
        self.physical_loss_weight = float(physical_loss_weight)
        self.event_loss_weight = float(event_loss_weight)
        self.event_loss_temperature = float(event_loss_temperature)
        self.event_loss_alpha = float(event_loss_alpha)
        self.event_loss_beta = float(event_loss_beta)
        self.event_thresholds = (5.0, 10.0, 25.0, 50.0)
        if not np.isfinite(self.event_loss_weight) or self.event_loss_weight < 0:
            raise ValueError("event_loss_weight must be finite and non-negative.")
        for name in ("event_loss_temperature", "event_loss_alpha", "event_loss_beta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(precipitation_scale) or precipitation_scale <= 0.0:
            raise ValueError("precipitation_scale must be positive.")
        if not np.isfinite(self.physical_loss_weight) or self.physical_loss_weight < 0.0:
            raise ValueError("physical_loss_weight must be non-negative.")
        self.physical_loss_thresholds = tuple(float(value) for value in physical_loss_thresholds)
        self.physical_loss_weights = tuple(float(value) for value in physical_loss_weights)
        if (
            not self.physical_loss_thresholds
            or not np.isfinite(self.physical_loss_thresholds).all()
            or self.physical_loss_thresholds[0] <= 0
            or np.any(np.diff(self.physical_loss_thresholds) <= 0)
        ):
            raise ValueError("physical_loss_thresholds must be finite, positive and strictly increasing.")
        if (
            len(self.physical_loss_weights) != len(self.physical_loss_thresholds) + 1
            or not np.isfinite(self.physical_loss_weights).all()
            or min(self.physical_loss_weights) < 1
            or np.any(np.diff(self.physical_loss_weights) < 0)
        ):
            raise ValueError("physical_loss_weights must be finite, nondecreasing, at least one and have one more entry than thresholds.")

        min_tensor = self._load_scale(input_min, self.in_channels, "input_min")
        max_tensor = self._load_scale(input_max, self.in_channels, "input_max")
        if not torch.all(max_tensor > min_tensor):
            invalid = torch.nonzero(max_tensor <= min_tensor, as_tuple=False)[:, 0].unique().tolist()
            raise ValueError(f"input_max must exceed input_min for every channel; invalid channels: {invalid}")
        self.register_buffer("input_min", min_tensor)
        self.register_buffer("input_max", max_tensor)
        self.register_buffer("precipitation_scale", torch.tensor(precipitation_scale, dtype=torch.float32))
        self.net = UNet3D(self.in_channels, 1, base_channels=base_channels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Datasets keep the configured normalization even when model buffers load.
        for name in ("input_min", "input_max"):
            key = prefix + name
            if key in state_dict and not torch.equal(state_dict[key].detach().cpu(), getattr(self, name).detach().cpu()):
                raise ValueError(f"Checkpoint {key} does not match the configured normalization statistics.")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _load_scale(value: ScaleValue, channels: int, name: str) -> torch.Tensor:
        if isinstance(value, (str, os.PathLike)):
            value = np.load(os.fspath(value))
        tensor = torch.as_tensor(value, dtype=torch.float32)
        expected_shape = (channels, 1, 1)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        return tensor.contiguous()

    @staticmethod
    def _masked_spatial_sum(error: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Return the valid weighted mean scaled by the number of grid cells."""
        weights = valid if weights is None else torch.where(valid, weights, 0.0)
        count = weights.sum(dim=(-2, -1))
        if torch.any(count == 0):
            raise ValueError("Every sample lead must contain valid observation cells.")
        total = (torch.where(valid, error, 0.0) * weights).sum(dim=(-2, -1))
        return (total / count * (error.shape[-2] * error.shape[-1])).mean()

    def _balanced_physical_losses(self, output: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Share capped observation weights between physical MAE and MSE."""
        weights = torch.full_like(target, self.physical_loss_weights[0])
        for threshold, weight in zip(self.physical_loss_thresholds, self.physical_loss_weights[1:]):
            weights = torch.where(target >= threshold, weight, weights)
        error = output - target
        return {
            "loss_physical_l1": self.physical_loss_weight * self._masked_spatial_sum(error.abs(), valid, weights),
            "loss_physical_l2": 0.0075 * self._masked_spatial_sum(error.square(), valid, weights),
        }

    @staticmethod
    def _smooth_l1_spatial_sum(output: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return PrcpCorrectionUNet3D._masked_spatial_sum(F.smooth_l1_loss(output, target, reduction="none"), valid)

    def _joint_event_losses(self, log_prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pool soft event counts over this process's batch, leads and valid cells.

        Alpha weights false alarms and beta weights misses. Both equal to one
        gives soft CSI (Jaccard), following CIRA's custom-loss guide and SMP's
        soft_tversky_score definition. A threshold with no observed events uses
        mean soft exceedance instead, retaining a false-alarm gradient.
        """
        if self.event_loss_weight == 0:
            return {}
        thresholds = log_prediction.new_tensor(self.event_thresholds, dtype=torch.float32).reshape(1, 1, -1, 1, 1)
        log_thresholds = torch.log1p(thresholds / self.precipitation_scale)
        # Supervise the unclamped log prediction so dry outputs can recover.
        safe_prediction = torch.where(valid, log_prediction.float(), 0.0)
        probability = torch.sigmoid((safe_prediction - log_thresholds) / self.event_loss_temperature)
        probability = torch.where(valid, probability, 0.0)
        observed = (target >= thresholds) & valid
        axes = (0, 1, 3, 4)
        hits = (probability * observed).sum(dim=axes)
        false_alarms = (probability * (~observed & valid)).sum(dim=axes)
        misses = ((1.0 - probability) * observed).sum(dim=axes)
        denominator = hits + self.event_loss_alpha * false_alarms + self.event_loss_beta * misses
        event_loss = 1.0 - hits / denominator.clamp_min(1e-7)
        empty_loss = probability.sum(dim=axes) / valid.sum()
        loss = torch.where(observed.sum(dim=axes) > 0, event_loss, empty_loss)
        scale = self.event_loss_weight * (target.shape[-2] * target.shape[-1]) / len(self.event_thresholds)
        return {f"loss_event_{threshold:g}mm": scale * loss[index] for index, threshold in enumerate(self.event_thresholds)}

    def _to_model(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _fengq_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.input_max[:1] - self.input_min[:1]
        offset = self.input_min[:1]
        precipitation = inputs[:, :, :1].float()
        return torch.clamp_min(precipitation * scale + offset, 0.0).to(inputs)

    def _reconstruct(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self._fengq_baseline(inputs).float()
        log_baseline = torch.log1p(baseline / self.precipitation_scale)
        log_prediction = log_baseline + residual.float()
        prediction = self.precipitation_scale * torch.expm1(torch.clamp_min(log_prediction, 0.0))
        return prediction, log_prediction, baseline

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, Any]:
        inputs = self._to_model(data["inputs"])
        residual = self.net(inputs)
        prediction, log_prediction, baseline = self._reconstruct(inputs, residual)

        if self.training:
            targets = data["targets"].to(device=self.device, dtype=torch.float32, non_blocking=True)
            valid = data["valid_mask"].to(device=self.device, non_blocking=True)
            if valid.dtype != torch.bool or valid.shape != targets.shape:
                raise ValueError("valid_mask must be boolean with the same shape as targets.")
            targets = torch.where(valid, targets, 0.0)
            log_targets = torch.log1p(targets / self.precipitation_scale)
            return {
                "loss_log_smooth_l1": self._smooth_l1_spatial_sum(log_prediction, log_targets, valid),
                **self._balanced_physical_losses(prediction, targets, valid),
                **self._joint_event_losses(log_prediction, targets, valid),
            }

        result = {
            "output": prediction,
            "output_by_mode": {"corrected": prediction, "baseline": baseline},
        }
        if "targets" in data:
            result["target"] = data["targets"].to(device=self.device, dtype=torch.float32, non_blocking=True)
            result["weight"] = data["valid_mask"].to(device=self.device, non_blocking=True)
        return result
