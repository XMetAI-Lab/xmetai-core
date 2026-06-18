from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from xmetai.core import DatasetEvaluator
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = [
    "ThreatScore",
    "FalseAlarmRate",
    "ProbabilityOfDetection",
    "FrequencyBias",
]


class _ThresholdContingencyMetric(DatasetEvaluator):
    metric_name = ""
    metric_prefix = ""

    def __init__(
        self,
        thresholds: Sequence[float],
        neighborhood_size: int = 1,
        channel_index: int = 0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.thresholds = tuple(float(v) for v in thresholds)
        self.neighborhood_size = int(neighborhood_size)
        self.channel_index = int(channel_index)
        self.output_key = output_key
        self.target_key = target_key
        if len(self.thresholds) == 0:
            raise ValueError("thresholds must not be empty")
        if self.neighborhood_size <= 0:
            raise ValueError(f"neighborhood_size must be positive, got {neighborhood_size}")
        if self.neighborhood_size % 2 == 0:
            raise ValueError(f"neighborhood_size must be odd to keep the neighborhood centered, got {neighborhood_size}")

    def reset(self) -> None:
        self.counts = {threshold: {"hit": 0.0, "false_alarm": 0.0, "miss": 0.0} for threshold in self.thresholds}

    def _neighbor_any(self, mask: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(mask.float(), kernel_size=self.neighborhood_size, stride=1, padding=self.neighborhood_size // 2)
        return pooled > 0

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        output = outputs[self.output_key]
        target = outputs[self.target_key]
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{type(self).__name__} expects output/target as (B, T, C, H, W), got {tuple(output.shape)} and {tuple(target.shape)}")
        if self.channel_index < 0 or self.channel_index >= output.shape[2] or self.channel_index >= target.shape[2]:
            raise ValueError(f"channel_index={self.channel_index} is out of range for output/target shapes {tuple(output.shape)} and {tuple(target.shape)}")

        pred = output[:, :, self.channel_index].reshape(-1, 1, *output.shape[-2:])
        obs = target[:, :, self.channel_index].reshape(-1, 1, *target.shape[-2:])
        for threshold in self.thresholds:
            pred_mask = pred >= threshold
            obs_mask = obs >= threshold
            obs_neighborhood = self._neighbor_any(obs_mask)
            pred_neighborhood = self._neighbor_any(pred_mask)
            self.counts[threshold]["hit"] += float((pred_mask & obs_neighborhood).sum().item())
            self.counts[threshold]["false_alarm"] += float((pred_mask & ~obs_neighborhood).sum().item())
            self.counts[threshold]["miss"] += float((obs_mask & ~pred_neighborhood).sum().item())

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        raise NotImplementedError

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered = comm.gather(self.counts, dst=0)
        if not comm.is_main_process():
            return None

        results: dict[str, list[float]] = {"Threshold": []}
        results[self.metric_prefix] = []
        eps = 1.0e-6
        for threshold in self.thresholds:
            hit = float(np.sum([item[threshold]["hit"] for item in gathered]))
            false_alarm = float(np.sum([item[threshold]["false_alarm"] for item in gathered]))
            miss = float(np.sum([item[threshold]["miss"] for item in gathered]))
            results["Threshold"].append(threshold)
            results[self.metric_prefix].append(self._metric_value(hit, false_alarm, miss, eps))
        return {self.metric_name: results}


@METRICS.register()
class ThreatScore(_ThresholdContingencyMetric):
    metric_name = "ThreatScore"
    metric_prefix = "TS"

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        return hit / (hit + false_alarm + miss + eps)


@METRICS.register()
class FalseAlarmRate(_ThresholdContingencyMetric):
    metric_name = "FalseAlarmRate"
    metric_prefix = "FAR"

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        return false_alarm / (hit + false_alarm + eps)


@METRICS.register()
class ProbabilityOfDetection(_ThresholdContingencyMetric):
    metric_name = "ProbabilityOfDetection"
    metric_prefix = "POD"

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        return hit / (hit + miss + eps)


@METRICS.register()
class FrequencyBias(_ThresholdContingencyMetric):
    metric_name = "FrequencyBias"
    metric_prefix = "Bias"

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        return (hit + false_alarm) / (hit + miss + eps)
