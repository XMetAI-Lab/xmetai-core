from collections.abc import Sequence
from typing import Any

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
        *,
        test_times: Sequence[int],
        neighborhood_size: int = 1,
        channel_index: int = 0,
        value_scale: float = 1.0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.thresholds = tuple(float(v) for v in thresholds)
        self.neighborhood_size = int(neighborhood_size)
        self.channel_index = int(channel_index)
        self.value_scale = float(value_scale)
        self.output_key = output_key
        self.target_key = target_key
        self.test_times = tuple(int(v) for v in test_times)
        if len(self.thresholds) == 0:
            raise ValueError("thresholds must not be empty")
        if len(self.test_times) == 0:
            raise ValueError("test_times must not be empty")
        if self.neighborhood_size <= 0:
            raise ValueError(f"neighborhood_size must be positive, got {neighborhood_size}")
        if self.neighborhood_size % 2 == 0:
            raise ValueError(f"neighborhood_size must be odd to keep the neighborhood centered, got {neighborhood_size}")
        if self.value_scale <= 0:
            raise ValueError(f"value_scale must be positive, got {value_scale}")

    def _empty_counts(self) -> dict[float, dict[str, float]]:
        return {threshold: {"hit": 0.0, "false_alarm": 0.0, "miss": 0.0} for threshold in self.thresholds}

    def reset(self) -> None:
        self.counts_by_frame: dict[int, dict[float, dict[str, float]]] = {}

    def _neighbor_any(self, mask: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(mask.float(), kernel_size=self.neighborhood_size, stride=1, padding=self.neighborhood_size // 2)
        return pooled > 0

    def _accumulate_counts(self, counts: dict[float, dict[str, float]], pred: torch.Tensor, obs: torch.Tensor) -> None:
        for threshold in self.thresholds:
            pred_mask = pred >= threshold
            obs_mask = obs >= threshold
            obs_neighborhood = self._neighbor_any(obs_mask)
            pred_neighborhood = self._neighbor_any(pred_mask)
            counts[threshold]["hit"] += float((pred_mask & obs_neighborhood).sum().item())
            counts[threshold]["false_alarm"] += float((pred_mask & ~obs_neighborhood).sum().item())
            counts[threshold]["miss"] += float((obs_mask & ~pred_neighborhood).sum().item())

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        output = outputs[self.output_key]
        target = outputs[self.target_key]
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{type(self).__name__} expects output/target as (B, T, C, H, W), got {tuple(output.shape)} and {tuple(target.shape)}")
        if self.channel_index < 0 or self.channel_index >= output.shape[2] or self.channel_index >= target.shape[2]:
            raise ValueError(f"channel_index={self.channel_index} is out of range for output/target shapes {tuple(output.shape)} and {tuple(target.shape)}")

        pred = output[:, :, self.channel_index] * self.value_scale
        obs = target[:, :, self.channel_index] * self.value_scale
        frame_start = int(outputs.get("frame_start", 0))
        if frame_start < 0:
            raise ValueError(f"frame_start must be non-negative, got {frame_start}")
        if frame_start + output.shape[1] > len(self.test_times):
            raise ValueError(
                f"{type(self).__name__} got output with {output.shape[1]} frames at frame_start={frame_start}, "
                f"but test_times has {len(self.test_times)} entries"
            )
        for i in range(output.shape[1]):
            frame_index = frame_start + i
            self.counts_by_frame.setdefault(frame_index, self._empty_counts())
            self._accumulate_counts(self.counts_by_frame[frame_index], pred[:, i : i + 1], obs[:, i : i + 1])

    def _metric_value(self, hit: float, false_alarm: float, miss: float, eps: float) -> float:
        raise NotImplementedError

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()

        gathered_by_frame = comm.gather(self.counts_by_frame, dst=0)
        if not comm.is_main_process():
            return None

        counts_by_frame: dict[int, dict[float, dict[str, float]]] = {}
        for worker_counts in gathered_by_frame:
            for frame_index, threshold_counts in worker_counts.items():
                frame_counts = counts_by_frame.setdefault(frame_index, self._empty_counts())
                for threshold in self.thresholds:
                    frame_counts[threshold]["hit"] += threshold_counts[threshold]["hit"]
                    frame_counts[threshold]["false_alarm"] += threshold_counts[threshold]["false_alarm"]
                    frame_counts[threshold]["miss"] += threshold_counts[threshold]["miss"]

        # One row per lead and one score column per threshold.
        results: dict[str, list[int | float]] = {"Times": list(self.test_times)}
        columns = {threshold: f"{self.metric_prefix}@{threshold:g}" for threshold in self.thresholds}
        for column in columns.values():
            results[column] = []
        eps = 1.0e-6
        for frame_index in range(len(self.test_times)):
            frame_counts = counts_by_frame.get(frame_index, self._empty_counts())
            for threshold in self.thresholds:
                hit = frame_counts[threshold]["hit"]
                false_alarm = frame_counts[threshold]["false_alarm"]
                miss = frame_counts[threshold]["miss"]
                results[columns[threshold]].append(self._metric_value(hit, false_alarm, miss, eps))
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
