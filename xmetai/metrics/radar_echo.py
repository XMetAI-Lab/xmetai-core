from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from xmetai.core import DatasetEvaluator
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = [
    "RadarSSIM",
    "SSIM",
    "RadarFSS",
    "FSS",
    "RadarLPIPS",
    "LPIPS",
]


def _create_gaussian_window(
    window_size: int,
    sigma: float,
    channel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gaussian_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute mean SSIM per batch element for ``(B, C, H, W)`` tensors."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must share shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    channel = pred.shape[1]
    window = _create_gaussian_window(window_size, sigma, channel, pred.device, pred.dtype)
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target, window, padding=padding, groups=channel)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.flatten(1).mean(dim=1)


def _fss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    neighborhood_size: int,
) -> tuple[float, float]:
    """Return ``(sum((M-O)^2), sum(M^2 + O^2))`` for Roberts & Lean (2008) FSS."""
    pred_fraction = F.avg_pool2d(
        (pred >= threshold).float(),
        kernel_size=neighborhood_size,
        stride=1,
        padding=neighborhood_size // 2,
    )
    obs_fraction = F.avg_pool2d(
        (target >= threshold).float(),
        kernel_size=neighborhood_size,
        stride=1,
        padding=neighborhood_size // 2,
    )
    diff_sq = (pred_fraction - obs_fraction) ** 2
    reference = pred_fraction**2 + obs_fraction**2
    return float(diff_sq.sum().item()), float(reference.sum().item())


def _resolve_data_range(pred: torch.Tensor, target: torch.Tensor, data_range: float | None) -> float:
    if data_range is not None:
        return float(data_range)
    value_range = max(pred.max().item(), target.max().item()) - min(pred.min().item(), target.min().item())
    return value_range if value_range > 1.0e-6 else 1.0


def _normalize_to_lpips_input(
    tensor: torch.Tensor,
    data_min: float,
    data_max: float,
) -> torch.Tensor:
    if data_max <= data_min:
        raise ValueError(f"data_max must be greater than data_min, got {data_max} and {data_min}")
    normalized = (tensor - data_min) / (data_max - data_min)
    normalized = normalized.clamp(0.0, 1.0)
    normalized = normalized * 2.0 - 1.0
    if tensor.shape[1] == 1:
        normalized = normalized.repeat(1, 3, 1, 1)
    return normalized


class _RadarEchoMetric(DatasetEvaluator):
    def __init__(
        self,
        test_times: Sequence[int],
        *,
        channel_index: int = 0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.test_times = tuple(int(v) for v in test_times)
        self.channel_index = int(channel_index)
        self.output_key = output_key
        self.target_key = target_key
        if len(self.test_times) == 0:
            raise ValueError("test_times must not be empty")

    def _extract_frames(self, outputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int]:
        output = outputs[self.output_key]
        target = outputs[self.target_key]
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"{type(self).__name__} expects output/target as (B, T, C, H, W), " f"got {tuple(output.shape)} and {tuple(target.shape)}")
        if self.channel_index < 0 or self.channel_index >= output.shape[2] or self.channel_index >= target.shape[2]:
            raise ValueError(f"channel_index={self.channel_index} is out of range for output/target shapes " f"{tuple(output.shape)} and {tuple(target.shape)}")

        pred = output[:, :, self.channel_index]
        obs = target[:, :, self.channel_index]
        frame_start = int(outputs.get("frame_start", 0))
        if frame_start < 0:
            raise ValueError(f"frame_start must be non-negative, got {frame_start}")
        if frame_start + output.shape[1] > len(self.test_times):
            raise ValueError(
                f"{type(self).__name__} got output with {output.shape[1]} frames at frame_start={frame_start}, "
                f"but test_times has {len(self.test_times)} entries"
            )
        return pred, obs, frame_start


@METRICS.register()
class RadarSSIM(_RadarEchoMetric):
    metric_name = "RadarSSIM"
    metric_prefix = "SSIM"

    def __init__(
        self,
        test_times: Sequence[int],
        *,
        data_range: float | None = None,
        window_size: int = 11,
        channel_index: int = 0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__(
            test_times,
            channel_index=channel_index,
            output_key=output_key,
            target_key=target_key,
        )
        self.data_range = data_range
        self.window_size = int(window_size)
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")

    def reset(self) -> None:
        self._sum_by_frame: dict[int, float] = {}
        self._count_by_frame: dict[int, float] = {}

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        pred, obs, frame_start = self._extract_frames(outputs)
        for frame_offset in range(pred.shape[1]):
            frame_index = frame_start + frame_offset
            pred_frame = pred[:, frame_offset : frame_offset + 1]
            obs_frame = obs[:, frame_offset : frame_offset + 1]
            data_range = _resolve_data_range(pred_frame, obs_frame, self.data_range)
            ssim_values = _ssim_per_sample(pred_frame, obs_frame, data_range, self.window_size)
            self._sum_by_frame[frame_index] = self._sum_by_frame.get(frame_index, 0.0) + float(ssim_values.sum().item())
            self._count_by_frame[frame_index] = self._count_by_frame.get(frame_index, 0.0) + float(ssim_values.numel())

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered_sum = comm.gather(self._sum_by_frame, dst=0)
        gathered_count = comm.gather(self._count_by_frame, dst=0)
        if not comm.is_main_process():
            return None

        sum_by_frame: dict[int, float] = {}
        count_by_frame: dict[int, float] = {}
        for worker_sum, worker_count in zip(gathered_sum, gathered_count):
            for frame_index, value in worker_sum.items():
                sum_by_frame[frame_index] = sum_by_frame.get(frame_index, 0.0) + value
            for frame_index, value in worker_count.items():
                count_by_frame[frame_index] = count_by_frame.get(frame_index, 0.0) + value

        results: dict[str, list[int | float]] = {"Times": [], self.metric_prefix: []}
        for frame_index, lead_time in enumerate(self.test_times):
            total = sum_by_frame.get(frame_index, 0.0)
            count = count_by_frame.get(frame_index, 0.0)
            value = total / count if count > 0 else 0.0
            results["Times"].append(lead_time)
            results[self.metric_prefix].append(value)
        return {self.metric_name: results}


@METRICS.register()
class RadarFSS(_RadarEchoMetric):
    metric_name = "RadarFSS"
    metric_prefix = "FSS"

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        test_times: Sequence[int],
        neighborhood_size: int = 5,
        channel_index: int = 0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__(
            test_times,
            channel_index=channel_index,
            output_key=output_key,
            target_key=target_key,
        )
        self.thresholds = tuple(float(v) for v in thresholds)
        self.neighborhood_size = int(neighborhood_size)
        if len(self.thresholds) == 0:
            raise ValueError("thresholds must not be empty")
        if self.neighborhood_size <= 0:
            raise ValueError(f"neighborhood_size must be positive, got {neighborhood_size}")
        if self.neighborhood_size % 2 == 0:
            raise ValueError(f"neighborhood_size must be odd to keep the neighborhood centered, got {neighborhood_size}")

    def _empty_components(self) -> dict[float, dict[str, float]]:
        return {threshold: {"diff_sq": 0.0, "reference": 0.0} for threshold in self.thresholds}

    def reset(self) -> None:
        self._components_by_frame: dict[int, dict[float, dict[str, float]]] = {}

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        pred, obs, frame_start = self._extract_frames(outputs)
        for frame_offset in range(pred.shape[1]):
            frame_index = frame_start + frame_offset
            pred_frame = pred[:, frame_offset : frame_offset + 1]
            obs_frame = obs[:, frame_offset : frame_offset + 1]
            frame_components = self._components_by_frame.setdefault(frame_index, self._empty_components())
            for threshold in self.thresholds:
                diff_sq, reference = _fss_components(pred_frame, obs_frame, threshold, self.neighborhood_size)
                frame_components[threshold]["diff_sq"] += diff_sq
                frame_components[threshold]["reference"] += reference

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered_by_frame = comm.gather(self._components_by_frame, dst=0)
        if not comm.is_main_process():
            return None

        components_by_frame: dict[int, dict[float, dict[str, float]]] = {}
        for worker_components in gathered_by_frame:
            for frame_index, threshold_components in worker_components.items():
                frame_components = components_by_frame.setdefault(frame_index, self._empty_components())
                for threshold in self.thresholds:
                    frame_components[threshold]["diff_sq"] += threshold_components[threshold]["diff_sq"]
                    frame_components[threshold]["reference"] += threshold_components[threshold]["reference"]

        results: dict[str, list[int | float]] = {"Times": [], "Threshold": [], self.metric_prefix: []}
        eps = 1.0e-6
for frame_index, lead_time in enumerate(self.test_times):
    has_frame = frame_index in components_by_frame
    frame_components = components_by_frame.get(frame_index, self._empty_components())
    for threshold in self.thresholds:
        diff_sq = frame_components[threshold]["diff_sq"]
        reference = frame_components[threshold]["reference"]
        if not has_frame:
            fss = 0.0
        else:
            fss = 1.0 - diff_sq / (reference + eps) if reference > 0 else 1.0
                results["Times"].append(lead_time)
                results["Threshold"].append(threshold)
                results[self.metric_prefix].append(fss)
        return {self.metric_name: results}


@METRICS.register()
class RadarLPIPS(_RadarEchoMetric):
    metric_name = "RadarLPIPS"
    metric_prefix = "LPIPS"

    def __init__(
        self,
        test_times: Sequence[int],
        *,
        data_min: float = 0.0,
        data_max: float = 70.0,
        net: str = "alex",
        channel_index: int = 0,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__(
            test_times,
            channel_index=channel_index,
            output_key=output_key,
            target_key=target_key,
        )
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.net = net
        self._lpips_model: Any = None

    def _get_lpips_model(self, device: torch.device) -> Any:
        if self._lpips_model is None:
            import lpips

            self._lpips_model = lpips.LPIPS(net=self.net).to(device)
            self._lpips_model.eval()
        return self._lpips_model

    def reset(self) -> None:
        self._sum_by_frame: dict[int, float] = {}
        self._count_by_frame: dict[int, float] = {}

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        pred, obs, frame_start = self._extract_frames(outputs)
        lpips_model = self._get_lpips_model(pred.device)
        for frame_offset in range(pred.shape[1]):
            frame_index = frame_start + frame_offset
            pred_frame = pred[:, frame_offset : frame_offset + 1]
            obs_frame = obs[:, frame_offset : frame_offset + 1]
            pred_input = _normalize_to_lpips_input(pred_frame, self.data_min, self.data_max)
            obs_input = _normalize_to_lpips_input(obs_frame, self.data_min, self.data_max)
            with torch.no_grad():
                distances = lpips_model(pred_input, obs_input)
            self._sum_by_frame[frame_index] = self._sum_by_frame.get(frame_index, 0.0) + float(distances.sum().item())
            self._count_by_frame[frame_index] = self._count_by_frame.get(frame_index, 0.0) + float(distances.numel())

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered_sum = comm.gather(self._sum_by_frame, dst=0)
        gathered_count = comm.gather(self._count_by_frame, dst=0)
        if not comm.is_main_process():
            return None

        sum_by_frame: dict[int, float] = {}
        count_by_frame: dict[int, float] = {}
        for worker_sum, worker_count in zip(gathered_sum, gathered_count):
            for frame_index, value in worker_sum.items():
                sum_by_frame[frame_index] = sum_by_frame.get(frame_index, 0.0) + value
            for frame_index, value in worker_count.items():
                count_by_frame[frame_index] = count_by_frame.get(frame_index, 0.0) + value

        results: dict[str, list[int | float]] = {"Times": [], self.metric_prefix: []}
        for frame_index, lead_time in enumerate(self.test_times):
            total = sum_by_frame.get(frame_index, 0.0)
            count = count_by_frame.get(frame_index, 0.0)
            value = total / count if count > 0 else 0.0
            results["Times"].append(lead_time)
            results[self.metric_prefix].append(value)
        return {self.metric_name: results}


SSIM = RadarSSIM
FSS = RadarFSS
LPIPS = RadarLPIPS
