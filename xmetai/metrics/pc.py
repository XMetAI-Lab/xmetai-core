# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 XMetAI Lab. Licensed under GPL-3.0-or-later.
#
# @Author  : Liu Junjie @ XmetAI lab & Anhui Climate Center
# @Email   : junjie.liu.china@gmail.com
#
# Source: 第三届全国智能气候预测技术方法竞赛预测质量检验方法，
#         第三部分（二）"预报准确率（PC）"，公式（5）–（7），第 4–5 页。

"""气温预报准确率（Prediction Correction，PC）评分。

适用于逐日日平均气温、日最高气温和日最低气温预测检验。PC 值范围为 0–100%，
值越高表示预测技巧越高；完美预测时 PC 为 100%。

15–60 天总体加权 PC 权重：
  15–20 天 = 5,  21–30 天 = 4,  31–40 天 = 3,  41–50 天 = 2,  51–60 天 = 1
"""

import math
from collections.abc import Sequence
from typing import Any

import torch

from xmetai.core import DatasetEvaluator
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = ["PC"]


def _lead_weight(lead: int) -> float:
    """竞赛规定的时效权重函数。"""
    if 15 <= lead <= 20:
        return 5.0
    if 21 <= lead <= 30:
        return 4.0
    if 31 <= lead <= 40:
        return 3.0
    if 41 <= lead <= 50:
        return 2.0
    if 51 <= lead <= 60:
        return 1.0
    return 0.0


@METRICS.register()
class PC(DatasetEvaluator):
    """按温度误差容差计算逐时效 PC 及 15–60 天总体加权 PC。

    Parameters
    ----------
    test_times : int sequence
        每个时效对应的模型输出时间（小时数），如 (360, 384, ...) 对应 lead 15, 16, ... 天。
    tol : float
        绝对误差容差（℃），默认 2.0。
    channel_index : int
        计算 PC 的通道索引（与 output 的通道维对齐）。
    channel_name : str | None
        如果提供，从 test_names 查找通道索引，优先级高于 channel_index。
    test_names : str sequence
        通道名列表（与输出的通道维对齐），配合 channel_name 使用。
    leads : int sequence | None
        每个帧对应的预报时效（天数）。默认 None 时要求 test_times 正好是 15–60 天的 46 个时效。
    output_key : str
        outputs 字典中的预测值 key，默认 "output"。
    target_key : str
        outputs 字典中的观测值 key，默认 "target"。
    """

    def __init__(
        self,
        *,
        test_times: Sequence[int],
        tol: float = 2.0,
        channel_index: int = 0,
        channel_name: str | None = None,
        test_names: Sequence[str] = (),
        leads: Sequence[int] | None = None,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.test_times = tuple(int(value) for value in test_times)
        self.tol = float(tol)
        self.channel_index = list(test_names).index(channel_name) if channel_name is not None else int(channel_index)
        self.output_key = output_key
        self.target_key = target_key

        if not self.test_times:
            raise ValueError("test_times must not be empty")
        if leads is None:
            if len(self.test_times) != 46:
                raise ValueError("leads must be provided unless test_times contains all 46 leads from 15 to 60")
            leads = tuple(range(15, 61))
        self.leads = tuple(int(value) for value in leads)
        if len(self.leads) != len(self.test_times):
            raise ValueError(f"leads has {len(self.leads)} entries but test_times has {len(self.test_times)}")
        self.lead_weights = tuple(_lead_weight(lead) for lead in self.leads)
        if not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and non-negative")
        self.reset()

    def reset(self) -> None:
        self.correct_by_frame = [0 for _ in self.test_times]
        self.valid_by_frame = [0 for _ in self.test_times]

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        forecast = outputs[self.output_key]
        observed = outputs[self.target_key]
        if forecast.ndim != 5 or observed.ndim != 5:
            raise ValueError("PC expects output and target shaped (B, T, C, H, W)")
        if forecast.shape != observed.shape:
            raise ValueError(f"output and target shapes differ: {tuple(forecast.shape)} vs {tuple(observed.shape)}")
        if not 0 <= self.channel_index < forecast.shape[2]:
            raise ValueError(f"channel_index={self.channel_index} is out of range for {tuple(forecast.shape)}")
        frame_start = int(outputs.get("frame_start", 0))
        if frame_start < 0 or frame_start + forecast.shape[1] > len(self.test_times):
            raise ValueError(f"output frames at frame_start={frame_start} exceed configured test_times")

        forecast = forecast[:, :, self.channel_index].detach()
        observed = observed[:, :, self.channel_index].detach()
        for frame in range(forecast.shape[1]):
            pred = forecast[:, frame]
            target = observed[:, frame]
            valid = torch.isfinite(pred) & torch.isfinite(target)
            frame_index = frame_start + frame
            self.valid_by_frame[frame_index] += int(valid.sum().item())
            self.correct_by_frame[frame_index] += int((((pred - target).abs() <= self.tol) & valid).sum().item())

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered_counts = comm.gather((self.correct_by_frame, self.valid_by_frame), dst=0)
        if not comm.is_main_process():
            return None

        correct = [0 for _ in self.test_times]
        valid = [0 for _ in self.test_times]
        for worker_correct, worker_valid in gathered_counts:
            for frame in range(len(self.test_times)):
                correct[frame] += worker_correct[frame]
                valid[frame] += worker_valid[frame]
        scores = [100.0 * n_correct / n_valid if n_valid else float("nan") for n_correct, n_valid in zip(correct, valid)]
        valid_scores = [(score, weight) for score, weight in zip(scores, self.lead_weights) if math.isfinite(score) and weight > 0]
        total_weight = sum(weight for _, weight in valid_scores)
        overall = sum(score * weight for score, weight in valid_scores) / total_weight if total_weight else float("nan")
        # Overall 以单元素列表返回，因为 inference_on_dataset 会用 tabulate 逐列迭代渲染，
        # 标量会触发 TypeError: 'float' object is not iterable。
        return {"PC": {"Times": list(self.test_times), f"PC@{self.tol:g}": scores, "Overall": [overall]}}
