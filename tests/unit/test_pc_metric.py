# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 XMetAI Lab. Licensed under GPL-3.0-or-later.
#
# @Time    : 2026-10-27 02:56:13 CST
# @Author  : Liu Junjie @ XmetAI lab & Anhui Climate Center
# @Email   : junjie.liu.china@gmail.com
# @File    : test_pc_metric.py
# @Software: xmetai-core
# @Source  : 第三届全国智能气候预测技术方法竞赛预测质量检验方法，
#            第三部分（二）“预报准确率（PC）”，公式（5）–（7），第 4–5 页。

"""Tests for the temperature PC evaluator."""

import math

import pytest
import torch

from xmetai.metrics import PC
from xmetai.registries import METRICS


def test_temperature_pc_calculates_percent_within_tolerance():
    metric = PC(test_times=(360,), tol=2.0, leads=(15,))
    target = torch.tensor([[[[[10.0, 10.0], [10.0, float("nan")]]]]])
    forecast = torch.tensor([[[[[12.0, 12.1], [float("nan"), 0.0]]]]])

    metric.process({}, {"output": forecast, "target": target})
    results = metric.evaluate()

    assert results is not None
    assert results == {"PC": {"Times": [360], "PC@2": [50.0], "Overall": 50.0}}


def test_temperature_pc_returns_nan_when_no_grid_cells_are_valid():
    metric = PC(test_times=(360,), tol=2.0, leads=(15,))
    missing = torch.full((1, 1, 1, 2, 2), float("nan"))

    metric.process({}, {"output": missing, "target": missing.clone()})
    results = metric.evaluate()

    assert results is not None
    assert math.isnan(results["PC"]["PC@2"][0])
    assert math.isnan(results["PC"]["Overall"])


def test_temperature_pc_selects_named_channel_and_respects_frame_start():
    metric = PC(test_times=(6, 12, 18), tol=2.0, channel_name="t2m", test_names=("tp", "t2m"), leads=(15, 16, 17))
    target = torch.zeros(1, 2, 2, 1, 2)
    forecast = target.clone()
    target[:, :, 1] = 10.0
    forecast[:, :, 1, :, 0] = 12.0  # exactly at tolerance, counted as correct
    forecast[:, :, 1, :, 1] = 12.1

    metric.process({}, {"output": forecast, "target": target, "frame_start": 1})
    results = metric.evaluate()

    assert results is not None
    assert results["PC"]["Times"] == [6, 12, 18]
    assert math.isnan(results["PC"]["PC@2"][0])
    assert results["PC"]["PC@2"][1:] == pytest.approx([50.0, 50.0])
    assert results["PC"]["Overall"] == pytest.approx(50.0)


def test_temperature_pc_overall_uses_contest_lead_weights():
    metric = PC(test_times=(360, 384), tol=2.0, leads=(15, 21))
    target = torch.zeros(1, 2, 1, 1, 2)
    forecast = torch.zeros_like(target)
    target[:, :, 0, 0, 0] = 10.0
    forecast[0, 0, 0, 0, 0] = 10.0  # lead 15 PC = 100%
    forecast[0, 1, 0, 0, 0] = 13.0  # lead 21 PC = 50%

    metric.process({}, {"output": forecast, "target": target})
    results = metric.evaluate()

    assert results is not None
    assert results["PC"]["PC@2"] == pytest.approx([100.0, 50.0])
    assert results["PC"]["Overall"] == pytest.approx((5.0 * 100.0 + 4.0 * 50.0) / 9.0)


def test_temperature_pc_is_registered():
    assert METRICS.get("PC") is PC


@pytest.mark.parametrize("tol", [-1.0, float("inf"), float("nan")])
def test_temperature_pc_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        PC(test_times=(360,), tol=tol, leads=(15,))
