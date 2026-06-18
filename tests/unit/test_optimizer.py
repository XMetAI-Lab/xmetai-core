from pathlib import Path

import pytest
import torch

from xmetai.config import LazyConfig
from xmetai.config.instantiate import instantiate
from xmetai.core.optimizer import get_default_optimizer_params

REPO_ROOT = Path(__file__).resolve().parents[2]


class BackboneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


def _group_weight_decay(params, target_param):
    for group in params:
        if any(param is target_param for param in group["params"]):
            return group.get("weight_decay")
    raise AssertionError("target parameter not found")


def _group_lr(params, target_param):
    for group in params:
        if any(param is target_param for param in group["params"]):
            return group.get("lr")
    raise AssertionError("target parameter not found")


def test_weight_decay_factor_scales_matching_parameter_groups():
    model = BackboneModel()

    params = get_default_optimizer_params(
        model,
        weight_decay=0.1,
        overrides={"backbone": {"weight_decay_factor": 0.1}},
    )

    assert _group_weight_decay(params, model.backbone.weight) == pytest.approx(0.01)
    assert _group_weight_decay(params, model.head.weight) == pytest.approx(0.1)


def test_weight_decay_factor_without_base_weight_decay_logs_error(caplog):
    model = BackboneModel()

    params = get_default_optimizer_params(
        model,
        overrides={"backbone": {"weight_decay_factor": 0.1}},
    )

    assert _group_weight_decay(params, model.backbone.weight) is None
    assert "weight_decay_factor override requires" in caplog.text


def test_lazyconfig_adamw_weight_decay_factor_scales_constructor_decay():
    model = BackboneModel()
    cfg = LazyConfig.load(str(REPO_ROOT / "common/optim.py"))
    cfg.AdamW.params.model = model
    cfg.AdamW.params.overrides = {"backbone": {"weight_decay_factor": 0.1}}

    optimizer = instantiate(cfg.AdamW)

    assert _group_weight_decay(optimizer.param_groups, model.backbone.weight) == pytest.approx(0.01)
    assert _group_weight_decay(optimizer.param_groups, model.head.weight) == pytest.approx(0.1)


def test_lazyconfig_sgd_lr_factor_scales_constructor_lr():
    model = BackboneModel()
    cfg = LazyConfig.load(str(REPO_ROOT / "common/optim.py"))
    cfg.SGD.params.model = model
    cfg.SGD.params.overrides = {"backbone": {"lr_factor": 0.1}}

    optimizer = instantiate(cfg.SGD)

    assert _group_lr(optimizer.param_groups, model.backbone.weight) == pytest.approx(0.002)
    assert _group_lr(optimizer.param_groups, model.head.weight) == pytest.approx(0.02)
