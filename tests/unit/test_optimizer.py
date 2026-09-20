import copy

import pytest
import torch

from xmetai.config import LazyConfig
from xmetai.config.instantiate import instantiate
from xmetai.core.optimizer import get_default_optimizer_params
from xmetai.recipes import recipe_path


class BackboneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


class FSDPStyleWrapper(torch.nn.Module):
    """Minimal wrapper matching FSDP1's module naming behavior."""

    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module


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
    cfg = LazyConfig.load(recipe_path("optim"))
    cfg.AdamW.params.model = model
    cfg.AdamW.params.overrides = {"backbone": {"weight_decay_factor": 0.1}}

    optimizer = instantiate(cfg.AdamW)

    assert _group_weight_decay(optimizer.param_groups, model.backbone.weight) == pytest.approx(0.01)
    assert _group_weight_decay(optimizer.param_groups, model.head.weight) == pytest.approx(0.1)


def test_lazyconfig_sgd_lr_factor_scales_constructor_lr():
    model = BackboneModel()
    cfg = LazyConfig.load(recipe_path("optim"))
    cfg.SGD.params.model = model
    cfg.SGD.params.overrides = {"backbone": {"lr_factor": 0.1}}

    optimizer = instantiate(cfg.SGD)

    assert _group_lr(optimizer.param_groups, model.backbone.weight) == pytest.approx(0.002)
    assert _group_lr(optimizer.param_groups, model.head.weight) == pytest.approx(0.02)


def test_lr_factor_scales_matching_parameter_groups_through_data_parallel_wrapper():
    model = BackboneModel()
    wrapped_model = torch.nn.DataParallel(model)

    params = get_default_optimizer_params(
        wrapped_model,
        base_lr=5e-4,
        overrides={"backbone": {"lr_factor": 0.1}},
    )

    assert _group_lr(params, model.backbone.weight) == pytest.approx(5e-5)
    assert _group_lr(params, model.head.weight) == pytest.approx(5e-4)


def test_lr_factor_scales_matching_parameter_groups_through_nested_fsdp_wrappers():
    model = BackboneModel()
    backbone = model.backbone
    model.backbone = FSDPStyleWrapper(backbone)
    wrapped_model = FSDPStyleWrapper(model)

    params = get_default_optimizer_params(
        wrapped_model,
        base_lr=5e-4,
        overrides={"backbone": {"lr_factor": 0.1}},
    )

    assert _group_lr(params, backbone.weight) == pytest.approx(5e-5)
    assert _group_lr(params, model.head.weight) == pytest.approx(5e-4)


def test_merged_adamw_groups_preserve_updates_and_roundtrip():
    model = torch.nn.Sequential(BackboneModel(), BackboneModel())
    reference = copy.deepcopy(model)
    kwargs = dict(base_lr=5e-4, weight_decay=0.1, weight_decay_norm=0.0, overrides={"0.backbone": {"lr_factor": 0.5}})
    groups = get_default_optimizer_params(model, **kwargs)
    assert len(groups) == 4
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))
    assert len({id(p) for group in groups for p in group["params"]}) == len(list(model.parameters()))
    singleton_groups = [{**group, "params": [param]} for group in get_default_optimizer_params(reference, **kwargs) for param in group["params"]]
    optimizer = torch.optim.AdamW(groups, foreach=True)
    reference_optimizer = torch.optim.AdamW(singleton_groups, foreach=True)
    for step in range(3):
        for param, ref in zip(model.parameters(), reference.parameters()):
            param.grad = torch.full_like(param, 0.25 + step)
            ref.grad = param.grad.clone()
        optimizer.step()
        reference_optimizer.step()
        for param, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param, ref)
    restored = torch.optim.AdamW(get_default_optimizer_params(model, **kwargs), foreach=True)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for param in model.parameters():
        torch.testing.assert_close(restored.state[param], optimizer.state[param])
