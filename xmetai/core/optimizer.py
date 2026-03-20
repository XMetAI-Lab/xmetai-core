# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import torch

from xmetai.config import CfgNode

from .scheduler import (
    CosineParamScheduler,
    CyclicScheduler,
    LRMultiplier,
    MultiStepParamScheduler,
    WarmupParamScheduler,
)

__all__ = [
    "build_lr_scheduler",
    "build_optimizer",
    "get_default_optimizer_params",
    "gradient_clipping",
    "get_muon_params_group",
]


_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):  # type: ignore[no-untyped-def]
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):  # type: ignore[no-untyped-def]
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert per_param_clipper is None or global_clipper is None, "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(optimizer_type, per_param_clipper=grad_clipper)
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """

    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        overrides=cfg.optimizer.overrides,
    )

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(  # type: ignore[call-arg]
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif optimizer_type == "ADAM":
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(  # type: ignore[call-arg]
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    elif optimizer_type == "ADAMW":
        return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(  # type: ignore[call-arg]
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif optimizer_type == "AdamW8bit":
        import bitsandbytes as bnb

        return maybe_add_gradient_clipping(cfg, bnb.optim.AdamW8bit)(  # type: ignore[call-arg]
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")


def gradient_clipping(optimizer: Type[torch.optim.Optimizer]):  # type: ignore[no-untyped-def]

    def grad_clipper(p):
        torch.nn.utils.clip_grad_norm_(p, 1)

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                grad_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def get_default_optimizer_params(  # type: ignore[no-untyped-def]
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    logger = logging.getLogger(__name__)

    if overrides is None:
        overrides = {}

    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        module_name = module_name.replace("_checkpoint_wrapped_module.", "")

        for param_name, value in module.named_parameters(recurse=False):

            if not value.requires_grad:
                logger.info(f"freeze {module_name}.{param_name}")
                continue

            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            # Special handling for LoRA parameters
            # LoRA layers can typically use higher learning rates
            if "lora_" in param_name:
                # LoRA parameters can use 2x learning rate for faster adaptation
                if base_lr is not None:
                    hyperparams["lr"] = base_lr * 2.0
                # LoRA parameters typically benefit from less weight decay
                if weight_decay is not None:
                    hyperparams["weight_decay"] = weight_decay * 0.1
                logger.debug(f"LoRA param {module_name}.{param_name}: lr={hyperparams.get('lr', 'N/A')}, wd={hyperparams.get('weight_decay', 'N/A')}")
            elif isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            elif value.dim() <= 1 and weight_decay_norm is not None:
                # 1D parameters are typically biases and normalization weights
                hyperparams["weight_decay"] = weight_decay_norm
            elif hasattr(module, "weight") and param_name == "weight" and "norm" in module.__class__.__name__.lower() and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            params.append({"params": [value], **hyperparams})

    return params


def get_muon_params_group(  # type: ignore[no-untyped-def]
    model: torch.nn.Module,
    base_lr: Optional[float] = 5e-4,
    muon_lr: Optional[float] = 0.02,
):
    # we shoud use muon for conv & mean/var, check them with params name
    distribute_names = ["cov", "mean", "var", "diag"]
    hidden_weights = [p for p in model.parameters() if len(p.shape) >= 4 and p.requires_grad]
    distribution_weights_names = [n for n, _ in model.named_parameters() if any(name in n for name in distribute_names) and "weight" in n]
    distribution_weights = [p for n, p in model.named_parameters() if n in distribution_weights_names]
    other_weights = [p for n, p in model.named_parameters() if p.requires_grad and len(p.shape) < 4 and n not in distribution_weights_names]
    param_groups = [
        dict(
            params=hidden_weights + distribution_weights,
            use_muon=True,
            lr=muon_lr,
            weight_decay=0.01,
        ),
        dict(
            params=other_weights,
            use_muon=False,
            lr=base_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        ),
    ]
    return param_groups


def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    ret = defaultdict(dict)  # type: ignore[var-annotated]
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.
    params = _expand_param_groups(params)
    # re-group all parameter groups by their hyperparams
    groups = defaultdict(list)
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = param_values
        ret.append(cur)
    return ret


def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "CyclicLR":
        sched = CyclicScheduler(
            optimizer,
            cfg.SOLVER.BASE_LR,
            anneal_decay=cfg.SOLVER.ANNEAL_DECAY,
            anneal_iters=cfg.SOLVER.ANNEAL_ITERS,
        )
        return sched

    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning("SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. " "These values will be ignored.")
        sched = MultiStepParamScheduler(  # type: ignore[assignment]
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        sched = CosineParamScheduler(1, 0)  # type: ignore[assignment]
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(  # type: ignore[assignment]
        sched,  # type: ignore[arg-type]
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)  # type: ignore[arg-type]
