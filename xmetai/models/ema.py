"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-14 17:32:55
Description: EMA (Exponential Moving Average) for model parameters.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import contextlib
import logging
from typing import ContextManager

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

__all__ = ["EMA"]


def is_model_fsdp(model: torch.nn.Module) -> bool:
    if isinstance(model, FSDP):
        return True
    for _, obj in model.named_children():
        if isinstance(obj, FSDP):
            return True
    return False


def _get_fsdp_context(model: torch.nn.Module) -> ContextManager[object]:
    if is_model_fsdp(model):
        return FSDP.summon_full_params(model)
    else:
        return contextlib.nullcontext()


class EMA(nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.logger = logging.getLogger(__name__)
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

        self.ema_params: dict[str, torch.Tensor] = {}
        self.tmp_params: dict[str, torch.Tensor] = {}
        with _get_fsdp_context(model):
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.ema_params[name] = p.clone()

    @torch.no_grad()
    def update_params(self, model: torch.nn.Module) -> None:
        """
        Update the EMA parameters with the current model parameters.
        This method should be called after load from checkpoint.
        """
        self.logger.info("Updating EMA parameters")
        with _get_fsdp_context(model):
            for name, param in model.named_parameters():
                if name in self.ema_params:
                    self.ema_params[name].copy_(param)
                else:
                    self.logger.warning(f"Parameter {name} not found in EMA parameters, skipping...")
        self.num_updates.fill_(0)  # type: ignore[operator]  # registered buffer

    @torch.no_grad()
    def forward(self, model: torch.nn.Module) -> None:
        _decay: torch.Tensor = self.decay  # type: ignore[assignment]
        decay_val: float = _decay.item()  # type: ignore[assignment,operator]
        # self.num_updates is a registered buffer of dtype int
        num_updates_tensor: torch.Tensor = self.num_updates  # type: ignore[assignment,operator]
        n = int(num_updates_tensor.item())
        if n >= 0:
            num_updates_tensor.add_(1)
            n += 1
            decay_val = min(decay_val, (1.0 + n) / (10.0 + n))
        with _get_fsdp_context(model):
            for name, param in model.named_parameters():
                if name in self.ema_params:
                    full_param = param.clone()
                    self.ema_params[name].mul_(decay_val).add_(full_param, alpha=(1.0 - decay_val))

    @torch.no_grad()
    def reverse(self, model: torch.nn.Module) -> None:
        self.logger.info("Reversing EMA parameters to model parameters")
        self.tmp_params = {}
        with _get_fsdp_context(model):
            for name, param in model.named_parameters():
                if name in self.ema_params:
                    self.tmp_params[name] = param.clone()
                    param.data.copy_(self.ema_params[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        self.logger.info("Restoring model parameters from EMA parameters")
        with _get_fsdp_context(model):
            for name, param in model.named_parameters():
                if name in self.tmp_params:
                    param.data.copy_(self.tmp_params[name])
        del self.tmp_params
