"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-19 10:55:22
Description: PyTorch Lightning Module for model wrapper
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import FSDPStrategy, ModelParallelStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from xmetai.config import LazyConfig, instantiate
from xmetai.core import DatasetEvaluators
from xmetai.core.defaults import apply_model_activation_checkpointing, create_fsdp2_model
from xmetai.recipes import recipe_path

__all__ = ["Lightning_Model_Wrapper"]


# results pattern: {key:{times:values, level1: values, ...}}
def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                if isinstance(vv, (Sequence, np.ndarray)) and not isinstance(vv, (str, bytes)):
                    vv = np.mean(vv)
                r[k + "_" + kk] = vv
        else:
            r[k] = v
    return r


class Lightning_Model_Wrapper(pl.LightningModule):
    def __init__(
        self,
        model_cfg: LazyConfig,
        evaluator_cfg: list[LazyConfig],
        lr: float = 0.0005,
        max_iter: int = 30000,
        warmup_rate: float = 0.01,
        train_cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["evaluator_cfg"])
        self.evaluator = DatasetEvaluators([instantiate(item) for item in evaluator_cfg])
        self.evaluator.reset()
        self.train_cfg = train_cfg if train_cfg is not None else LazyConfig.load(recipe_path("train")).train
        self._model_configured = False

    def configure_model(self) -> None:
        if self._model_configured:
            return
        self.model = instantiate(self.hparams.model_cfg)
        self.model.freeze(self.train_cfg.freeze)
        # Native weights must be loaded before AC and FSDP change module boundaries.
        path = self.train_cfg.init_checkpoint
        if path and path.endswith(".pth"):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"], strict=True)
        ac = OmegaConf.create(OmegaConf.to_container(self.train_cfg.ac, resolve=True), flags={"allow_objects": True})
        fsdp = OmegaConf.create(OmegaConf.to_container(self.train_cfg.fsdp, resolve=True), flags={"allow_objects": True})
        ac_mods, fsdp_mods = self.model.parallelize()
        if ac.mods is None:
            ac.mods = ac.default_mods if ac_mods is None else ac_mods
        if fsdp.mods is None:
            fsdp.mods = fsdp.default_mods if fsdp_mods is None else fsdp_mods
        apply_model_activation_checkpointing(self.model, ac)
        strategy = self.trainer.strategy
        if isinstance(strategy, ModelParallelStrategy):
            self.model = create_fsdp2_model(self.model, device_mesh=self.device_mesh["data_parallel"], **fsdp)
        elif isinstance(strategy, FSDPStrategy):
            # Lightning performs the FSDP1 wrapping after this hook, once defaults are resolved.
            strategy.kwargs["auto_wrap_policy"] = ModuleWrapPolicy(set(fsdp.mods))
        self._model_configured = True

    def forward(self, x):
        with torch.autocast(device_type=self.device.type, dtype=self.model.dtype, enabled=self.model.dtype in (torch.float16, torch.bfloat16)):
            results = self.model(x)
        return results

    def training_step(self, batch, batch_idx):
        loss_dict = self(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
            self.log("train_loss", losses, prog_bar=True)
        else:
            losses = sum(loss_dict.values())
            for k, v in loss_dict.items():
                self.log(k, v, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        return losses

    def on_validation_epoch_start(self) -> None:
        self.evaluator.reset()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        result = self(batch)
        self.evaluator.process(batch, result)
        return {}

    def on_validation_epoch_end(self) -> None:
        results = self.evaluator.evaluate()
        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            v = float(v)
            self.log(k, v, prog_bar=True if k.find("z500") > -1 else False, sync_dist=True)
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.evaluator.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        self.model.logger.info(f"lr: {self.hparams.get('lr', 0.0005)}")
        self.model.logger.info(f"max_iter: {self.hparams.get('max_iter', 1000)}")
        self.model.logger.info(f"warmup_rate: {self.hparams.get('warmup_rate', 0.01)}")
        optimizer = AdamW(self.parameters(), lr=self.hparams.get("lr", 0.0005))

        max_iter = self.hparams.get("max_iter", 1000)
        warmup_steps = int(max_iter * self.hparams.get("warmup_rate", 0.01))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # 线性 warmup
                return float(current_step) / float(max(1, warmup_steps))
            # warmup 后进入 cosine
            progress = float(current_step - warmup_steps) / float(max(1, max_iter - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # 按 step 更新
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
