"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-19 10:55:22
Description: PyTorch Lightning Module for model wrapper
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import math
from typing import Iterable, Mapping

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from xmetai.config import LazyConfig, instantiate
from xmetai.core import DatasetEvaluators

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
                if isinstance(vv, Iterable):  # use mean for list
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["evaluator_cfg"])
        self.evaluator = DatasetEvaluators([instantiate(item) for item in evaluator_cfg])
        self.evaluator.reset()
        pass

    def configure_model(self) -> None:
        model_cfg = self.hparams.model_cfg  # type: ignore[attr-defined]
        self.model = instantiate(model_cfg)
        return

    def forward(self, x):
        with torch.autocast(device_type=self.device.type, dtype=self.model.dtype):
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
