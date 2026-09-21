"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-19 10:56:12
Description: Training script for PyTorch Lightning
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import sys
from typing import Any

import torch
from torch.distributed.fsdp import MixedPrecision
from typing_extensions import override

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        Callback,
        DeviceStatsMonitor,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy, ModelParallelStrategy

    from xmetai.config import LazyConfig, instantiate
    from xmetai.core import default_argument_parser, default_setup
    from xmetai.lightning import Lightning_Model_Wrapper
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you have installed the required dependencies and that your PYTHONPATH is set correctly.")
    sys.exit(1)


class MemoryCleaner(Callback):
    def on_val_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
        pass


class AccumulateGradientCallback(Callback):
    def __init__(self, scheduling: dict[int, int]):
        super().__init__()
        minimal_step = min(scheduling.keys())
        if minimal_step < 0:
            raise IndexError(f"Steps indexing from 1, step {minimal_step} cannot be interpreted correct")
        if minimal_step != 0:  # if user didn't define first step accumulation factor
            scheduling.update({0: 1})
        self.scheduling = scheduling
        self.steps = sorted(scheduling.keys())

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, step: int) -> int:
        accumulate_grad_batches = 1
        for iter_step in reversed(self.steps):
            if step >= iter_step:
                accumulate_grad_batches = self.scheduling[iter_step]
                break
        return accumulate_grad_batches

    @override
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(batch_idx)


class EMACallback(Callback):
    """
    Custom EMA Callback that only copies parameters' tensor data.
    This avoids deepcopying the entire model which could raise PicklingError
    if the model contains unpicklable objects like Python loggers.
    """

    def __init__(self, decay=0.99):
        self.decay = decay
        self.ema_state = {}

    def on_fit_start(self, trainer, pl_module):
        # 初始化 EMA 参数
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 更新 EMA 参数
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.ema_state[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        # 评估时用 EMA 参数
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.ema_state:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.ema_state[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        # 恢复原始参数
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


def main(args):
    torch.set_float32_matmul_precision("medium")
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if not hasattr(cfg.scheduler, "warmup_length"):
        cfg.scheduler.warmup_length = 0.01
    train_loader = instantiate(cfg.dataloader.train)
    val_loader = instantiate(cfg.dataloader.test)

    # Configure callbacks
    val_metric = cfg.train.val_metric.replace("/", "_")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.output_dir,
        filename="model-{step:07d}-{" + val_metric + ":.1f}",
        save_top_k=1,
        monitor=val_metric,
        mode="min",
        every_n_train_steps=1000,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    train_bar = TQDMProgressBar(refresh_rate=1, leave=True)
    accumulate_callback = AccumulateGradientCallback(scheduling={0: 1, 100: 2, 400: 4})

    # Configure trainer
    trainer_args = {
        "max_steps": cfg.train.max_iter,
        "val_check_interval": cfg.train.eval_period,
        "log_every_n_steps": cfg.train.log_period,
        "callbacks": [
            MemoryCleaner(),
            checkpoint_callback,
            lr_monitor,
            train_bar,
            accumulate_callback,
            DeviceStatsMonitor(),
        ],
        "accelerator": ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        "devices": args.num_gpus,
        "num_nodes": args.num_machines,
        "use_distributed_sampler": False,
        "strategy": getattr(cfg.train, "strategy", "ddp" if args.num_machines > 1 else "auto"),
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0.1,
        "precision": "bf16-mixed",
        "default_root_dir": cfg.train.output_dir,
    }

    # configure_model applies AC once; Lightning owns distributed setup and checkpoint IO.
    current_strategy = trainer_args["strategy"]
    if cfg.train.use_fsdp and cfg.train.use_fsdp2:
        raise ValueError("Select only one of train.use_fsdp and train.use_fsdp2")
    if cfg.train.use_fsdp2:
        trainer_args["strategy"] = ModelParallelStrategy(data_parallel_size=args.num_gpus * args.num_machines, tensor_parallel_size=1)
        trainer_args["precision"] = "32-true"
    elif cfg.train.use_fsdp:
        precision = cfg.train.fsdp.precision
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        fsdp_mp = MixedPrecision(
            param_dtype=dtype[precision.param_dtype], reduce_dtype=dtype[precision.reduce_dtype], buffer_dtype=dtype[precision.buffer_dtype]
        )
        trainer_args["strategy"] = FSDPStrategy(cpu_offload=cfg.train.fsdp.cpu_offload, mixed_precision=fsdp_mp)
    elif current_strategy == "ddp" or cfg.train.use_ddp:
        trainer_args["strategy"] = DDPStrategy(find_unused_parameters=cfg.train.ddp.find_unused_parameters)

    # 添加 EMA 回调
    use_ema = getattr(cfg.train, "use_ema", False)
    if use_ema:
        trainer_args["callbacks"].append(EMACallback(decay=0.99))

    trainer = pl.Trainer(**trainer_args)
    model = Lightning_Model_Wrapper(
        cfg.model, cfg.dataloader.evaluator, lr=cfg.optimizer.lr, max_iter=cfg.train.max_iter, warmup_rate=cfg.scheduler.warmup_length, train_cfg=cfg.train
    )
    # Lightning restores .ckpt files/directories after configure_model and strategy setup.
    ckpt_path = cfg.train.init_checkpoint if cfg.train.init_checkpoint and not cfg.train.init_checkpoint.endswith(".pth") else None
    if args.eval:
        trainer.test(model, dataloaders=val_loader, ckpt_path=ckpt_path, weights_only=False)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path, weights_only=False)


def cli():
    args = default_argument_parser().parse_args()
    main(args)


if __name__ == "__main__":
    cli()
