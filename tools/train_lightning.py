'''
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-19 10:56:12
Description: Training script for PyTorch Lightning
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved. 
'''
import os
import sys
from typing import Dict, Any
from typing_extensions import override
import torch
import pytorch_lightning as pl
from torch.distributed.fsdp import MixedPrecision
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, Callback, DeviceStatsMonitor


# set python path from ../
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_PATH + "/../")
sys.path.append(CUR_PATH + "/../configs/")

try:
    from xmetai.lightning import Lightning_Model_Wrapper
    from xmetai.core import default_argument_parser, default_setup
    from xmetai.config import LazyConfig, instantiate
    from xmetai.models.transformer import SwinBlock
    from xmetai.layers import MambaVisionLayer
except ImportError as e:
    print(f"ImportError: {e}. You can do this by running `pip install -e .` in the root directory.")
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
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(batch_idx)


class EMACallback(Callback):
    def __init__(self, decay=0.999):
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
    torch.set_float32_matmul_precision('medium')
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if not hasattr(cfg.scheduler, 'warmup_length'):
        cfg.scheduler.warmup_length = 0.01
    train_loader = instantiate(cfg.dataloader.train)
    val_loader = instantiate(cfg.dataloader.test)

    # Configure callbacks
    val_metric = cfg.train.val_metric.replace("/", "_")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.output_dir,
        filename="model-{step:07d}-{"+val_metric+":.1f}",
        save_top_k=1,
        monitor=val_metric,
        mode="min",
        every_n_train_steps=1000,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    train_bar = TQDMProgressBar()
    accumulate_callback = AccumulateGradientCallback(scheduling={0: 1, 100: 2, 400: 4})

    # Configure trainer
    trainer_args = {
        "max_steps": cfg.train.max_iter,
        "val_check_interval": cfg.train.eval_period,
        "log_every_n_steps": cfg.train.log_period,
        "callbacks": [MemoryCleaner(), checkpoint_callback, lr_monitor, train_bar, accumulate_callback, DeviceStatsMonitor()],
        "accelerator": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        "devices": args.num_gpus,
        "num_nodes": args.num_machines,
        "use_distributed_sampler": False,
        "strategy": "ddp" if args.num_machines > 1 else "auto",
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0.1,
        "precision": "bf16-mixed",
        "default_root_dir": cfg.train.output_dir,
    }

    cfg.train.use_fsdp = True
    # 启用 activation_checkpointing
    if cfg.train.use_fsdp:
        fsdp_mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        trainer_args["strategy"] = FSDPStrategy(
            cpu_offload=False,
            mixed_precision=fsdp_mp,
            activation_checkpointing_policy={
                MambaVisionLayer,
                SwinBlock,
                torch.nn.Conv2d,
                torch.nn.TransformerEncoderLayer,
                torch.nn.ConvTranspose2d,
                torch.nn.Linear,
            },
        )

    # 添加 EMA 回调
    if cfg.train.use_ema:
        trainer_args["callbacks"].append(EMACallback(decay=0.99))

    trainer = pl.Trainer(**trainer_args)
    model = Lightning_Model_Wrapper(cfg.model, cfg.dataloader.evaluator, lr=cfg.optimizer.lr,
                                    max_iter=cfg.train.max_iter, warmup_rate=cfg.scheduler.warmup_length)
    if os.path.exists(cfg.train.init_checkpoint):
        if cfg.train.init_checkpoint.endswith('.pth'):
            state_dict = torch.load(cfg.train.init_checkpoint, map_location="cpu", weights_only=False)
            model.model.load_state_dict(state_dict["model"], strict=True)
        elif cfg.train.init_checkpoint.endswith('.ckpt'):
            model = model.__class__.load_from_checkpoint(cfg.train.init_checkpoint)

    if args.eval:
        trainer.test(model, dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
