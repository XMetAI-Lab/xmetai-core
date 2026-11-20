#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training configuration for six-hour FENGQ15 precipitation correction."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from xmetai import DATASETS, METRICS, MODELS, LazyConfig, build_test_loader
from xmetai import LazyCall as L
from xmetai.data.weather.prcpcorrect_dataset import build_prcp_train_loader
from xmetai.recipes import recipe_path


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32,expandable_segments:True")

XMETAI_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = XMETAI_ROOT.parent

distributed_mode = os.environ.get("DISTRIBUTED_MODE", "none").strip().lower()
if distributed_mode not in {"none", "ddp", "fsdp", "fsdp2"}:
    raise ValueError("DISTRIBUTED_MODE must be one of none, ddp, fsdp, or fsdp2.")
world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
if world_size > 1 and distributed_mode == "none":
    raise ValueError("DISTRIBUTED_MODE must be ddp, fsdp, or fsdp2 when multiple processes are launched.")

data_root = Path(os.environ.get("PRCP_DATA_ROOT", PROJECT_ROOT / "Dataset" / "Dataset_FENGQ15_CMPAS")).expanduser().resolve()
fengq_zarr_path = Path(os.environ.get("FENGQ_ZARR_PATH", data_root / "FENGQ15.zarr")).expanduser().resolve()
cmpas_zarr_path = Path(os.environ.get("CMPAS_ZARR_PATH", data_root / "CMPAS_6h.zarr")).expanduser().resolve()
event_manifest_path = Path(os.environ.get("PRCP_EVENT_MANIFEST", data_root / "prcp_event_train.json")).expanduser().resolve()
input_min_path = Path(os.environ.get("FENGQ_MIN_PATH", data_root / "FENGQ15_min_20220401_20250427.npy")).expanduser().resolve()
input_max_path = Path(os.environ.get("FENGQ_MAX_PATH", data_root / "FENGQ15_max_20220401_20250427.npy")).expanduser().resolve()
explicit_output_dir = os.environ.get("OUTPUT_DIR")
run_name = os.environ.get("RUN_NAME")
if run_name is None:
    if (world_size > 1 or distributed_mode != "none") and explicit_output_dir is None:
        raise ValueError("Distributed training requires a shared RUN_NAME or OUTPUT_DIR.")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_root = Path(os.environ.get("OUTPUT_ROOT", XMETAI_ROOT / "output" / "prcpcorrect_unet3d")).expanduser().resolve()
output_dir = Path(explicit_output_dir or output_root / run_name).expanduser().resolve()
if rank == 0 and output_dir.exists() and not output_dir.is_dir():
    raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
if rank == 0 and output_dir.exists() and any(output_dir.iterdir()) and not _env_flag("ALLOW_OUTPUT_REUSE", False):
    raise FileExistsError(f"Output directory is not empty: {output_dir}. Set ALLOW_OUTPUT_REUSE=1 to reuse it.")

train_split_csv = os.environ.get("PRCP_TRAIN_CSV")
val_split_csv = os.environ.get("PRCP_VAL_CSV")
# FENGQ15 stores contain April-October initializations only. Three-day gaps
# keep the twelve six-hour observation windows disjoint across splits.
train_start = os.environ.get("TRAIN_START", "2022-04-01 08:00:00")
train_end = os.environ.get("TRAIN_END", "2025-04-27 20:00:00")
val_start = os.environ.get("VAL_START", "2025-05-01 08:00:00")
val_end = os.environ.get("VAL_END", "2025-07-28 20:00:00")

lr = float(os.environ.get("LR", "0.0005"))
total_batch_size = int(os.environ.get("TOTAL_BATCH_SIZE", "8"))
eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
num_workers = int(os.environ.get("PRCP_NUM_WORKERS", "8"))
pin_memory = _env_flag("PRCP_PIN_MEMORY", True)
seed = int(os.environ.get("SEED", "42"))
base_channels = int(os.environ.get("BASE_CHANNELS", "128"))
precipitation_scale = float(os.environ.get("PRECIPITATION_SCALE", "1.0"))
physical_loss_weight = float(os.environ.get("PHYSICAL_LOSS_WEIGHT", "0.1"))
physical_loss_thresholds = (10.0, 25.0, 50.0)
physical_loss_weights = (1.0, 4.0, 4.0, 8.0)
event_loss_weight = float(os.environ.get("EVENT_LOSS_WEIGHT", "0.05"))
event_loss_temperature = float(os.environ.get("EVENT_LOSS_TEMPERATURE", "0.25"))
event_loss_alpha = float(os.environ.get("EVENT_LOSS_ALPHA", "1.0"))
event_loss_beta = float(os.environ.get("EVENT_LOSS_BETA", "1.25"))
if total_batch_size < world_size or total_batch_size % world_size != 0:
    raise ValueError(f"TOTAL_BATCH_SIZE={total_batch_size} must be at least and divisible by world_size={world_size}.")
if eval_batch_size < world_size or eval_batch_size % world_size != 0:
    raise ValueError(f"EVAL_BATCH_SIZE={eval_batch_size} must be at least and divisible by world_size={world_size}.")
if num_workers < 0:
    raise ValueError("PRCP_NUM_WORKERS must be non-negative.")

train = LazyConfig.load(recipe_path("train")).train
optimizer = LazyConfig.load(recipe_path("optim")).Adam
scheduler = LazyConfig.load(recipe_path("schedule")).cosine_lr

train.output_dir = os.fspath(output_dir)
train.init_checkpoint = os.environ.get("INIT_CHECKPOINT", "")
train.max_iter = int(os.environ.get("MAX_ITER", "8100"))
train.eval_period = int(os.environ.get("EVAL_PERIOD", "80"))
train.log_period = int(os.environ.get("LOG_PERIOD", "20"))
train.checkpointer.period = int(os.environ.get("CHECKPOINT_PERIOD", "400"))
train.val_metric = "Selection/csi_mean6_error"
train.device_type = os.environ.get("DEVICE", "cuda")
train.seed = seed
train.use_ddp = distributed_mode == "ddp"
train.use_fsdp = distributed_mode == "fsdp"
train.use_fsdp2 = distributed_mode == "fsdp2"
train.ac.enable = False
train.use_ema = _env_flag("USE_EMA", True)

optimizer.lr = lr

model = L(MODELS.get("PrcpCorrectionUNet3D"))(
    in_channels=30,
    base_channels=base_channels,
    precipitation_scale=precipitation_scale,
    physical_loss_weight=physical_loss_weight,
    physical_loss_thresholds=physical_loss_thresholds,
    physical_loss_weights=physical_loss_weights,
    event_loss_weight=event_loss_weight,
    event_loss_temperature=event_loss_temperature,
    event_loss_alpha=event_loss_alpha,
    event_loss_beta=event_loss_beta,
    input_min=os.fspath(input_min_path),
    input_max=os.fspath(input_max_path),
    precision=os.environ.get("PRECISION", "fp32"),
    device=train.device_type,
)


def _dataset_kwargs(split_csv: Optional[str], start_time: str, end_time: str) -> dict:
    common = {
        "fengq_zarr_path": os.fspath(fengq_zarr_path),
        "cmpas_zarr_path": os.fspath(cmpas_zarr_path),
        "input_min": os.fspath(input_min_path),
        "input_max": os.fspath(input_max_path),
    }
    if split_csv:
        return {**common, "split_csv": split_csv}
    return {**common, "start_time": start_time, "end_time": end_time}


eval_times = list(range(6, 73, 6))
dataloader = {
    "train": L(build_prcp_train_loader)(
        dataset=L(DATASETS.get("PrcpCorrectionUNet3DDataset"))(**_dataset_kwargs(train_split_csv, train_start, train_end)),
        total_batch_size=total_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        event_manifest=os.fspath(event_manifest_path),
    ),
    "test": L(build_test_loader)(
        dataset=L(DATASETS.get("PrcpCorrectionUNet3DDataset"))(**_dataset_kwargs(val_split_csv, val_start, val_end)),
        total_batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ),
    "evaluator": [
        L(METRICS.get("PrcpMaskedRMSE"))(test_names=["tp"], test_times=eval_times),
        L(METRICS.get("PrcpMaskedCSI"))(test_names=["tp"], test_times=eval_times),
    ],
}

config_summary = {
    "data_root": os.fspath(data_root),
    "fengq_zarr_path": os.fspath(fengq_zarr_path),
    "cmpas_zarr_path": os.fspath(cmpas_zarr_path),
    "input_min_path": os.fspath(input_min_path),
    "input_max_path": os.fspath(input_max_path),
    "output_dir": os.fspath(output_dir),
    "init_checkpoint": train.init_checkpoint,
    "device": train.device_type,
    "distributed_mode": distributed_mode,
    "total_batch_size": total_batch_size,
    "eval_batch_size": eval_batch_size,
    "num_workers": num_workers,
    "seed": seed,
    "base_channels": base_channels,
    "precipitation_scale": precipitation_scale,
    "physical_loss_weight": physical_loss_weight,
    "physical_mse_weight": 0.0075,
    "physical_loss_thresholds_mm_6h": physical_loss_thresholds,
    "physical_loss_weights": physical_loss_weights,
    "sampling": "repeat_factor_training_sampler",
    "event_manifest_path": os.fspath(event_manifest_path),
    "loss": "log_smooth_l1_plus_balanced_physical_l1_l2" + ("_plus_joint_event4" if event_loss_weight else ""),
    "event_thresholds_mm_6h": (5.0, 10.0, 25.0, 50.0),
    "event_loss_weight": event_loss_weight,
    "event_loss_temperature": event_loss_temperature,
    "event_loss_alpha": event_loss_alpha,
    "event_loss_beta": event_loss_beta,
    "event_loss_reduction": "equal_mean_of_four_batch_pooled_soft_event_losses_times_grid_cells",
    "event_loss_empty_target": "mean_soft_exceedance_over_valid_cells",
    "observation_mask": "valid_mask",
    "cmpas_schema": "xmetai.prcp_correction.cmpas_6h_v2",
    "loss_reduction": "valid_spatial_mean_times_grid_cells",
    "physical_loss_reduction": "valid_weighted_spatial_mean_times_grid_cells",
    "selection_metric": train.val_metric,
    "selection_rule": "minimize_1_minus_equal_mean_of_six_global_valid_cell_CSIs",
    "selection_thresholds_mm_6h": (0.1, 1.0, 5.0, 10.0, 25.0, 50.0),
    "selection_weights": (1.0 / 6,) * 6,
    "selection_checkpoint": "model_best.pth",
    "training_entry_point": "xmetai.cli.train",
    "forecast_hours": eval_times,
    "train_range": (train_start, train_end) if not train_split_csv else train_split_csv,
    "val_range": (val_start, val_end) if not val_split_csv else val_split_csv,
}
