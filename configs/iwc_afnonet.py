#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
from omegaconf import OmegaConf

from xmetai import (
    DATASETS,
    METRICS,
    MODELS,
    LazyConfig,
    build_test_loader,
    build_train_loader,
    get_data_info,
)
from xmetai import LazyCall as L

######################################################################
# Global settings
######################################################################
# set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")
data_dir = os.environ.get("DATA_DIR", "/workspace/data/era5.1979-2024.c78.p25.h6")
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/", os.path.basename(CONFIG_PATH)))

#####################################################################
# Training settings
#####################################################################
lr = float(os.environ.get("LR", 0.0005))  # get lr from environment variable, 1e-5 ~ 2e-4.
total_batch_size = 1
epochs = 30
iters_per_epoch = 60000

# learning rate scheduler settings
train = LazyConfig.load(os.path.join(ROOT_PATH, "common/train.py")).train
# optimizer settings for training
optimizer = LazyConfig.load(os.path.join(ROOT_PATH, "common/optim.py")).AdamW
# Don't use common/schedule.py for now.
scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).cosine_lr

load_from_pretrain = False
if os.path.exists(f"{output_dir}/model_final.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 4
    output_dir += "/mstep/"
    lr = 1.28e-5
    epochs = epochs // 3
    load_from_pretrain = True
    scheduler.scheduler.end_value = 1e-2
    # scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).const_lr

train.max_iter = epochs * iters_per_epoch // total_batch_size  # max_steps with 60000 iterations per epoch
train.output_dir = output_dir
train.activation_checkpointing = True
train.use_fsdp2 = True
train.use_fsdp = False
train.use_ddp = False
# train.eval_period = iters_per_epoch//total_batch_size  # eval for each epoch
optimizer.lr = lr

model_name = "AFNONet"

######################################################################
# Model settings
######################################################################
data_info = get_data_info([data_dir])
in_names = data_info["in_names"]
out_names = data_info["out_names"]
logid = data_info["logid"]
accumid = data_info["accumid"]
maskid = data_info["maskid"]
buffers = data_info["buffers"]

test_names = ["z500", "u200", "u850", "msl", "t2m", "sst", "ttr", "tp"]
test_names = [ch for ch in test_names if ch in out_names]
test_chans = [out_names.index(n) for n in test_names]

test_frames = np.arange(1, 33).tolist()  # predict 1-8(32 * 6 / 24) days ahead with 6-hour frequency
step_range = [max(test_frames)]

in_chans = len(in_names)
in_frames = 2
out_chans = len(out_names)
freq = 6

embed_dim = 48
img_size = (120, 240)
patch_size = (5, 5)
depth = 12
mlp_ratio = 4.0
drop_rate = 0.1
drop_path_rate = 0.0
num_blocks = 16
sparsity_threshold = 0.01
hard_thresholding_fraction = 1.0

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    img_size=img_size,
    patch_size=patch_size,
    depth=depth,
    mlp_ratio=mlp_ratio,
    drop_rate=drop_rate,
    drop_path_rate=drop_path_rate,
    num_blocks=num_blocks,
    sparsity_threshold=sparsity_threshold,
    hard_thresholding_fraction=hard_thresholding_fraction,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    freq=freq,
    accumid=logid,
    logid=logid,
    maskid=[],  # maskid,
    zero_accum=False,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=1,
    save_dir=output_dir + "/pred/",
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
)

model.buffers = buffers
model.precision = "bf16"

######################################################################
# Data settings
######################################################################
dataloader = OmegaConf.create()

train_times = ("19500101-00", "20231231-18")
test_times = ("20240101-18", "20251231-18")
# test_times = ("20050101-18", "20250502-18")
dataloader.train = L(build_train_loader)(
    dataset=L(DATASETS.get("ERA5"))(
        data_name=data_dir,
        hist_frames=in_frames,
        fcst_frames=1,
        interval=1,
        freq=freq,
        training=True,
        years=train_times,
    ),
    total_batch_size=total_batch_size,
    num_workers=2,
    pin_memory=True,
)

dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("ERA5"))(
        data_name=data_dir,
        hist_frames=in_frames,
        fcst_frames=test_frames,
        interval=16,
        freq=freq,
        training=False,
        inference_only=False,
        years=test_times,
    ),
    total_batch_size=total_batch_size,
    num_workers=2,
    pin_memory=False,
)

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
    model.alpha = 1.1

# for evaluation with mse.
eval_times = [int(t * freq) for t in test_frames]
dataloader.evaluator = [
    L(METRICS.get("MSE"))(
        test_names=test_names,
        test_times=eval_times,
    ),
]

export_onnx = False
if export_onnx:
    from xmetai.data import make_onnx

    train.use_ddp = False
    train.use_fsdp = False
    train.activation_checkpointing = False
    dataloader.test.total_batch_size = 1
    make_onnx(
        model=train,
        dataloader_cfg=dataloader.test,
        mean=buffers.get("mean", None),
        std=buffers.get("std", None),
        in_names=in_names,
        output_dir=output_dir,
        # embed_types=["step"],
        embed_types=["step", "hour", "doy"],
        model_name="s2s_pred.onnx",
        device=train.device_type,
    )
