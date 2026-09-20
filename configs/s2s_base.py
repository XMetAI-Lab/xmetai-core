#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file for training a SwinVRNN model on the ERA5 dataset.
This configuration file is designed to be used with the FanJiang framework.
"""

import os
from datetime import datetime

import numpy as np

from xmetai import (
    LazyConfig,
    get_data_info,
)
from xmetai.recipes import recipe_path

######################################################################
# Global settings
######################################################################
# set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")
data_dir = os.environ.get("DATA_DIR", "/workspace/data/s2s.1950-2024.c76")
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/", os.path.basename(CONFIG_PATH)))
cur_date = datetime.now().strftime("%Y%m%d")

#####################################################################
# Training settings
#####################################################################
lr = float(os.environ.get("LR", 0.0005))  # get lr from environment variable, 1e-5 ~ 2e-4.
total_batch_size = 1
max_iter = 30 * 20000 // total_batch_size  # allow for 30 epochs with 20000 iterations per epoch

# learning rate scheduler settings
train = LazyConfig.load(recipe_path("train")).train
# optimizer settings for training
optimizer = LazyConfig.load(recipe_path("optim")).AdamW
# Don't use the shared schedule recipe for now.
scheduler = LazyConfig.load(recipe_path("schedule")).cosine_lr

load_from_pretrain = False
if os.path.exists(f"{output_dir}/model_final.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 20  # teach for 20 steps with 80GB GPU memory
    output_dir += f"/mstep-{cur_date}/"
    lr = 1.28e-5
    max_iter = max_iter // 3
    load_from_pretrain = True
    scheduler.scheduler.end_value = 1e-2
    # scheduler = LazyConfig.load(recipe_path("schedule")).const_lr

train.activation_checkpointing = True
train.max_iter = max_iter
train.output_dir = output_dir
optimizer.lr = lr
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
# test_names = ["z500", "t2m", "msl", "tp"]
test_chans = [out_names.index(n) for n in test_names]
test_frames = np.arange(1, 43).tolist()  # predict 1-42 days ahead
step_range = [max(test_frames)]

in_chans = len(in_names)
in_frames = 2
out_chans = len(out_names)
freq = 24  # 24 hours for 1 day

# coefficients for pinn losses (override in experiment configs)
pinn_coeffs = {
    # mass continuity (disabled by default; enable explicitly if desired)
    "mass": 0.0,
    # geostrophic balance (enabled by default; adjust or disable in experiment configs if needed)
    "geo": 1e2,
    "hydro": 0.0,  # 2e-7
    "energy": 0.0,  # 1e-08
    "moisture": 1e6,
    "thermal": 0.0,
    "pv": 1e10,
}

depth_list = os.environ.get("DEPTH_LIST", "6,6,12,12")  # default depth list
depths = [int(x) for x in depth_list.split(",") if x.strip().isdigit()]

######################################################################
# Data settings
######################################################################
train_times = ("19500101", "20240131")
test_times = ("20240101", "20261231")
# test_times = ("20050101", "20261231")
dataloader = LazyConfig.load(recipe_path("dataloader"), "zarr_dataloader")(
    data_paths=[data_dir],
    train_times=train_times,
    test_times=test_times,
    eval_test_names=test_names,
    test_target_names=test_names,
    total_batch_size=total_batch_size,
)
if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

export_onnx = False
if export_onnx:
    from xmetai.data import make_onnx

    train.use_ddp = False
    train.use_fsdp = False
    train.use_fsdp2 = False
    train.activation_checkpointing = False
    dataloader.test.total_batch_size = 1
    dataloader.test.num_workers = 1
    dataloader.test.pin_memory = False
    dataloader.test.dataset.inference_only = True
    dataloader.test.dataset.fcst_frames = [1]

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
