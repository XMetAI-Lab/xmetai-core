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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")
data_dirs = [f for f in os.environ["DATA_DIR"].split(":") if f]  # support multiple output directories
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/", os.path.basename(CONFIG_PATH)))
cur_date = datetime.now().strftime("%Y%m%d")

#####################################################################
# Training settings
#####################################################################
lr = float(os.environ.get("LR", 0.0005))  # get lr from environment variable, 1e-5 ~ 2e-4.
total_batch_size = 1
max_iter = 30 * 20000 // total_batch_size  # allow for 30 epochs with 20000 iterations per epoch
train_times = ("19500101", "20230228")

# learning rate scheduler settings
train = LazyConfig.load(recipe_path("train")).train
# optimizer settings for training
optimizer = LazyConfig.load(recipe_path("optim")).AdamW
# Don't use the shared schedule recipe for now.
scheduler = LazyConfig.load(recipe_path("schedule")).cosine_lr

load_from_pretrain = False
if os.path.exists(f"{output_dir}/model_final.pth"):
    train_times = ("20010101", "20230228")
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
data_info = get_data_info(data_dirs)
in_names = data_info["in_names"]
out_names = data_info["out_names"]
logid = data_info["logid"]
accumid = data_info["accumid"]
maskid = data_info["maskid"]
landid = data_info["landid"]
buffers = data_info["buffers"]

test_names = ["z500", "u200", "u850", "msl", "t2m", "sst", "ttr", "tp", "stl1", "swvl1"]
# test_names = ["z500", "t2m", "msl", "tp"]
test_chans = [out_names.index(n) for n in test_names]
test_frames = np.arange(1, 43).tolist()  # predict 1-42 days ahead
step_range = [max(test_frames)]

in_chans = len(in_names)
in_frames = 2
out_chans = in_chans
freq = 24  # 24 hours for 1 day

depth_list = os.environ.get("DEPTH_LIST", "6,6,12,12")  # default depth list
depths = [int(x) for x in depth_list.split(",") if x.strip().isdigit()]

######################################################################
# Data settings
######################################################################


test_times = ("20230101", "20241231")
# test_times = ("20050101", "20250202")

dataloader = LazyConfig.load(recipe_path("dataloader"), "zarr_dataloader")(
    data_paths=data_dirs,
    train_times=train_times,
    test_times=test_times,
    eval_test_names=test_names,
    total_batch_size=total_batch_size,
)
if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

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
        embed_types=["step"],
        model_name="land_pred.onnx",
        device=train.device_type,
    )
