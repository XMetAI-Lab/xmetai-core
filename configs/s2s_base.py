#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file for training a SwinVRNN model on the ERA5 dataset.
This configuration file is designed to be used with the FanJiang framework.
"""

import os
import numpy as np
from omegaconf import OmegaConf
from xmetai import LazyCall as L
from xmetai import MODELS, METRICS, DATASETS, LazyConfig, build_test_loader, build_train_loader, get_data_info

######################################################################
# Global settings
######################################################################
# set the environment variable for PyTorch CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")
data_dir = os.environ.get("DATA_DIR", "/workspace/data/s2s.1950-2024.c76")
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/", os.path.basename(CONFIG_PATH)))

#####################################################################
# Training settings
#####################################################################
lr = float(os.environ.get("LR", 0.0005))  # get lr from environment variable, 1e-5 ~ 2e-4.
total_batch_size = 1
max_iter = 30 * 20000//total_batch_size # allow for 30 epochs with 20000 iterations per epoch

# learning rate scheduler settings
train = LazyConfig.load(os.path.join(ROOT_PATH, "common/train.py")).train
# optimizer settings for training
optimizer = LazyConfig.load(os.path.join(ROOT_PATH, "common/optim.py")).AdamW
# Don't use common/schedule.py for now.
scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).cosine_lr

load_from_pretrain = False
if os.path.exists(f'{output_dir}/model_final.pth'):
    train.init_checkpoint = f'{output_dir}/model_best.pth'
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 20  # teach for 20 steps with 80GB GPU memory
    output_dir += "/mstep/"
    lr = 1.28e-5
    max_iter = max_iter//3
    load_from_pretrain = True
    scheduler.scheduler.end_value = 1e-2
    #scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).const_lr

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
#test_names = ["z500", "t2m", "msl", "tp"]
test_chans = [out_names.index(n) for n in test_names]
test_frames = np.arange(1, 43).tolist()  # predict 1-42 days ahead
step_range = [max(test_frames)]

in_chans = len(in_names)
in_frames = 2
out_chans = in_chans
freq = 24  # 24 hours for 1 day

# model setting
depth_list = os.environ.get("DEPTH_LIST", "6,6,12,12")  # default depth list
depths = [int(x) for x in depth_list.split(",") if x.strip().isdigit()]
loss_weight = [0.5, 1.0]

######################################################################
# Data settings
######################################################################

dataloader = OmegaConf.create()

train_times = ("19500101", "20230228")
test_times = ("20230101", "20241231")
#test_times = ("20050101", "20250202")
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
    num_workers=8,
    pin_memory=True,
)

dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("ERA5"))(
        data_name=data_dir,
        hist_frames=in_frames,
        fcst_frames=test_frames,
        interval=-1,
        freq=freq,
        training=False,
        inference_only=False,
        years=test_times,
    ),
    total_batch_size=total_batch_size,
    num_workers=4,
    pin_memory=True,
)

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

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
        embed_types=["step"],
        model_name="s2s_pred.onnx",
        device=train.device_type
    )