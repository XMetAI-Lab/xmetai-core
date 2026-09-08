#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file for training a ENS model on the ERA5 dataset.
This configuration file is designed to be used for Weather/Seasonal Forecasting.
"""

import os
from datetime import datetime

import numpy as np

from xmetai import (
    METRICS,
    LazyConfig,
    get_data_info,
)
from xmetai.config import LazyCall as L
from xmetai.recipes import recipe_path

######################################################################
# Global settings
######################################################################
# set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")
data_dir = os.environ["DATA_DIR"]
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/", os.path.basename(CONFIG_PATH)))
cur_date = datetime.now().strftime("%Y%m%d")

#####################################################################
# Training settings
#####################################################################
lr = float(os.environ.get("LR", 0.0005))  # get lr from environment variable, 1e-5 ~ 2e-4.
total_batch_size = 1
epochs = 30
iters_per_epoch = 60000

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
    train.teacher_forcing.teach_steps = 4
    output_dir += f"/mstep-{cur_date}/"
    lr = 1.28e-5
    epochs = epochs // 3
    load_from_pretrain = True
    scheduler.scheduler.end_value = 1e-2
    # scheduler = LazyConfig.load(recipe_path("schedule")).const_lr

train.max_iter = epochs * iters_per_epoch // total_batch_size  # max_steps with 60000 iterations per epoch
train.output_dir = output_dir
train.activation_checkpointing = True
train.use_fsdp2 = True
train.use_fsdp = False
train.use_ddp = False
# train.eval_period = iters_per_epoch//total_batch_size  # eval for each epoch
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

test_names = ["z500", "msl", "t2m", "sst", "ttr", "tp", "u850", "v850"]
# test_names = ["z500", "t2m", "tp"]
test_names = [ch for ch in test_names if ch in out_names]
test_chans = [out_names.index(n) for n in test_names]
test_frames = np.arange(1, 13).tolist()  # predict 1-3 days ahead with 6-hour frequency
test_frames = np.arange(1, 61).tolist()  # predict 1-15 days ahead with 6-hour frequency
# test_frames = np.concatenate([test_frames, np.arange(61, 170, 4)]) # predict 42 days ahead
step_range = [max(test_frames)]

in_chans = len(in_names)
in_frames = 2
out_chans = len(out_names)
freq = 6
const_chans = 6
degree = 0.25
out_frames = 1

depth_list = os.environ.get("DEPTH_LIST", "6,6,12,12")  # default depth list
depths = [int(x) for x in depth_list.split(",") if x.strip().isdigit()]

######################################################################

train_times = ("19500101-00", "20231231-18")
test_times = ("20240101-18", "20261231-18")
# test_times = ("20050101-18", "20250502-18")

dataloader = LazyConfig.load(recipe_path("dataloader"), "zarr_dataloader")(
    data_paths=[data_dir],
    train_times=train_times,
    test_times=test_times,
    total_batch_size=total_batch_size,
    train_workers=1,
    test_workers=1,
    test_pin_memory=False,
    test_interval=16,
    eval_test_names=test_names,
    test_frames=test_frames,
    freq=freq,
)
# Follow the shared RMSE channel/lead selection when a derived config overrides it.
dataloader.evaluator.append(
    L(METRICS.get("ForecastFidelity"))(
        test_names="${..0.test_names}",
        test_times="${..0.test_times}",
        image_size=(int(180 / degree) + 1, int(360 / degree)),
    )
)
# Physical TP is already in millimeters per 6 hours, matching the corrector loss.
precip_mm_scale = 1.0
# Expose as a config field (not just a python constant) so it can be overridden from the
# CLI/yaml with a single key, e.g. "dataloader.precip_mm_scale=21.6", instead of having to
# patch every appended evaluator entry individually.
dataloader.precip_mm_scale = precip_mm_scale
precip_thresholds_mm = (0.1, 10.0, 25.0, 50.0, 100.0, 250.0)
precip_neighborhood_size = 1
if "tp" in test_names:
    for metric_name in ("ThreatScore", "ProbabilityOfDetection", "FalseAlarmRate", "FrequencyBias"):
        dataloader.evaluator.append(
            L(METRICS.get(metric_name))(
                thresholds=precip_thresholds_mm,
                composite_weights=(1.0,) * len(precip_thresholds_mm) if metric_name in ("ThreatScore", "FrequencyBias") else (),
                test_times="${..0.test_times}",
                neighborhood_size=precip_neighborhood_size,
                channel_name="tp",
                test_names="${..0.test_names}",
                value_scale="${dataloader.precip_mm_scale}",
            )
        )
    for window_size in (1, 3, 5, 15, 31, 63):
        dataloader.evaluator.append(
            L(METRICS.get("RadarFSS"))(
                thresholds=(13.0,),
                channel_name="tp",
                metric_name=f"PrecipFSS_w{window_size}",
                empty_score=float("nan"),
                test_times="${..0.test_times}",
                test_names="${..0.test_names}",
                neighborhood_size=window_size,
                value_scale="${dataloader.precip_mm_scale}",
            )
        )
if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

# note: explicit pretrain fcst_frames is set above in the try block;
# no helper call is needed.

export_onnx = False
if export_onnx:
    train.use_ddp = False
    train.use_fsdp = False
    train.use_fsdp2 = False
    train.activation_checkpointing = False
    dataloader.test.total_batch_size = 1
    dataloader.test.num_workers = 1
    dataloader.test.pin_memory = False
    dataloader.test.dataset.inference_only = True
    dataloader.test.dataset.fcst_frames = [1]

    from xmetai.data import make_onnx

    make_onnx(
        model=train,
        dataloader_cfg=dataloader.test,
        mean=buffers.get("mean", None),
        std=buffers.get("std", None),
        in_names=in_names,
        output_dir=output_dir,
        embed_types=["step", "hour", "doy"],
        model_name="iwc_pred.onnx",
        device=train.device_type,
    )
