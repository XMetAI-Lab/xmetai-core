"""
Author: Zhaohan && hanzhao1228@gmail.com
Date: 2026-04-21 10:28:00
Description: Iwc deterministic model with FourcastNet.
Copyright (c) 2025 by XMetAI/zhaohan, All Rights Reserved.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import numpy as np

from xmetai import (
    DATASETS,
    MODELS,
    build_test_loader,
    build_train_loader,
)
from xmetai import LazyCall as L

from .iwc_base import (  # noqa: F401
    buffers,
    data_dir,
    dataloader,
    in_names,
    load_from_pretrain,
    optimizer,
    output_dir,
    scheduler,
    total_batch_size,
    train,
)

logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
model_name = "AFNONet"

if os.path.exists(f"{output_dir}/../model_final.pth"):  # check if original model exists, start stage 2: ensemble
    train.init_checkpoint = f"{output_dir}/../model_best.pth"
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 1  # rollout 1 during ensemble training

train.output_dir = output_dir
train.eval_period = 4000
######################################################################
# Model settings
######################################################################
vars_afnonet = ["u10m", "v10m", "t2m", "msl", "u1000", "v1000", "z1000", "t850", "u850", "v850", "z850", "t500", "u500", "v500", "z500", "z50", "tp"]
# get id of vars_afnonet from data_info['in_names']
needed_ids = [in_names.index(var) for var in vars_afnonet]
buffers["channel_weights"] = buffers["channel_weights"][needed_ids]
buffers["mean"] = buffers["mean"][needed_ids]
buffers["std"] = buffers["std"][needed_ids]

# update in_names
in_names = vars_afnonet
out_names = vars_afnonet
logid = accumid = [in_names.index("tp")] if "tp" in in_names else []
maskid = [in_names.index("sst")] if "sst" in in_names else []
in_chans = len(in_names)
in_frames = 1
out_chans = len(out_names)

test_names = ["z500", "u10m", "v10m", "t2m", "t850", "tp"]
test_names = [ch for ch in test_names if ch in out_names]
test_chans = [out_names.index(n) for n in test_names]
test_frames = np.arange(1, 33).tolist()  # predict 1-8 days ahead with 6-hour frequency
step_range = [max(test_frames)]

freq = 6
embed_dim = 768
img_size = (720, 1440)
patch_size = (8, 8)
depth = 12
mlp_ratio = 4.0
drop_rate = 0.1
drop_path_rate = 0.0
num_blocks = 8
sparsity_threshold = 0.01
hard_thresholding_fraction = 1.0

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
    accumid=accumid,
    logid=logid,
    maskid=maskid,  # maskid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=1,
    save_dir=output_dir + "/pred/",
    in_names=in_names,
)
if load_from_pretrain:
    model.alpha = 1.1  # sequence loss exponential weight factor

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

train_times = ("19500101-00", "20171231-18")
test_times = ("20180101-18", "20181231-18")
# test_times = ("20050101-18", "20250502-18")
dataloader.train = L(build_train_loader)(
    dataset=L(DATASETS.get("MultiZarrDataset"))(
        data_paths=[data_dir],
        hist_frames=in_frames,
        fcst_frames=2,
        interval=1,
        freq=freq,
        training=True,
        years=train_times,
        in_names=in_names,
    ),
    total_batch_size=total_batch_size,
    num_workers=0,
    pin_memory=True,
)

dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("MultiZarrDataset"))(
        data_paths=[data_dir],
        hist_frames=in_frames,
        fcst_frames=test_frames,
        interval=16,
        freq=freq,
        training=False,
        inference_only=False,
        years=test_times,
        in_names=in_names,
    ),
    total_batch_size=total_batch_size,
    num_workers=0,
    pin_memory=True,
)
