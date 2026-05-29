#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file for training a SwinVRNN model on the ERA5 dataset.
This configuration file is designed to be used for s2s deterministic training.
"""

import logging
import os

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    buffers,
    dataloader,
    freq,
    in_chans,
    in_frames,
    load_from_pretrain,
    logid,
    maskid,
    optimizer,
    out_chans,
    output_dir,
    scheduler,
    step_range,
    test_chans,
    test_frames,
    test_names,
    train,
)

from xmetai import MODELS, LazyConfig
from xmetai import LazyCall as L

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

if load_from_pretrain:
    train.teacher_forcing.teach_start = 2
    train.teacher_forcing.teach_steps = 20
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    train.eval_period = train.teacher_forcing.teach_iters
    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    test_frames = [1]
    step_range = [1]
    dataloader.test.dataset.fcst_frames = test_frames
    dataloader.evaluator[0].test_times = [freq]

######################################################################
# Model settings
######################################################################
# model setting
zero_accum = True  # whether to zero the accumulation variables after each optimization step.
num_heads = 16
image_size = (120, 240)  # resize from 121x240->120x240 with input data

model = L(MODELS.get("DiffusionForecast"))(
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    step_range=step_range,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    save_dir=output_dir + "/pred/",
    freq=freq,
    # start Diffusion args.
    base_channels=128,
    ch_mults=(1, 2, 3),
    attn_res=(16, 8, 4),
    num_heads=num_heads,
    dropout=0.05,
    layers_per_block=3,
    backbone="darknet",
    num_steps=1000,
    sample_steps=100,
    sample_timestep_max=999,
    train_timestep_max=999,
    scheduler_type="ddim",
    ddim_eta=0.0,
    clip_x0=False,
    predict_type="v_prediction",
    predict_residual=True,
    sample_init="zeros",
    train_chans=[test_chans[0]],
    const_chans=2,
    image_size=image_size,
)

model.buffers = buffers
model.zero_accum = zero_accum
model.precision = "bf16"
