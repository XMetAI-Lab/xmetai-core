#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-09 12:53:45
Description: Inference config for s2s model with onnx.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging

import numpy as np

from xmetai import MODELS
from xmetai import LazyCall as L

from .s2s_base import (  # noqa: F401
    accumid,
    buffers,
    dataloader,
    freq,
    in_chans,
    in_frames,
    logid,
    maskid,
    optimizer,
    out_chans,
    output_dir,
    scheduler,
    test_chans,
    test_names,
    train,
)

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
train.use_ddp = False  # Disable DDP for ONNX inference
train.use_fsdp = False  # Disable FSDP for ONNX inference
train.use_fsdp2 = False  # Disable FSDP2 for ONNX inference

######################################################################
# Model settings
######################################################################
test_frames = np.arange(1, 15).tolist()  # predict 1-14 days ahead
step_range = [max(test_frames)]
members = 8  # for ensemble prediction inference mse calculation.
zero_accum = True  # set accumid to zero every time. For CMA env, some variables are not available in some time steps.
model = L(MODELS.get("OnnxInferModel"))(
    onnx_path="s2s.onnx",
    save_dir=output_dir + "/pred/",
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    accumid=accumid,
    logid=logid,
    maskid=maskid,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=freq,
)

model.members = members
model.step_range = step_range
model.buffers = buffers
model.zero_accum = zero_accum
model.precision = "fp32"

######################################################################
# Data settings
######################################################################
dataloader.test.dataset.interval = 100
dataloader.test.dataset.fcst_frames = test_frames
eval_times = [int(t * freq) for t in test_frames]
dataloader.evaluator[0].test_times = eval_times
