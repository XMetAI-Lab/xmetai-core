"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-11 08:07:31
Description:
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

from iwc_base import (  # noqa: F401
    buffers,
    dataloader,
    depths,
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

from xmetai import MODELS
from xmetai.config import LazyCall as L

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)
#####################################################################
# Training settings
#####################################################################
model_name = "ViTModel"
if os.path.exists(f"{output_dir}/../model_final.pth"):  # check if original model exists, start stage 2: ensemble
    train.ens_train = True
    train.init_checkpoint = f"{output_dir}/../model_best.pth"
    output_dir += "/../ens/"
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 1  # rollout 1 during ensemble training
    if os.path.exists(f"{output_dir}/model_final.pth"):  # check if ensemble model exists, start stage 3: rollout
        train.init_checkpoint = f"{output_dir}/model_best.pth"
        output_dir += "/mstep/"
        train.teacher_forcing.teach_steps = 12  # teach for 12 steps with 80GB GPU memory
    model_name = "ViTModelEns"

train.output_dir = output_dir
train.eval_period = 4000
######################################################################
# Model settings
######################################################################
embed_dim, num_heads, patch_size = 1536, 24, 8
image_size = (720, 1440)
members = 8
model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    patch_size=patch_size,
    image_size=image_size,
    drop_rate=0.1,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=members,
    save_dir=output_dir + "/pred/",
    freq=6,
    layer_type="mamba",  # Enable Mamba for better sequence modeling
)
if load_from_pretrain:
    model.alpha = 1.1  # sequence loss exponential weight factor

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
