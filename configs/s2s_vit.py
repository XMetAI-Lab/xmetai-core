"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-11-07 21:09:40
Description: subseasonal to seasonal deterministic model with ViT backbone.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

from s2s_base import (  # noqa: F401
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
    pinn_enable,
    pinn_coeffs,
    in_names,
)

from xmetai import MODELS
from xmetai import LazyCall as L

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
        train.teacher_forcing.teach_steps = 20  # teach for 20 steps with 80GB GPU memory
    model_name = "ViTEnsModel"

train.output_dir = output_dir
######################################################################
# Model settings
######################################################################
zero_accum = True  # whether to zero the accumulation variables after each optimization step.
freq = 24  # 24 hours for 1 day
# model setting
patch_size = 2
embed_dim = 1536
num_heads = 24
image_size = (120, 240)  # resize from 121x240->120x240 with input data
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
    freq=freq,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    zero_accum=zero_accum,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=members,
    save_dir=output_dir + "/pred/",
    layer_type="mamba",  # Enable Mamba for better sequence modeling
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=True,
    pinn_coeffs=pinn_coeffs,
    in_names=in_names,
)
if load_from_pretrain:
    model.alpha = 0.9  # sequence loss exponential weight factor

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
