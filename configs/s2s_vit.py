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
    ROOT_PATH,
    buffers,
    dataloader,
    depths,
    in_chans,
    in_frames,
    in_names,
    load_from_pretrain,
    logid,
    maskid,
    optimizer,
    out_chans,
    output_dir,
    pinn_coeffs,
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
#####################################################################
# Training settings
#####################################################################
model_name = "PerturbUMamba"

if load_from_pretrain:
    train.ens_train = True
    model_name = "PerturbUMambaEns"
    # Enable cascaded training for M-Step rollout logic.
    train.teacher_forcing.teach_start = 1
    train.teacher_forcing.teach_steps = 10
    train.teacher_forcing.teach_iters = train.max_iter // train.teacher_forcing.teach_steps
    train.eval_period = train.teacher_forcing.teach_iters

    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)

train.output_dir = output_dir
######################################################################
# Model settings
######################################################################
zero_accum = True  # whether to zero the accumulation variables after each optimization step.
freq = 24  # 24 hours for 1 day
# model setting
patch_size = 2
embed_dim = 256
num_heads = 32
image_size = (120, 240)  # resize from 121x240->120x240 with input data
members = 8

model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    patch_size=patch_size,
    image_size=image_size,
    window_size=10,
    drop_rate=0.01,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
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
    layer_type="mamba_met_embed",  # Use local windowed Mamba blocks
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
    pinn_use_denorm=True,  # whether to denormalize the inputs before computing PINN losses; typically improves PINN performance
    # use_pertub=True,
    pertub_mixup=0.0,
    pertub_mask=0.0,
)

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
    model.alpha = 0.9  # Start with some noise regularization even during fine-tuning, can be tuned based on validation performance.
