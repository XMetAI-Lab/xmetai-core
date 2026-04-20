#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    buffers,
    dataloader,
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
model_name = "AFNONet"

train.output_dir = output_dir
######################################################################
# Model settings
######################################################################
freq = 24  # 24 hours for 1 day
# model setting
embed_dim = 768
img_size = (120, 140)
patch_size=(5, 5)
depth=12
mlp_ratio=4.
drop_rate=0.1
drop_path_rate=0.
num_blocks=16
sparsity_threshold=0.01
hard_thresholding_fraction=1.0
in_frames = 1 # 原论文中输入为1帧

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
    accumid=logid,
    logid=logid,
    maskid=[],  # maskid,
    zero_accum=False,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=1,
    save_dir=output_dir + "/pred/",
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
)

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
    model.alpha = 0.9  # Start with some noise regularization even during fine-tuning, can be tuned based on validation performance.
