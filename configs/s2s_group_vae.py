#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-03-06 12:00:00
Description: Level-wise VAE predictor config for s2s experiments.

Group 0 (1000hPa + surface): 16 channels (auto-detected from in_names)
Group 1 (925-400hPa): 5 channels each (weight-shared across 6 levels)
Group 2 (300-50hPa): 5 channels each (weight-shared across 6 levels)

Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import os

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    accumid,
    buffers,
    dataloader,
    depths,
    in_chans,
    in_names,
    load_from_pretrain,
    logid,
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

vae_type = "absolute"
if os.path.exists(f"{output_dir}/model_best.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"
if os.path.exists(f"{output_dir}/model_posterior_fix_{vae_type}.pth"):
    train.init_checkpoint = f"{output_dir}/model_posterior_fix_{vae_type}.pth"
    train.freeze = ("posterior_encoder_group",)

loss_weight = (0.2, 0.2, 1.0, 0.002, 1.0)  # (vae_input, vae_target, pred_rec, latent_reg, latent_pred_rec)
if load_from_pretrain:
    # freeze upper-level VAE to preserve pretrained dynamics, only fine-tune lower-level VAE and main predictor.
    train.teacher_forcing.teach_start = 2
    train.teacher_forcing.teach_steps = 10
    train.freeze = (
        "posterior_encoder_group.2",
        "vae_encoder_group.2",
    )
    # train.teacher_forcing.teach_start = 10
    # train.teacher_forcing.teach_steps = 20
    # train.freeze = (
    #    "posterior_encoder_group.1",
    #    "vae_encoder_group.1",
    # )
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    train.eval_period = train.teacher_forcing.teach_iters
    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

######################################################################
# Model settings
######################################################################
members = 1

model = L(MODELS.get("GroupVAEPredictor"))(
    # UTransformer shared architecture params
    embed_dim=256,  # 3*embed_dim for mix predictor
    image_size=(120, 240),
    window_size=10,
    patch_size=2,
    down_times=0,
    encoder_depths=depths[:2],
    decoder_depths=depths[-2:-1],
    num_heads=16,
    drop_mcmc=0.05,
    attn_type="v2",
    mask_type="h",
    head_type="2fc",
    norm_type="adaln",
    ffn_type="gegluffn",
    # VAE settings
    vae_type=vae_type,
    level_groups=[[0], [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
    split_surface=False,
    # Vertical transformer settings for VAE latent prediction
    vt_num_heads=24,
    vt_num_layers=depths[-1],
    vt_layer_type="mamba3_met_embed",
    vt_window_size=10,
    vt_attn_type="rotary_flash",
    vt_use_proj=True,
    # Overall model settings
    alpha=0.9,
    loss_weight=loss_weight,
    # BasePredictor params
    in_names=in_names,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
    in_frames=2,
    out_frames=1,
    logid=logid,
    maskid=[],  # maskid,
    accumid=accumid,
    zero_accum=False,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=24,
    # pinn_enable=True,
    # pinn_coeffs=pinn_coeffs,
    # pinn_use_denorm=True,
    save_dir=output_dir + "/pred/",
)

model.members = members
model.step_range = step_range
model.buffers = buffers
model.precision = "bf16"
