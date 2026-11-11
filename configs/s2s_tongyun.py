#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-03-06 12:00:00
Description: Model Architecture Designed by XMetAI.

See docs/tongyun/README.md for architecture and training details.

Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import os

from xmetai import MODELS, LazyConfig
from xmetai import LazyCall as L

from .s2s_base import (  # noqa: F401
    ROOT_PATH,
    accumid,
    buffers,
    dataloader,
    depths,
    export_onnx,
    freq,
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

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

if os.path.exists(f"{output_dir}/model_best.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"
loss_weight = (0.2, 0.2, 1.0, 0.002, 0.8)  # (vae_input, vae_target, pred_rec, latent_reg, latent_pred_rec)

in_frames = 2
out_frames = 1
dataloader.train.dataset.hist_frames = in_frames
dataloader.test.dataset.hist_frames = in_frames

if load_from_pretrain:
    # Resume TongYun rollout training.
    train.teacher_forcing.teach_start = 2
    train.teacher_forcing.teach_steps = 20
    train.freeze = ()
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    train.eval_period = train.teacher_forcing.teach_iters
    from xmetai.recipes import recipe_path

    scheduler = LazyConfig.load(recipe_path("schedule")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    train.teacher_forcing.teach_iters = -1
    train.teacher_forcing.teach_steps = 1
    test_frames = [i for i in range(1, out_frames + 2)]
    dataloader.test.dataset.fcst_frames = test_frames
    dataloader.evaluator[0].test_times = [t * freq for t in test_frames]


######################################################################
# Model settings
######################################################################
members = 1

model = L(MODELS.get("TongYun"))(
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
    # Use TongYun's default reconstruction settings.
    # Shared latent dynamics settings
    vt_num_heads=24,
    vt_num_layers=depths[-1],
    vt_layer_type="mamba3_met_embed",
    vt_use_proj=True,
    vt_window_size=10,
    vt_attn_type="rotary_flash",
    vt_group_norm_mode="norm",
    # Overall model settings
    alpha=0.9,
    loss_weight=loss_weight,
    # BasePredictor params
    in_names=in_names,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
    in_frames=in_frames,
    out_frames=out_frames,
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
    rollout_max=10,
)

model.members = members
model.step_range = step_range
model.buffers = "${buffers}"
model.precision = "bf16"

if export_onnx:
    train.export_kwargs.save_name = os.path.join(output_dir, "s2s_tongyun.onnx")
