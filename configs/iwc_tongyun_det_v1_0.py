#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-09-17 15:16:51
Description: Model Architecture Designed by XMetAI.

See docs/tongyun/README.md for architecture and training details.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import os

import numpy as np

from xmetai import MODELS
from xmetai.config import LazyCall as L

from .iwc_base import (  # noqa: F401
    ROOT_PATH,
    accumid,
    buffers,
    dataloader,
    depths,
    export_onnx,
    freq,
    in_chans,
    in_frames,
    in_names,
    load_from_pretrain,
    logid,
    optimizer,
    out_chans,
    out_names,
    output_dir,
    scheduler,
    test_chans,
    test_frames,
    test_names,
    train,
)

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)
if os.path.exists(f"{output_dir}/model_best.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"
loss_weight = (0.1, 0.2, 1.0, 0.01, 1.0)  # (vae_input, vae_target, pred_rec, kl, latent_pred_rec)
dataloader.train.num_workers = 2
dataloader.test.num_workers = 1
if load_from_pretrain:
    loss_weight = (0.0, 0.0, 0.1, 0.0, 0.0)  # (vae_input, vae_target, pred_rec, latent_reg, latent_pred_rec)
    train.teacher_forcing.teach_start = 42
    train.teacher_forcing.teach_steps = 42
    train.max_iter = 3000
    train.eval_period = 1000
    train.segmented_backward = True
    train.rollout_segment_size = 6
    # optimizer.params.overrides = {"posterior_encoder_group": {"lr_factor": 0.1}}
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    # scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    # scheduler.end_value = 1e-2
    # scheduler.period = 1.0 * train.eval_period / train.max_iter
    optimizer.lr = 1e-5
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
    train_times = ("20100101-00", "20231231-18")
    dataloader.train.dataset.years = train_times
else:
    train.teacher_forcing.teach_iters = -1
    train.teacher_forcing.teach_steps = 1
    test_frames = [1, 2]
    dataloader.test.dataset.fcst_frames = test_frames
    dataloader.evaluator[0].test_times = [t * freq for t in test_frames]
    if os.path.exists("/workspace/cma/output/pretrain/iwc_gssp/model_final.pth"):
        train.max_iter = 60000
        loss_weight = (0.1, 0.1, 1.0, 0.01, 0.01)  # (vae_input, vae_target, pred_rec, kl, latent_pred_rec)
        train.init_checkpoint = "/workspace/cma/output/pretrain/iwc_gssp/model_final.pth"
        optimizer.params.overrides = {"posterior_encoder_group": {"lr_factor": 0.1}}
######################################################################
# Model settings
######################################################################
# Per-channel training weights.
channel_weights = buffers.get("channel_weights", np.ones(len(out_names), dtype=np.float32))
channel_weights[out_names.index("tp")] = 2.0
channel_weights[out_names.index("t2m")] = 1.5
channel_weights[out_names.index("msl")] = 1.5
buffers["channel_weights"] = channel_weights
members = 1

# Spatial resolution and patch layout.
image_size = (721, 1440)
patch_size = 8
window_size = 18

model = L(MODELS.get("TongYun"))(
    # UTransformer shared params
    embed_dim=512,
    image_size=image_size,
    window_size=window_size,
    patch_size=patch_size,
    down_times=0,
    encoder_depths=depths[:2],
    decoder_depths=depths[-2:-1],
    num_heads=16,
    drop_mcmc=0.01,
    attn_type="rotary_flash",
    mask_type="h",
    head_type="2fc",
    norm_type="adaln",
    ffn_type="gegluffn",
    # Shared latent dynamics.
    vt_num_heads=24,
    vt_num_layers=depths[-1],
    vt_layer_type="gdn2_swin",
    vt_use_proj=True,
    vt_window_size=window_size,
    vt_attn_type="rotary_flash",
    # loss / perturbation
    alpha=0.95,
    loss_weight=loss_weight,
    # BasePredictor params
    in_names=in_names,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=6,
    in_frames=in_frames,
    out_frames=1,
    logid=logid,
    maskid=[],
    accumid=accumid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=freq,
    amse_weight=0.9,
    latent_dyn_weight=0.1,
    precip_loss_enable=False,
    precip_loss_coeffs={"occurrence": 0.01, "distribution": 0.005},
    precip_threshold_mm=0.1,
    precip_names=("tp",),
    save_dir=output_dir + "/pred/",
    rollout_max=-1,
)

model.members = members
model.buffers = "${buffers}"
model.precision = "bf16"
model.kl_min_weight = 1.0
model.kl_max_weight = 10.0
model.kl_cycle_iters = 100 if load_from_pretrain else 30000

if export_onnx:
    model.precision = "fp32"
    train.export_kwargs.save_name = os.path.join(output_dir, "tongyun_det.onnx")
