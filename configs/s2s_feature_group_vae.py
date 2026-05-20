#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-grouped VAE predictor config for s2s experiments.

Groups follow FeatureGroupVAEPredictor.GROUP_VAR_MAP semantics:
    - residual_state: encode [x_tm1, x_t - x_tm1], decode residual
    - abs_wind: encode [x_tm1, x_t], decode absolute value
    - abs_scalar: encode [x_tm1, x_t], decode absolute value
"""

import logging
import os

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    dataloader,
    depths,
    freq,
    load_from_pretrain,
    optimizer,
    output_dir,
    scheduler,
    step_range,
    test_frames,
    test_names,
    train,
)

from xmetai import MODELS, LazyConfig, get_data_info
from xmetai import LazyCall as L

logging.basicConfig(level=logging.INFO)

######################################################################
# Data settings
######################################################################
data_dir = "/workspace/data/cra.1979_2025.c87.1p5.h24.new.zarr"
train_times = ("19500101", "20240228")
test_times = ("20240101", "20251231")

data_info = get_data_info([data_dir])
in_names = data_info["in_names"]
out_names = data_info["out_names"]
logid = data_info["logid"]
accumid = data_info["accumid"]
buffers = data_info["buffers"]

in_chans = len(in_names)
out_chans = len(out_names)
test_chans = [out_names.index(n) for n in test_names]

dataloader.train.dataset.data_name = data_dir
dataloader.train.dataset.years = train_times
dataloader.test.dataset.data_name = data_dir
dataloader.test.dataset.years = test_times

######################################################################
# Training settings
######################################################################
if os.path.exists(f"{output_dir}/model_best.pth"):
    train.init_checkpoint = f"{output_dir}/model_best.pth"

loss_weight = (0.2, 0.2, 1.0, 0.1, 0.8)  # (vae_input, vae_target, pred_rec, latent_reg, latent_pred_rec)
if load_from_pretrain:
    train.max_iter = train.max_iter * 2
    train.teacher_forcing.teach_start = 2
    train.teacher_forcing.teach_steps = 20
    train.freeze = (
        "posterior_encoder_group.1",
        "posterior_encoder_group.2",
        "vae_encoder_group.1",
        "vae_encoder_group.2",
    )
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    train.eval_period = train.teacher_forcing.teach_iters
    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    test_frames = [1, 2]
    dataloader.test.dataset.fcst_frames = test_frames
    dataloader.evaluator[0].test_times = [t * freq for t in test_frames]
    if os.path.exists("/workspace/cma/output/cra_group_perturb/pretrain_new/model_final.pth"):
        train.init_checkpoint = "/workspace/cma/output/cra_group_perturb/pretrain_new/model_final.pth"
        train.freeze = ("posterior_encoder_group",)
        train.max_iter = train.max_iter // 2


######################################################################
# Model settings
######################################################################
members = 8

model = L(MODELS.get("FeatureGroupVAEPredictor"))(
    # UTransformer shared architecture params
    embed_dim=512,
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
    # Vertical transformer settings for VAE latent prediction
    vt_num_heads=24,
    vt_num_layers=depths[-1],
    vt_layer_type="mamba_met_embed",
    vt_window_size=None,
    vt_attn_type="std",
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
    maskid=[],
    accumid=accumid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=24,
    save_dir=output_dir + "/pred/",
    rollout_max=10,
)

model.members = members
model.step_range = step_range
model.buffers = buffers
model.precision = "bf16"
