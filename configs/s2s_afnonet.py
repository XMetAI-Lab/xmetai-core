"""
Author: Zhaohan && hanzhao1228@gmail.com
Date: 2026-04-21 10:28:00
Description: subseasonal to seasonal deterministic model with FourcastNet.
Copyright (c) 2025 by XMetAI/zhaohan, All Rights Reserved.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from s2s_base import (  # noqa: F401
    buffers,
    dataloader,
    freq,
    in_chans,
    in_names,
    load_from_pretrain,
    logid,
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
from xmetai import LazyCall as L

in_frames = 1
dataloader.train.dataset.hist_frames = in_frames
dataloader.test.dataset.hist_frames = in_frames

if load_from_pretrain:
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 20  # teach for 20 steps with 80GB GPU memory
    scheduler.scheduler.end_value = 1e-2

train.output_dir = output_dir
model_name = "AFNONet"

######################################################################
# Model settings
######################################################################
embed_dim = 768
img_size = (120, 240)
patch_size = (5, 5)
depth = 12
mlp_ratio = 4.0
drop_rate = 0.1
drop_path_rate = 0.0
num_blocks = 16
sparsity_threshold = 0.01
hard_thresholding_fraction = 1.0

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

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

export_onnx = False
if export_onnx:
    from xmetai.data import make_onnx

    train.use_ddp = False
    train.use_fsdp = False
    train.activation_checkpointing = False
    dataloader.test.total_batch_size = 1
    make_onnx(
        model=train,
        dataloader_cfg=dataloader.test,
        mean=buffers.get("mean", None),
        std=buffers.get("std", None),
        in_names=in_names,
        output_dir=output_dir,
        # embed_types=["step"],
        embed_types=["step", "hour", "doy"],
        model_name="s2s_pred.onnx",
        device=train.device_type,
    )
