"""
Subseasonal to seasonal deterministic model with PuYun LKA-FCN backbone.
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

logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
model_name = "PuYunModel"

if load_from_pretrain:
    # PuYun uses dynamic-step autoregressive finetuning after one-step pretraining.
    train.teacher_forcing.teach_start = 1
    train.teacher_forcing.teach_steps = 12
    train.teacher_forcing.teach_iters = train.max_iter // train.teacher_forcing.teach_steps
    train.eval_period = train.teacher_forcing.teach_iters

    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    # PuYun single-step pretraining (paper stage-1).
    train.max_iter = 10000
    train.eval_period = 3000
    train.teacher_forcing.teach_iters = -1
    dataloader.train.dataset.fcst_frames = 1

train.output_dir = output_dir

######################################################################
# Model settings
######################################################################
freq = 24

# The paper uses 1536d + (12,12,12,12) @ K9 at 0.25 degrees.
# This config keeps the same LKA-FCN topology but scales width to fit the
# project's 1.5 degree S2S setup and single-GPU debugging workflow.
patch_size = 2
embed_dim = 256
kernel_size = 9
members = 1

model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    depths=depths,
    image_size=(120, 240),
    patch_size=patch_size,
    kernel_size=kernel_size,
    mlp_ratio=4.0,
    drop_rate=0.05,
    drop_path_rate=0.2,
    loss_type="puyun",
    use_pixel_shuffle=True,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
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
    members=members,
    save_dir=output_dir + "/pred/",
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
    pinn_use_denorm=True,
)

model.buffers = buffers
model.precision = "bf16"

export_onnx = os.getenv("EXPORT_ONNX", "").lower() in ("1", "true", "yes", "on")
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
        embed_types=["step", "hour", "doy"],
        model_name="s2s_puyun.onnx",
        device=train.device_type,
    )
