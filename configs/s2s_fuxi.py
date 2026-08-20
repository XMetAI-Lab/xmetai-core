"""
Subseasonal to seasonal deterministic model with the FuXi-2.1 backbone.

Project-native downscaled FuXi-2.1 (flat single-trunk Transformer: Swin window
attention + 1-D RoPE + adaLN time conditioning + SwiGLU FFN, variable-aware
multi-head decoder), trained from scratch on the project's 1.5-degree ERA5
dataset. Official 0.25-degree weights are not loaded.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    accumid,
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

logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
model_name = "Fuxi2"

train.output_dir = output_dir

######################################################################
# Model settings
######################################################################
freq = 24

# Downscaled from the official 0.25-degree config (embed_dim=1536, depth=30,
# num_heads=24, patch_size=6) to the project's 1.5-degree grid:
#   image_size (120, 240) + patch_size 2 -> 60x120 tokens.
#   window_size 6 divides both 60 and 120 (shift = 3).
#
# 4xA100 server config: ~195M params. Fits comfortably on one A100 (bf16 +
# activation checkpointing + AdamW), so DDP over 4 GPUs just widens the
# effective batch. Keep embed_dim % num_heads == 0 (768 / 12 = 64 head_dim).
patch_size = 2
embed_dim = 768
depth = 16
num_heads = 12
window_size = 6
mlp_ratio = 8 / 3
members = 1

# Distributed / data settings for 4xA100 (DDP, world_size=4).
#
# total_batch_size is GLOBAL; xmetai/data/loader.py splits it as
#   per_gpu_batch = total_batch_size // world_size
# so it must be >= world_size and a multiple of it. 8 -> 2 per GPU.
total_batch_size = 8
dataloader.train.total_batch_size = total_batch_size
dataloader.test.total_batch_size = 8

# Recompute the per-batch iteration budget (~30 epochs over the ~20k-day set).
base_iter = 30 * 20000 // total_batch_size

# ---- Training curriculum: single-step from scratch, then rollout fine-tune ----
# `s2s_base` flips `load_from_pretrain` to True once <output>/model_final.pth
# exists (single-step finished); it also lowers LR and points the run at a fresh
# `output/mstep-<date>/` dir. Mirror s2s_group_vae's rollout schedule: ramp the
# autoregressive horizon 2 -> 20 steps once per circular-LR cycle over a
# shortened budget. Fuxi2.forward already rolls out data["future_frames"] steps,
# so no model change is required.
if load_from_pretrain:
    train.teacher_forcing.teach_start = 2
    train.teacher_forcing.teach_steps = 20
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.max_iter = base_iter // 3
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    train.eval_period = train.teacher_forcing.teach_iters
    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    # FuXi-2.1 is single-step autoregressive; keep one-step teacher forcing.
    train.teacher_forcing.teach_iters = -1
    dataloader.train.dataset.fcst_frames = 1
    train.max_iter = base_iter

model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    window_size=window_size,
    mlp_ratio=mlp_ratio,
    patch_size=patch_size,
    image_size=(120, 240),
    drop_rate=0.0,
    drop_path_rate=0.1,
    loss_type="mse",
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
    in_frames=in_frames,
    out_frames=1,
    freq=freq,
    accumid=accumid,
    logid=logid,
    maskid=[],  # maskid,
    zero_accum=True,  # FuXi-2.1 zeroes derived (radiation/precip) channels before feedback
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
        model_name="s2s_fuxi.onnx",
        device=train.device_type,
    )
