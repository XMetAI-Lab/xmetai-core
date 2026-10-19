#!/usr/bin/env python3
"""Unified S2S configuration for diffusion, flow matching, and JiT models."""

import logging
import os

from xmetai import MODELS, LazyConfig
from xmetai import LazyCall as L

from .s2s_base import (  # noqa: F401
    ROOT_PATH,
    accumid,
    buffers,
    dataloader,
    freq,
    in_chans,
    in_frames,
    in_names,
    load_from_pretrain,
    logid,
    maskid,
    optimizer,
    out_chans,
    output_dir,
    step_range,
    test_chans,
    test_frames,
    test_names,
    train,
)
from .s2s_base import scheduler as base_scheduler

logging.basicConfig(level=logging.INFO)

model_type = os.environ.get("GENERATIVE_MODEL_TYPE", "FlowMatchingForecast")
if model_type not in {"DiffusionForecast", "FlowMatchingForecast", "JiTForecast"}:
    raise ValueError(f"Unsupported GENERATIVE_MODEL_TYPE={model_type!r}")

if load_from_pretrain:
    train.teacher_forcing.teach_start = 10
    train.teacher_forcing.teach_steps = 19
    rollout_steps = train.teacher_forcing.teach_steps - train.teacher_forcing.teach_start + 1
    train.max_iter = rollout_steps * 1000
    train.teacher_forcing.teach_iters = train.max_iter // rollout_steps
    from xmetai.recipes import recipe_path

    scheduler = LazyConfig.load(recipe_path("schedule")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / rollout_steps
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
else:
    train.ddp.static_graph = True
    train.teacher_forcing.teach_iters = -1
    train.teacher_forcing.teach_steps = 1
    test_frames = [1, 2]
    dataloader.test.dataset.fcst_frames = test_frames
    dataloader.evaluator[0].test_times = [t * freq for t in test_frames]
    scheduler = base_scheduler
    step_range = [max(test_frames)]

######################################################################
# Model settings
######################################################################
members = 1

model = L(MODELS.get(model_type))(
    # Shared generative backbone
    backbone="darknet_unet",
    backbone_kwargs=dict(base_channels=614, ch_mults=(1, 2, 3), layers_per_block=3, dropout=0.05),
    sample_steps=50,
    sample_init="zeros",
    noise_scale=1.0,
    predict_residual=True,
    use_timestamp=False,  # Keep compatibility with the current pretrained Flow Matching checkpoint.
    # Shared rollout training
    rollout_max=5,
    rollout_alpha=0.98,
    rollout_sample_steps=1,
    # BasePredictor params
    in_names=in_names,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    const_chans=2,
    accumid=accumid,
    logid=logid,
    maskid=maskid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=freq,
    save_dir=output_dir + "/pred/",
)

if model_type == "DiffusionForecast":
    model.num_steps = 1_000
    model.sample_steps = 100
    model.sample_timestep_max = 999
    model.train_timestep_max = 999
    model.scheduler_type = "ddim"
    model.ddim_eta = 0.0
    model.clip_x0 = False
    model.predict_type = "v_prediction"
elif model_type == "FlowMatchingForecast":
    model.time_scale = 1_000.0
    model.sampling_method = "heun"
else:
    model.p_mean = -0.8
    model.p_std = 0.8
    model.t_eps = 0.05
    model.sampling_method = "heun"

model.members = members
model.step_range = step_range
model.buffers = "${buffers}"
model.precision = "bf16"
