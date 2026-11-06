#!/usr/bin/env python3
"""Train an IWC precipitation adapter on top of a frozen ONNX model."""

import logging
import os

from xmetai import MODELS
from xmetai import LazyCall as L

from .iwc_base import (  # noqa: F401
    accumid,
    buffers,
    dataloader,
    export_onnx,
    freq,
    in_chans,
    in_frames,
    in_names,
    logid,
    maskid,
    optimizer,
    out_chans,
    out_names,
    output_dir,
    precip_mm_scale,
    scheduler,
    test_chans,
    test_frames,
    test_names,
    train,
)

logging.basicConfig(level=logging.INFO)

######################################################################
# Training settings
######################################################################
train.use_ddp = True
train.use_fsdp = False
train.use_fsdp2 = False
train.activation_checkpointing = True
train.max_iter = 5000
train.eval_period = 1000
# Expand by one lead every 100 iterations, reaching all 60 IWC leads at iter 5900.
train.teacher_forcing.teach_iters = 100
train.teacher_forcing.teach_start = 10
train.teacher_forcing.teach_steps = 10
train.segmented_backward = True
train.rollout_segment_size = 1
train.val_metric = "RMSE/tp"
dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
dataloader.train.dataset.years = ("20100101-00", "20231231-18")
dataloader.test.dataset.target_names = test_names

######################################################################
# IWC precipitation contract
######################################################################
correct_names = ["tp"]
correct_chans = [out_names.index(name) for name in correct_names]
dataloader.train.dataset.target_names = correct_names
######################################################################
# Model settings
######################################################################
model = L(MODELS.get("ForecastAdapter"))(
    onnx_path="/workspace/tmp/douzsh/models/20261013/tongyun_det.onnx",
    correct_chans=correct_chans,
    target_chans=correct_chans,
    backbone_kwargs=dict(base_channels=128, ch_mults=(1, 2, 4, 8), layers_per_block=3, time_dim=512, dropout=0.02),
    rollout_max=0,
    rollout_alpha=1.0,
    zero_init_output=True,
    gpu_mem_fraction=0.6,
    linear_loss_enable=True,  # Enable observed-mm linear weighting of the residual loss.
    linear_loss_a=0.3,
    linear_loss_use_latitude=True,
    in_names=in_names,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    members=1,
    step_range=[max(test_frames)],
    accumid=accumid,
    logid=logid,
    maskid=maskid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    freq=freq,
    save_dir=os.path.join(output_dir, "pred"),
    buffers=buffers,
    precision="bf16",
    precip_loss_enable=True,
    precip_loss_coeffs={"occurrence": 0.01, "distribution": 0.005},
    precip_names=("tp",),
    precip_threshold_mm=0.1,
    precip_mm_scale=precip_mm_scale,
    precip_occurrence_scale=10.0,
)

model.custom_ops_library = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"

######################################################################
# Standalone ONNX: raw_pred + lead -> physical corrected TP.
######################################################################
if export_onnx:
    import torch

    from xmetai.data import make_onnx

    train.use_ddp = False
    train.use_fsdp = False
    train.use_fsdp2 = False
    train.activation_checkpointing = False
    train.init_checkpoint = os.path.join(train.output_dir, "model_best.pth")
    model.precision = "fp32"
    make_onnx(
        model=train,
        dataloader_cfg=dataloader.test,
        mean=buffers["mean"],
        std=buffers["std"],
        in_names=in_names,
        output_dir=train.output_dir,
        embed_types=[],
        forward_name="export_onnx",
        model_name="iwc_tp_adapter.onnx",
        device=train.device_type,
    )
    # Only the forecast frame is needed; the original ONNX is not traced.
    train.export_kwargs.data = dict(raw_pred=train.export_kwargs.data.input[:, -1:], lead=torch.ones(1, dtype=torch.float32))
