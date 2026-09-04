#!/usr/bin/env python3
"""Train an IWC precipitation corrector on top of a frozen ONNX model."""

import logging
import os

from xmetai import METRICS, MODELS
from xmetai import LazyCall as L

from .iwc_base import (  # noqa: F401
    accumid,
    buffers,
    dataloader,
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
train.init_checkpoint = ""
train.use_ddp = True
train.use_fsdp = False
train.use_fsdp2 = False
train.activation_checkpointing = True
train.max_iter = 20000
train.teacher_forcing.teach_iters = -1
train.teacher_forcing.teach_steps = 1
dataloader.train.dataset.fcst_frames = 1
dataloader.train.dataset.years = ("20100101-00", "20231231-18")

######################################################################
# IWC precipitation contract
######################################################################
correct_names = ["tp"]
correct_chans = [out_names.index(name) for name in correct_names]
precip_output_channel = test_names.index("tp")
precip_mm_scale = 1.0
precip_thresholds_mm = (0.1, 10.0, 25.0, 50.0)
precip_neighborhood_size = 1
eval_times = [int(step * freq) for step in test_frames]

######################################################################
# Model settings
######################################################################
model = L(MODELS.get("CorrectionModel"))(
    onnx_path="/workspace/tmp/douzsh/models/iwc_fgvp_gdn2.onnx",
    correct_chans=correct_chans,
    backbone_kwargs=dict(base_channels=128, ch_mults=(1, 2, 4, 8), layers_per_block=3, time_dim=512, dropout=0.02),
    rollout_max=1,
    rollout_alpha=1.0,
    zero_init_output=True,
    gpu_mem_fraction=0.6,
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

######################################################################
# Evaluation: thresholds are reported in millimeters per 6 hours.
######################################################################
dataloader.evaluator = [
    dataloader.evaluator[0],
    L(METRICS.get("ThreatScore"))(
        thresholds=precip_thresholds_mm,
        test_times=eval_times,
        neighborhood_size=precip_neighborhood_size,
        channel_index=precip_output_channel,
        value_scale=precip_mm_scale,
    ),
    L(METRICS.get("ProbabilityOfDetection"))(
        thresholds=precip_thresholds_mm,
        test_times=eval_times,
        neighborhood_size=precip_neighborhood_size,
        channel_index=precip_output_channel,
        value_scale=precip_mm_scale,
    ),
    L(METRICS.get("FalseAlarmRate"))(
        thresholds=precip_thresholds_mm,
        test_times=eval_times,
        neighborhood_size=precip_neighborhood_size,
        channel_index=precip_output_channel,
        value_scale=precip_mm_scale,
    ),
    L(METRICS.get("FrequencyBias"))(
        thresholds=precip_thresholds_mm,
        test_times=eval_times,
        neighborhood_size=precip_neighborhood_size,
        channel_index=precip_output_channel,
        value_scale=precip_mm_scale,
    ),
]
