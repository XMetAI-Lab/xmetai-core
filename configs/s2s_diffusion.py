#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file for training a SwinVRNN model on the ERA5 dataset.
This configuration file is designed to be used for s2s deterministic training.
"""
import logging
from xmetai import LazyCall as L
from xmetai import MODELS
from s2s_base import train, optimizer, scheduler, dataloader
from s2s_base import output_dir, buffers
from s2s_base import in_chans, out_chans, in_frames, test_chans, test_names, step_range, test_frames
from s2s_base import logid, maskid

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)
######################################################################
# Model settings
######################################################################
# model setting
zero_accum = True  # whether to zero the accumulation variables after each optimization step.
num_heads = 24
image_size=(120, 240) # resize from 121x240->120x240 with input data

model = L(MODELS.get("DiffusionForecast"))(
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    step_range=step_range,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    save_dir=output_dir+"/pred/",
# start Diffusion args.
    base_channels = 256,
    ch_mults = (1, 2, 4),
    attn_res = (16, 8, 4),
    num_heads = num_heads,
    dropout = 0.1,
    predict_type = "epsilon",
)

model.buffers = buffers
model.zero_accum = zero_accum
model.precision = "fp16"