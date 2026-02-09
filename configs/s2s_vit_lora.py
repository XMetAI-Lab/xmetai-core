"""
Example LoRA fine-tuning configuration for S2S ViT model.

This config demonstrates how to use LoRA for parameter-efficient fine-tuning
of a pretrained ViT model. It loads a pretrained checkpoint and only trains
LoRA parameters, drastically reducing memory usage and training time.

Usage:
    # Fine-tune with LoRA
    bash scripts/train.bash -m configs/s2s_vit_lora.py

    # Evaluate
    bash scripts/train.bash -m configs/s2s_vit_lora.py --eval

Author: XMetAI Team
Date: 2026-02-02
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
    load_from_pretrain,
    logid,
    maskid,
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

from xmetai import MODELS
from xmetai import LazyCall as L

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings for LoRA fine-tuning
#####################################################################
model_name = "ViTModel"
if load_from_pretrain:  # check if original model exists, start stage 2: ensemble
    train.ens_train = True
    train.init_checkpoint = f"{output_dir}/../model_best.pth"
    model_name = "ViTEnsModel"
    output_dir += "/../ens/"
    train.teacher_forcing.teach_iters = 100
    train.teacher_forcing.teach_steps = 5  # teach for 5 steps with 80GB GPU memory
    if os.path.exists(f"{output_dir}/model_final.pth"):  # check if ensemble model exists, start stage 3: rollout
        train.init_checkpoint = f"{output_dir}/model_best.pth"
        output_dir += "/mstep/"
        train.teacher_forcing.teach_steps = 20  # teach for 20 steps with 80GB GPU memory
    train.max_iter = 50000  # Reduce iterations for fine-tuning

train.output_dir = output_dir
# Enable LoRA
train.lora.enabled = True
train.lora.r = 16  # LoRA rank
train.lora.alpha = 32  # LoRA scaling (alpha/r = 2.0)
train.lora.dropout = 0.05
# Target attention and Mamba mixer layers in the encoder
# Based on actual weight names with activation checkpointing:
# encoder.X._checkpoint_wrapped_module.blocks.Y.mixer.{in_proj,out_proj,qkv,proj}
train.lora.target_modules = [
    # Attention layers (transformer blocks): qkv and proj
    r"encoder\.\d+\.(_checkpoint_wrapped_module\.)?blocks\.\d+\.mixer\.qkv$",
    r"encoder\.\d+\.(_checkpoint_wrapped_module\.)?blocks\.\d+\.mixer\.proj$",
    # Mamba layers: in_proj and out_proj
    r"encoder\.\d+\.(_checkpoint_wrapped_module\.)?blocks\.\d+\.mixer\.in_proj$",
    r"encoder\.\d+\.(_checkpoint_wrapped_module\.)?blocks\.\d+\.mixer\.out_proj$",
    # Optionally add MLP layers for deeper fine-tuning
    # r"encoder\.\d+\.(_checkpoint_wrapped_module\.)?blocks\.\d+\.mlp\.fc[12]$",
]
train.lora.num_steps = train.teacher_forcing.teach_steps

# Reduce learning rate for fine-tuning (LoRA params will use 2x this rate automatically)
optimizer.lr = 1e-5
optimizer.weight_decay = 0.01
######################################################################
# Model settings (same as s2s_vit.py)
######################################################################
zero_accum = True  # whether to zero the accumulation variables after each optimization step.
freq = 24  # 24 hours for 1 day
# model setting
patch_size = 2
embed_dim = 1536
num_heads = 24
image_size = (120, 240)  # resize from 121x240->120x240 with input data
members = 8

model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    patch_size=patch_size,
    image_size=image_size,
    drop_rate=0.1,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    freq=freq,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    zero_accum=zero_accum,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=members,
    save_dir=output_dir + "/pred/",
    layer_type="mamba",  # Enable Mamba for better sequence modeling
)

model.buffers = buffers
model.precision = "bf16"

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
    model.alpha = 0.9  # for exponential weights of future frames
