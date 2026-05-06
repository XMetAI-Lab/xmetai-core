"""
Author: Your Name
Date: 2026-04-19
Description: Config for TianQuan (ResCast sub-seasonal-to-seasonal forecasting model).
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

from xmetai import (
    DATASETS,
    METRICS,
    LazyConfig,
    build_test_loader,
    build_train_loader,
    get_data_info,
)

from s2s_base import (  # noqa: F401
    ROOT_PATH,
    buffers,
    depths,
    in_chans,
    load_from_pretrain,
    maskid,
    optimizer,
    output_dir,
    pinn_coeffs,
    scheduler,
    train,
    in_names, out_chans, logid, test_chans, test_names,
)

from xmetai import MODELS, LazyConfig
from xmetai import LazyCall as L
from omegaconf import OmegaConf
import numpy as np


# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
# 1. 指定你的模型注册名
model_name = "PhyCast"

if load_from_pretrain:
    # Enable cascaded training for M-Step rollout logic.
    train.teacher_forcing.teach_start = 1
    train.teacher_forcing.teach_steps = 10
    train.teacher_forcing.teach_iters = train.max_iter // train.teacher_forcing.teach_steps
    train.eval_period = train.teacher_forcing.teach_iters

    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)

train.output_dir = output_dir
train.eval_period = 30000
######################################################################
# Model settings
######################################################################
freq = 24  # 24 hours for 1 day

# phyCast 参数
patch_size = 4         # 根据参考配置，这里使用 4
embed_dim = 12
num_heads = 1
decoder_depth = 2
token_spatial_size=(15, 30)
mlp_ratio = 4.0
drop_path = 0.1
drop_rate = 0.1
# test 参数
out_frames = 60
in_frames = 16

test_frames = np.arange(1, 61).tolist()  # predict 1-42 days ahead
step_range = [max(test_frames)]



# 3. 实例化模型
model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,  # 保持 128，保证基础特征容量
    depths=decoder_depth,  # 层数从 12 砍到 6 (参数量直接减半)
    num_heads=num_heads,  # 头数降为 4 (每个头的维度变成 128/4=32，非常合理的比例)
    token_spatial_size=token_spatial_size,
    drop_rate=drop_rate,
    mlp_ratio=mlp_ratio,
    mask_schedule="cosine",  # 掩码调度策略
    config_path="configs/s2s_phyVAE.py",  # vae 相关参数
    vocab_size=1024,
    num_quantizers=2,  # <--- 【新增】传入量化层数
    const_chans=0,
    # 下面为 BasePredict 的参数
    in_chans=in_chans,  # ERA5 输入变量数
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=out_frames,
    freq=freq,
    accumid=logid,
    logid=logid,
    maskid=[],  # maskid,
    zero_accum=False,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    save_dir=output_dir + "/pred/",
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
    pinn_use_denorm=True,  # whether to denormalize the inputs before computing PINN losses; typically improves PINN performance
    # use_pertub=True,
)

model.buffers = buffers
model.precision = "bf16"  # 开启 BFloat16 半精度训练，防止 384 维度的模型显存溢出

######################################################################
# Data settings
######################################################################
data_dir = os.environ.get("DATA_DIR", "/mnt/d/s2s.1950-2024.c76")
dataloader = OmegaConf.create()
total_batch_size = 1

train_times = ("19500101", "20230228")
test_times = ("20231001", "20241231")
eval_times = [int(t * freq) for t in test_frames]
# 配置测评哪些参数
dataloader.evaluator = [
    L(METRICS.get("MSE"))(
        test_names=test_names,
        test_times=eval_times,
    ),
]

dataloader.train = L(build_train_loader)(
    dataset=L(DATASETS.get("ERA5"))(
        data_name=data_dir,
        hist_frames=in_frames,
        fcst_frames=out_frames,
        interval=1,
        freq=freq,
        training=True,
        years=train_times,
    ),
    total_batch_size=total_batch_size,
    num_workers=0,
    pin_memory=True,
)

dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("ERA5"))(
        data_name=data_dir,
        hist_frames=in_frames,
        fcst_frames=test_frames,
        interval=-1,
        freq=freq,
        training=False,
        inference_only=False,
        years=test_times,
    ),
    total_batch_size=total_batch_size,
    num_workers=0,
    pin_memory=True,
)

# 动态修改 Dataset 配置以适应连续预测 (仅在预训练微调时激活)
if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps


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


