#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: shenZheWen
Date: 2026-04-19
Description: Config for TianQuan (ResCast sub-seasonal-to-seasonal forecasting model).
"""
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
    in_frames,
    load_from_pretrain,
    maskid,
    optimizer,
    output_dir,
    pinn_coeffs,
    scheduler,
    step_range,
    test_frames,
    train,
    in_names,
)

from xmetai import LazyCall as L, MODELS
from omegaconf import OmegaConf
import numpy as np

# set logging to debug level if needed
logging.basicConfig(level=logging.INFO)

#####################################################################
# Training settings
#####################################################################
# 1. 指定你的模型注册名
model_name = "TianQuan"
CHECKPOINT_FILENAME = "model_final.pth"

if load_from_pretrain:
    train.ens_train = True
    train.init_checkpoint = os.path.join(output_dir, CHECKPOINT_FILENAME)
    # Enable cascaded training for M-Step rollout logic.
    train.teacher_forcing.teach_start = 1
    train.teacher_forcing.teach_steps = 10
    train.teacher_forcing.teach_iters = train.max_iter // train.teacher_forcing.teach_steps
    train.eval_period = train.teacher_forcing.teach_iters

    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)

train.output_dir = output_dir
# train.eval_period = 30
######################################################################
# Model settings
######################################################################
freq = 24  # 24 hours for 1 day

# TianQuan 模型专属超参数设置
img_size = (8, 16)  # 降低分辨率，token 数减少 4 倍，防止 OOM
patch_size = 4
embed_dim = 12
num_heads = 1
decoder_depth = 1  # 从 2 降到 1
mlp_ratio = 4.0
drop_path = 0.1
drop_rate = 0.1

tianquan_vars = [
    "2m_temperature", "wind_speed_10m",
    "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200",
    "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500",
    "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925",
    "geopotential_1000",
    "wind_speed_50", "wind_speed_100", "wind_speed_150", "wind_speed_200",
    "wind_speed_250", "wind_speed_300", "wind_speed_400", "wind_speed_500",
    "wind_speed_600", "wind_speed_700", "wind_speed_850", "wind_speed_925",
    "wind_speed_1000",
    "temperature_50", "temperature_100", "temperature_150", "temperature_200",
    "temperature_250", "temperature_300", "temperature_400", "temperature_500",
    "temperature_600", "temperature_700", "temperature_850", "temperature_925",
    "temperature_1000",
    "relative_humidity_50", "relative_humidity_100", "relative_humidity_150", "relative_humidity_200",
    "relative_humidity_250", "relative_humidity_300", "relative_humidity_400", "relative_humidity_500",
    "relative_humidity_600", "relative_humidity_700", "relative_humidity_850", "relative_humidity_925",
    "relative_humidity_1000",
    "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200",
    "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500",
    "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925",
    "specific_humidity_1000",
]

# 严格匹配 TianQuan 模型的 67 个气象变量（已转换为标准库存短命名）
tianquan_vars_2 = [
    # ---- 2 个地表变量 ----
    "t2m", "10u",  # 用 10u 代替 10m风速 (库存无标量风速)

    # ---- 13 层位势高度 (geopotential -> z) ----
    "z50", "z100", "z150", "z200",
    "z250", "z300", "z400", "z500",
    "z600", "z700", "z850", "z925",
    "z1000",

    # ---- 13 层 U 侧风 (u，代替原本的 wind_speed) ----
    "u50", "u100", "u150", "u200",
    "u250", "u300", "u400", "u500",
    "u600", "u700", "u850", "u925",
    "u1000",

    # ---- 13 层温度 (temperature -> t) ----
    "t50", "t100", "t150", "t200",
    "t250", "t300", "t400", "t500",
    "t600", "t700", "t850", "t925",
    "t1000",

    # ---- 13 层 V 侧风 (v，代替原本的 relative_humidity) ----
    "v50", "v100", "v150", "v200",
    "v250", "v300", "v400", "v500",
    "v600", "v700", "v850", "v925",
    "v1000",

    # ---- 13 层比湿 (specific_humidity -> q) ----
    "q50", "q100", "q150", "q200",
    "q250", "q300", "q400", "q500",
    "q600", "q700", "q850", "q925",
    "q1000",
]

needed_ids = [in_names.index(var) for var in tianquan_vars_2]
buffers["channel_weights"] = buffers["channel_weights"][needed_ids]
buffers["mean"] = buffers["mean"][needed_ids]
buffers["std"] = buffers["std"][needed_ids]

in_names = tianquan_vars_2
idx = [in_names.index('tp')] if 'tp' in in_names else []
logid, accumid = idx, idx

test_names = ["z500", "u200", "u850", "t2m", "t850"]
out_chans = len(tianquan_vars_2)
test_chans = [tianquan_vars_2.index(n) for n in test_names]

# 3. 实例化你的模型 (严格按照 TianQuan 的 __init__ 传参)
model = L(MODELS.get(model_name))(
    default_vars=tianquan_vars,
    img_size=img_size,
    window_size=(8, 8),
    patch_size=patch_size,
    embed_dim=embed_dim,
    decoder_depth=decoder_depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop_path=drop_path,
    drop_rate=drop_rate,
    in_chans=in_chans,
    out_chans=out_chans,
    const_chans=2,
    needed_ids=needed_ids,
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
    save_dir=output_dir + "/pred/",
    # PINN integration: enable physics-informed losses for ViT experiments by default
    pinn_enable=False,
    # pinn_coeffs=pinn_coeffs,
    in_names=in_names,
    pinn_use_denorm=True,
    # whether to denormalize the inputs before computing PINN losses; typically improves PINN performance
    # use_pertub=True,
    climate_zarr="/mnt/d/cla.zarr",
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
# test_times = ("20050101", "20250202")
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
        fcst_frames=2,
        interval=1,
        freq=freq,
        training=True,
        years=train_times,
        in_names=tianquan_vars_2
    ),
    total_batch_size=total_batch_size,
    num_workers=1,
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
        in_names=tianquan_vars_2
    ),
    total_batch_size=total_batch_size,
    num_workers=1,
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
