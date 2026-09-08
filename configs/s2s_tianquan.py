#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: shenZheWen
Date: 2026-04-19
Description: Config for TianQuan (ResCast sub-seasonal-to-seasonal forecasting model).
"""

import logging
import os

from omegaconf import OmegaConf

from xmetai import (
    DATASETS,
    METRICS,
    MODELS,
    LazyConfig,
    build_test_loader,
    build_train_loader,
)
from xmetai import LazyCall as L

from .s2s_base import (  # noqa: F401
    ROOT_PATH,
    buffers,
    depths,
    export_onnx,
    in_chans,
    in_frames,
    in_names,
    load_from_pretrain,
    maskid,
    optimizer,
    output_dir,
    pinn_coeffs,
    scheduler,
    step_range,
    test_frames,
    train,
)

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

    from xmetai.recipes import recipe_path

    scheduler = LazyConfig.load(recipe_path("schedule")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)

train.output_dir = output_dir
# train.eval_period = 30
######################################################################
# Model settings
######################################################################
freq = 24  # 预报时间步长（小时）

# TianQuan 模型专属超参数
img_size = (8, 16)  # 输入重采样尺寸(H, W)
patch_size = 4  # patch 大小
embed_dim = 12  # 隐藏维度
num_heads = 1  # 注意力头数
decoder_depth = 1  # 主干深度
mlp_ratio = 4.0  # FFN 扩展倍数
drop_path = 0.1  # 随机深度
drop_rate = 0.1  # dropout

# 统一维护高空层级（与数据通道命名保持一致）
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# TianQuan 默认变量命名（模型内部语义名）：
# 2 个地表变量 + 5 组高空变量（每组 13 层）。
tianquan_vars = [
    "2m_temperature",
    "wind_speed_10m",
    *[f"geopotential_{lvl}" for lvl in pressure_levels],
    *[f"wind_speed_{lvl}" for lvl in pressure_levels],
    *[f"temperature_{lvl}" for lvl in pressure_levels],
    *[f"relative_humidity_{lvl}" for lvl in pressure_levels],
    *[f"specific_humidity_{lvl}" for lvl in pressure_levels],
]

# 严格匹配数据集中的 67 通道短命名：
# 2 个地表变量 + z/u/t/v/q 各 13 层。
tianquan_vars_2 = [
    "t2m",
    "10u",  # 数据集中无标量 10m 风速，使用 10u 作为近地风代表。
    *[f"z{lvl}" for lvl in pressure_levels],
    *[f"u{lvl}" for lvl in pressure_levels],
    *[f"t{lvl}" for lvl in pressure_levels],
    *[f"v{lvl}" for lvl in pressure_levels],
    *[f"q{lvl}" for lvl in pressure_levels],
]

needed_ids = [in_names.index(var) for var in tianquan_vars_2]
buffers["channel_weights"] = buffers["channel_weights"][needed_ids]
buffers["mean"] = buffers["mean"][needed_ids]
buffers["std"] = buffers["std"][needed_ids]

in_names = tianquan_vars_2
idx = [in_names.index("tp")] if "tp" in in_names else []
logid, accumid = idx, idx

# 评估时重点关注的变量
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
    climate_zarr="/mnt/d/cla.zarr",  # 气候态 zarr 路径
)

model.buffers = buffers
model.precision = "bf16"  # 开启 BFloat16 半精度训练，防止 384 维度的模型显存溢出

######################################################################
# Data settings
######################################################################
data_dir = os.environ.get("DATA_DIR", "/mnt/d/s2s.1950-2024.c76")  # 训练数据路径
dataloader = OmegaConf.create()
total_batch_size = 1  # 总 batch（会按 world_size 均分）

train_times = ("19500101", "20230228")  # 训练时间范围
test_times = ("20231001", "20241231")  # 测试时间范围
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
        data_name=data_dir, hist_frames=in_frames, fcst_frames=2, interval=1, freq=freq, training=True, years=train_times, in_names=tianquan_vars_2
    ),
    total_batch_size=total_batch_size,
    num_workers=1,  # 训练集 DataLoader 进程数
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
        in_names=tianquan_vars_2,
    ),
    total_batch_size=total_batch_size,
    num_workers=1,  # 测试集 DataLoader 进程数
    pin_memory=True,
)

# 动态修改 Dataset 配置以适应连续预测 (仅在预训练微调时激活)
if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps

if export_onnx:
    train.export_kwargs.save_name = os.path.join(output_dir, "s2s_tianquan.onnx")
