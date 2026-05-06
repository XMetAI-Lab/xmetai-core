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
model_name = "Phy3DRQVAE"
CHECKPOINT_FILENAME = "best.pth"

if load_from_pretrain:
    train.init_checkpoint = CHECKPOINT_FILENAME
    # Enable cascaded training for M-Step rollout logic.
    train.teacher_forcing.teach_start = 1
    train.teacher_forcing.teach_steps = 10
    train.teacher_forcing.teach_iters = train.max_iter // train.teacher_forcing.teach_steps
    train.eval_period = train.teacher_forcing.teach_iters

    scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).circular_lr
    scheduler.end_value = 1e-2
    scheduler.period = 1.0 / (train.teacher_forcing.teach_steps - 1)

train.output_dir = output_dir
train.eval_period = 3000
######################################################################
# Model settings
######################################################################
freq = 24  # 24 hours for 1 day

embed_dim = 256
token_spatial_size = (15, 30)  # 假设 RQ-VAE 降采样后的 Token 空间分辨率 (H', W')
num_quantizers = 4
codebook_size = 2048
drop_rate = 0.1
mlp_ratio = 4

out_frames = 2

# 假设你的变量命名规则是 z500, u500, v500...
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# 2. 自动提取索引
# 使用 f-string 匹配变量名，比如 'z1000', 'u1000', 'v1000'
idx_z_list = [in_names.index(f'z{l}') for l in levels]
idx_u_list = [in_names.index(f'u{l}') for l in levels]
idx_v_list = [in_names.index(f'v{l}') for l in levels]


# 检查三者长度是否一致
assert len(idx_z_list) == len(idx_u_list) == len(idx_v_list), "各层级变量不匹配，请检查 in_names"

# 3. 实例化你的模型 (严格按照 TianQuan 的 __init__ 传参)
model = L(MODELS.get(model_name))(
    embed_dim=embed_dim,  # 隐空间通道数 (Latent Dimension)
    num_quantizers=num_quantizers,  # 残差量化级数
    codebook_size=codebook_size,  # 每一级词表大小
    commit_weight=1.0,  # VQ Commitment 损失权重
    geo_weight=0.05,  # 地转平衡物理损失权重
    idx_phi=idx_z_list,  # 位势高度 (Geopotential) 在 in_chans 中的通道索引
    idx_u=idx_u_list,  # U 风分量在 in_chans 中的通道索引
    idx_v=idx_v_list,  # V 风分量在 in_chans 中的通道索引
    image_size=(121, 240),  # 空间分辨率
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
# test_times = ("20050101", "20250202")
eval_times = [int(t * freq) for t in test_frames]
test_frames = np.arange(1, 61).tolist()  # predict 1-42 days ahead
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

logger = logging.getLogger("xmetai")
logger.info(f"评估周期 (eval_period): {train.eval_period}")
logger.info(f"输出帧数 (out_frames): {out_frames}")

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
