"""
配置文件：hydro_vit.py
描述：山洪 ViT 模型定义
继承自：configs/hydro_base.py
"""
import logging
import os

# 【关键】通过 Python 导入机制获取基础配置
# 这就像 import math 一样，直接读取 hydro_base.py 文件里的变量
from hydro_base import (
    dataloader,
    in_chans,  # 9
    in_frames,  # 3
    out_chans,  # 3
    output_dir,
    train,
    optimizer,
    scheduler,
    buffers,
)

from xmetai import MODELS
from xmetai import LazyCall as L

# 设置日志级别
logging.basicConfig(level=logging.INFO)

#####################################################################
# 模型结构定义
#####################################################################
# 1000x1000 的图非常大，我们需要调整参数以防显存爆炸

# Patch Size: 就像把一张大披萨切成小块。
# 气象模型通常用 2 或 4。但对于 1000x1000，如果用 2，会产生 (500*500)=25万个序列，显存瞬间爆炸。
# 我们设为 8 或 16。先尝试 8 (产生 125*125=1.5万个序列，这在 ViT 可接受范围内)。
patch_size = 8

embed_dim = 128  # 嵌入维度 (脑容量)，先设小一点确保能跑起来
num_heads = 4  # 注意头数要能被 embed_dim 整除 (128 / 4 = 32)
depths = [2, 2, 4, 2]  # 模型层数，先不要太深

# 定义模型
# 使用系统现有的 ViTModel，但填入我们的参数
model = L(MODELS.get("HydroViTModel"))(
    # 核心 ViT 参数
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    patch_size=patch_size,
    image_size=(1000, 1000),  # 【关键】你的栅格尺寸
    drop_rate=0.1,

    # 输入输出通道
    in_chans=in_chans,  # 9
    out_chans=out_chans,  # 3
    in_frames=in_frames,  # 3
    out_frames=1,  # 预测未来 1 帧

    # 占位参数 (兼容性设置)
    freq=1,
    accumid=[],
    logid=[],
    maskid=[],
    zero_accum=False,
    test_frames=[1],
    test_chans=[0, 1, 2],
    test_names=["h", "u", "v"],
    step_range=[1],

    members=1,

    # 输出路径
    save_dir=output_dir + "/pred/",

    # 使用 Swin 风格还是 Mamba 风格？
    # 如果你的环境支持 Mamba (安装了相关库)，可以用 "mamba"
    # 如果不确定，先用 "swin" (更稳) 或者 "vit"
    layer_type="swin",
)

# 附加配置
model.buffers = buffers
model.precision = "bf16"  # 使用半精度 (BF16) 节省显存，A100/H800/3090/4090 均支持