#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件：hydro_base.py
描述：山洪模型的基础配置，定义数据路径、变量名和训练参数。
"""

import os
import numpy as np
from omegaconf import OmegaConf

# 从核心库导入必要的组件
from xmetai import (
    DATASETS,
    METRICS,
    LazyConfig,
    build_test_loader,
    build_train_loader,
)
from xmetai import LazyCall as L

######################################################################
# 1. 全局路径与环境设置
######################################################################
# 显存优化：避免碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")

# 【关键修改】你的数据存放路径
data_dir = "/workspace/data/duanjiacheng/hydrodata"

# 输出路径：训练日志和模型会保存在这里
output_dir = os.environ.get("OUTPUT_DIR", os.path.join(ROOT_PATH, "output/hydro_base_check"))

#####################################################################
# 2. 训练超参数 (学习率、BatchSize)
#####################################################################
# 学习率
lr = 5e-4

# 【关键修改】批次大小。
# 你的图片是1000x1000，非常大，单卡显存可能只能跑1张，所以设为1
total_batch_size = 1

# 迭代次数：根据你的样本量大概估算。假设想跑 50 个 Epoch
# 这里的 20000 是个示例，后续可以根据实际样本数调整
max_iter = 50000

# 加载系统通用的训练配置 (继承自 common 文件夹)
# 这一步会加载优化器(AdamW)和调度器(Scheduler)的默认设置
train = LazyConfig.load(os.path.join(ROOT_PATH, "common/train.py")).train
optimizer = LazyConfig.load(os.path.join(ROOT_PATH, "common/optim.py")).AdamW
scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).cosine_lr

# 更新训练参数
train.max_iter = max_iter
train.output_dir = output_dir
optimizer.lr = lr

######################################################################
# 3. 变量与通道定义 (手动指定)
######################################################################
# 原有的 get_data_info 是读气象元数据的，我们不需要，直接手写：

# 输入变量名 (仅用于记录)
in_names = ["rain", "dem", "lai", "sand", "clay", "som", "h_prev", "u_prev", "v_prev"]
# 输出变量名
out_names = ["h", "u", "v"]

# 【关键】通道数配置
in_chans = 9   # 1(雨) + 5(地形) + 3(历史水流)
out_chans = 3  # h, u, v
in_frames = 3  # 时间窗口大小 (Window Size)

# 占位符 (为了兼容原有代码结构，保留这些变量名，但设为 {} 或默认值)
buffers = {}   # 均值方差缓存，暂时不用
freq = 1         # 时间频率
depth_list = "2,2,6,2" # 默认模型深度
depths = [int(x) for x in depth_list.split(",")]

######################################################################
# 4. 数据加载器配置 (Data Loader)
######################################################################
dataloader = OmegaConf.create()

# 训练集加载器
dataloader.train = L(build_train_loader)(
    dataset=L(DATASETS.get("FloodDataset"))(  # 使用我们在 hydro.py 定义的类
        data_dir=data_dir,
        training=True,
        window_size=in_frames,
    ),
    total_batch_size=total_batch_size,
    num_workers=4,        # CPU线程数
    pin_memory=True,
    do_augment=False,     # 【再次确认】关闭经度平移增强
)

# 测试/验证集加载器
dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("FloodDataset"))(
        data_dir=data_dir,
        training=False,   # 验证模式
        window_size=in_frames,
    ),
    total_batch_size=total_batch_size,
    num_workers=2,
    pin_memory=True,
)

# 评估指标 (简单使用 MSE)
# test_times 设为 [1] 是为了骗过原有代码的检查，实际在 FloodDataset 里我们不依赖这个
dataloader.evaluator = [
    L(METRICS.get("MSE"))(
        test_names=out_names,
        test_times=[1],
    ),
]