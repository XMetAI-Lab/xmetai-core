#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
from omegaconf import OmegaConf

from xmetai import (
    DATASETS,
    METRICS,
    LazyConfig,
    build_test_loader,
    build_train_loader,
)
from xmetai import LazyCall as L


def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def getenv_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default))
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------------------
# paths and environment
# ---------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:32,expandable_segments:True"
)

CONFIG_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(CONFIG_PATH, "../")

data_dir = getenv_str("DATA_DIR", "/workspace/data/duanjiacheng/hydrodata")
output_dir = getenv_str(
    "OUTPUT_DIR",
    os.path.join(ROOT_PATH, "output/hydro_base_check")
)
os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------------------------------------
# training hyper-parameters
# ---------------------------------------------------------------------
lr = getenv_float("LR", 5e-4)
total_batch_size = 1
max_iter = getenv_int("MAX_ITER", 100000)

eval_period = getenv_int("EVAL_PERIOD", 1000)
checkpoint_period = getenv_int("CHECKPOINT_PERIOD", 1000)
log_period = getenv_int("LOG_PERIOD", 20)

train = LazyConfig.load(os.path.join(ROOT_PATH, "common/train.py")).train
optimizer = LazyConfig.load(os.path.join(ROOT_PATH, "common/optim.py")).AdamW
scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).cosine_lr

train.max_iter = max_iter
train.output_dir = output_dir
train.eval_period = eval_period
train.checkpointer.period = checkpoint_period
train.log_period = log_period
train.val_metric = "RMSE/h"

optimizer.lr = lr


# ---------------------------------------------------------------------
# multi-step training control
# ---------------------------------------------------------------------
# tools/train.py 中：
#   当 train.teacher_forcing.teach_iters > 0 时，
#   会启用 ExpandTrainer。
#
# ExpandTrainer 的逻辑是：
#   current_step = iter // teach_iters
#   future_step  = min(current_step + 1, data["targets"].shape[1])
#
# 因此：
# - teach_iters 控制“每多少 iter 扩 1 步”
# - 真正最大能扩到多少步，取决于训练集返回的 targets.shape[1]
#
# 这里的 teach_steps 主要作为训练集返回 target 长度的默认目标值。

enable_multi_step_train = getenv_bool("ENABLE_MULTI_STEP_TRAIN", True)
teach_iters = getenv_int("TEACH_ITERS", 20000)
teach_steps = getenv_int("TEACH_STEPS", 24)
ens_train = getenv_bool("ENS_TRAIN", False)

if enable_multi_step_train:
    train.teacher_forcing.teach_iters = teach_iters
    train.teacher_forcing.teach_steps = teach_steps
else:
    train.teacher_forcing.teach_iters = -1
    train.teacher_forcing.teach_steps = 1

train.ens_train = ens_train


# ---------------------------------------------------------------------
# task definitions
# ---------------------------------------------------------------------
history_steps = getenv_int("HISTORY_STEPS", 2)

input_channel_names = ["rain", "h", "u", "v"]
const_channel_names = ["dem", "lai", "sand", "clay", "som"]
out_names = ["h", "u", "v"]

in_chans = 4
out_chans = 3

# 输入帧 = 历史 history_steps 帧 + 当前 forcing 1 帧
in_frames = history_steps + 1

# 模型每次 forward_step 输出 1 帧
out_frames = 1

# 山洪任务按小时步长推进
freq = getenv_int("FREQ", 1)

depth_list = getenv_str("DEPTH_LIST", "2,2,6,2")
depths = [int(x.strip()) for x in depth_list.split(",") if x.strip()]

dem_source = getenv_str("DEM_SOURCE", "static")


# ---------------------------------------------------------------------
# normalization / denormalization meta
# ---------------------------------------------------------------------
dem_scale = getenv_float("DEM_SCALE", 8500.0)
lai_scale = getenv_float("LAI_SCALE", 20.0)
sand_scale = getenv_float("SAND_SCALE", 100.0)
clay_scale = getenv_float("CLAY_SCALE", 100.0)
som_scale = getenv_float("SOM_SCALE", 100.0)

h_scale = getenv_float("H_SCALE", 30.0)
u_scale = getenv_float("U_SCALE", 12.0)
v_scale = getenv_float("V_SCALE", 12.0)

rain_max_raw = getenv_float("RAIN_MAX_RAW", 250.0)
rain_log_max = math.log1p(rain_max_raw)

# 当前 hydro 任务没有走 BasePredictor 的 mean/std 反归一化链路
buffers = {}


# ---------------------------------------------------------------------
# dataloader / dataset
# ---------------------------------------------------------------------
train_num_workers = getenv_int("TRAIN_NUM_WORKERS", 4)
test_num_workers = getenv_int("TEST_NUM_WORKERS", 2)
pin_memory = getenv_bool("PIN_MEMORY", True)
do_augment = getenv_bool("DO_AUGMENT", False)

use_sqlite_meta = getenv_bool("USE_SQLITE_META", True)
allow_leading_padding = getenv_bool("ALLOW_LEADING_PADDING", True)
consolidated = getenv_bool("CONSOLIDATED", False)

# 评估时默认只推理少量事件，便于调试
# 若想跑全量，请显式传一个更大的值，或把 hydro.py 改成支持 None
test_event_limit = getenv_int("TEST_EVENT_LIMIT", 2)

# 训练集最多返回多少步目标序列
# 注意：这里要和 hydro.py 中的参数名保持一致
train_target_steps = getenv_int("TRAIN_TARGET_STEPS", teach_steps)

dataloader = OmegaConf.create()

# ---------------------------------------------------------------------
# train loader
# ---------------------------------------------------------------------
dataloader.train = L(build_train_loader)(
    dataset=L(DATASETS.get("FloodDataset"))(
        data_dir=data_dir,
        training=True,
        history_steps=history_steps,
        dem_source=dem_source,
        use_sqlite_meta=use_sqlite_meta,
        allow_leading_padding=allow_leading_padding,
        consolidated=consolidated,
        max_train_future_frames=train_target_steps,
    ),
    total_batch_size=total_batch_size,
    num_workers=train_num_workers,
    pin_memory=pin_memory,
    do_augment=do_augment,
)

# ---------------------------------------------------------------------
# test loader
# ---------------------------------------------------------------------
dataloader.test = L(build_test_loader)(
    dataset=L(DATASETS.get("FloodDataset"))(
        data_dir=data_dir,
        training=False,
        history_steps=history_steps,
        dem_source=dem_source,
        use_sqlite_meta=use_sqlite_meta,
        allow_leading_padding=allow_leading_padding,
        consolidated=consolidated,
        test_event_limit=test_event_limit,
    ),
    total_batch_size=total_batch_size,
    num_workers=test_num_workers,
    pin_memory=pin_memory,
)

# ---------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------
dataloader.evaluator = [
    L(METRICS.get("MSE"))(
        test_names=out_names,
        test_times=list(range(1, 25)),
    ),
]