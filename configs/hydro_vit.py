#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from hydro_base import (
    dataloader,
    in_chans,
    in_frames,
    out_chans,
    out_frames,
    out_names,
    output_dir,
    train,
    optimizer,
    scheduler,
    buffers,
    depths,
    freq,
)

from xmetai import MODELS
from xmetai import LazyCall as L

logging.basicConfig(level=logging.INFO)


def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


# ---------------------------------------------------------------------
# model hyper-parameters
# ---------------------------------------------------------------------
patch_size = getenv_int("PATCH_SIZE", 8)
embed_dim = getenv_int("EMBED_DIM", 128)
num_heads = getenv_int("NUM_HEADS", 4)

image_height = getenv_int("IMAGE_HEIGHT", 1000)
image_width = getenv_int("IMAGE_WIDTH", 1000)
image_size = (image_height, image_width)

drop_rate = getenv_float("DROP_RATE", 0.1)
mlp_ratio = getenv_int("MLP_RATIO", 4)
layer_type = getenv_str("LAYER_TYPE", "mamba").lower().strip()

const_chans = getenv_int("CONST_CHANS", 5)
residual_mode = getenv_str("RESIDUAL_MODE", "none").lower().strip()
precision = getenv_str("PRECISION", "bf16")

# 24-step evaluation
test_frames = list(range(1, 25))
step_range = list(range(1, 25))


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
model = L(MODELS.get("HydroViTModel"))(
    # hydro-specific
    const_chans=const_chans,
    residual_mode=residual_mode,
    rain_ch_idx=0,
    hydro_ch_start=1,

    # ViT backbone
    embed_dim=embed_dim,
    depths=depths,
    num_heads=num_heads,
    patch_size=patch_size,
    image_size=image_size,
    drop_rate=drop_rate,
    mlp_ratio=mlp_ratio,
    layer_type=layer_type,

    # task I/O
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=out_frames,

    # BasePredictor-compatible args
    freq=freq,
    accumid=[],
    logid=[],
    maskid=[],
    zero_accum=False,

    # 24-step evaluation
    test_frames=test_frames,
    test_chans=[0, 1, 2],
    test_names=out_names,
    step_range=step_range,

    members=1,

    # output
    save_dir=os.path.join(output_dir, "pred"),

    # test output control
    save_test_fig=True,
    save_test_nc=False,
    debug_h_scale=30.0,
)

model.precision = precision