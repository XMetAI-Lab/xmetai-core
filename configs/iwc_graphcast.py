"""
GraphCast-style integrated weather climate configuration (0.25°).
"""

import logging

from iwc_base import (  # noqa: F401
    buffers,
    dataloader,
    depths,
    freq,
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

from xmetai import DATASETS, MODELS
from xmetai.config import LazyCall as L

logging.basicConfig(level=logging.INFO)

train.output_dir = output_dir
train.eval_period = 4000

# Model settings (0.25° grid)
embed_dim = 384
num_pre_layers = depths[0]
num_layers = depths[1]
num_post_layers = depths[2]
image_size = (720, 1440)

edge_trig = True
edge_dim = 7 if edge_trig else 3
static_fields = []
static_levels = {}

model = L(MODELS.get("GraphCastPredictor"))(
    embed_dim=embed_dim,
    num_layers=num_layers,
    num_pre_layers=num_pre_layers,
    num_post_layers=num_post_layers,
    use_multiscale=True,
    coarse_factors=[2, 3],
    num_coarse_layers=depths[3],
    dropout=0.1,
    edge_dim=edge_dim,
    static_dim=6 + len(static_fields),
    image_size=image_size,
    in_chans=in_chans,
    out_chans=out_chans,
    in_frames=in_frames,
    out_frames=1,
    freq=freq,
    accumid=logid,
    logid=logid,
    maskid=maskid,
    zero_accum=True,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=step_range,
    members=1,
    save_dir=output_dir + "/pred/",
)

model.buffers = buffers
model.precision = "bf16"

# Dataset settings
neighbors = 8
periodic_lon = True

# Keep the normal nested dataloader override interface for GraphCast.
dataloader.train.dataset = L(DATASETS.get("GraphCastDataset"))(
    data_paths=dataloader.train.dataset.data_paths,
    hist_frames=in_frames,
    fcst_frames=1,
    interval=1,
    freq=freq,
    training=True,
    years=dataloader.train.dataset.years,
    neighbors=neighbors,
    periodic_lon=periodic_lon,
    edge_trig=edge_trig,
    static_fields=static_fields,
    static_levels=static_levels,
)

dataloader.test.dataset = L(DATASETS.get("GraphCastDataset"))(
    data_paths=dataloader.test.dataset.data_paths,
    hist_frames=in_frames,
    fcst_frames=test_frames,
    interval=16,
    freq=freq,
    training=False,
    inference_only=False,
    years=dataloader.test.dataset.years,
    neighbors=neighbors,
    periodic_lon=periodic_lon,
    edge_trig=edge_trig,
    static_fields=static_fields,
    static_levels=static_levels,
)

if load_from_pretrain:
    dataloader.train.dataset.fcst_frames = train.teacher_forcing.teach_steps
