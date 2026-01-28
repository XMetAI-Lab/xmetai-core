from xmetai.config import LazyCall as L
from xmetai.core import (
    CircularParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
    WarmupParamScheduler,
)

const_lr = L(ConstantParamScheduler)(value=1)

cosine_lr = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1, end_value=0),
    warmup_length=0.02,
    warmup_factor=0.001,
)

circular_lr = L(CircularParamScheduler)(start_value=1, end_value=0, period=0.1)
