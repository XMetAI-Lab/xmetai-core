from xmetai.config import LazyCall as L
from xmetai.core import (
    CosineParamScheduler,
    WarmupParamScheduler,
    ConstantParamScheduler,
)

const_lr = L(ConstantParamScheduler)(value=1)

cosine_lr = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1, end_value=0),
    warmup_length=0.02,
    warmup_factor=0.001,
)
